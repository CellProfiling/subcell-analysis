"""MLP probes with a hyperparameter grid, cross-validated on frozen embeddings.

The classifier behind GO / Reactome membership prediction (Fig. 3d, Extended Data
Fig. 8c; ``subcell.classification.function_prediction``), the CM4AI drug-prediction
probe (``subcell.perturbation.cm4ai_drug_prediction``) and the interaction
enrichment (``subcell.multimodal.interaction_enrichment``). Library module; no
entry point.

``cross_val_mlp`` scores every (layers, hidden units, dropout) combination of a
grid on one fold, averaging the validation predictions of ten seeds.
``cross_val_mlp_folds`` does the same for several folds at once, training all
replicas of an architecture as one stack of batched matrices on the GPU
(``train_models_batched``); ``train_model`` is the one-replica-at-a-time reference
it reproduces up to dropout draws.
"""

import copy

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from tqdm import tqdm

from subcell.models.mlp import MLPClassifier
from subcell.utils.losses import SigmoidFocalLoss
from subcell.utils.metrics import compute_metrics, get_optimal_threshold
from subcell.utils.seed import set_random_seed


def compute_nested_metrics(y_val, val_pred, test_pred, y_test):
    """`compute_metrics` on the validation set, and on the test set at the
    threshold chosen on the validation set (keys prefixed ``test_``)."""
    threshold = get_optimal_threshold(y_val, val_pred)
    val = compute_metrics(y_val, val_pred, threshold)
    test = compute_metrics(y_test, test_pred, threshold)
    return {**val, **{f"test_{k}": v for k, v in test.items()}}


def train_single_model(train_loader, val_dataset, model, y_true):
    criterion = SigmoidFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_map = 0
    epochs_without_improvement = 0
    best_model = None

    model.to(val_dataset.tensors[0].device)
    for epoch in range(200):
        if epochs_without_improvement >= 20:
            print(
                f"No improvement after 20 epochs, stopping at epoch {epoch}, best map: {best_map}",
                flush=True,
            )
            break

        model.train()
        for _, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        # no_grad only skips building the throw-away autograd graph; the model
        # is in eval mode so dropout draws no RNG and the values are unchanged.
        with torch.no_grad():
            y_pred = F.sigmoid(model(val_dataset.tensors[0])).cpu().numpy()
        val_map = average_precision_score(y_true, y_pred, average="macro")

        if val_map > best_map:
            best_map = val_map
            best_model = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr_scheduler.step(best_map)

    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        y_pred = F.sigmoid(model(val_dataset.tensors[0])).detach().cpu().numpy()

    return y_pred, best_map


def train_model(train_dataset, val_dataset, n_layer, hidden_dim, dropout):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, drop_last=True
    )

    y_true = val_dataset.tensors[1].cpu().numpy()
    all_y_pred = []
    all_val_map = []

    for random_seed in range(10):
        set_random_seed(random_seed)
        model = MLPClassifier(
            input_dim=train_loader.dataset.tensors[0].shape[1],
            output_dim=train_loader.dataset.tensors[1].shape[1],
            hidden_dim=hidden_dim,
            n_layers=n_layer,
            dropout=dropout,
        )

        y_pred, best_map = train_single_model(train_loader, val_dataset, model, y_true)

        all_val_map.append(best_map)
        all_y_pred.append(y_pred)

    y_pred = np.mean(all_y_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred)

    print(
        f"Hidden Dim: {hidden_dim}, Dropout: {dropout}, Number of Layers: {n_layer}",
        f"accuracy: {metrics['acc']} micro_AP: {metrics['micro_ap']}, macro_AP: {metrics['macro_ap']}",
        flush=True,
    )

    return (
        n_layer,
        hidden_dim,
        dropout,
        metrics["acc"],
        metrics["macro_f1"],
        metrics["micro_f1"],
        metrics["macro_ap"],
        metrics["micro_ap"],
        metrics["mlrap"],
    )


def macro_ap(y_true, y_score):
    """``average_precision_score(y_true, y_score[s], average="macro")`` for every
    model ``s`` at once, on the scores' device.

    Tied scores are one threshold, as in sklearn, and a label column without
    positives scores 0, which is what sklearn returns for it (with a warning).
    y_true: (N, C) shared by every model, or (S, N, C); y_score: (S, N, C).
    Rows padded with a score below every real one and no positives do not
    change the result. Returns a float64 (S,) tensor.
    """
    S, N, C = y_score.shape
    scores, order = torch.sort(y_score.transpose(1, 2), dim=-1, descending=True)
    truth = y_true.to(torch.float64).transpose(-1, -2).expand(S, C, N).gather(-1, order)
    tps = truth.cumsum(-1)
    # the last position of each run of equal scores is a threshold
    is_end = torch.ones_like(scores, dtype=torch.bool)
    is_end[..., :-1] = scores[..., :-1] != scores[..., 1:]
    # tps is non-decreasing, so a running max over the thresholds seen so far is
    # the tps of the previous threshold
    prev = torch.cummax(torch.where(is_end, tps, 0.0), dim=-1).values
    prev = torch.cat([torch.zeros_like(prev[..., :1]), prev[..., :-1]], dim=-1)
    rank = torch.arange(1, N + 1, device=tps.device, dtype=torch.float64)
    area = torch.where(is_end, (tps - prev) * tps / rank, 0.0).sum(-1)
    positives = tps[..., -1]
    ap = torch.where(positives > 0, area / positives.clamp(min=1), 0.0)
    return ap.mean(-1)


class StackedMLP:
    """``n`` MLPClassifier replicas held as stacked tensors, so each layer of
    every replica is one batched kernel.

    Built from real MLPClassifier modules, so every replica starts from the
    weights its sequential run would have. Parameters are (n, ...) leaf tensors;
    ``select`` drops replicas, e.g. ones that have stopped early.
    """

    def __init__(self, models, device):
        self.n_layers = models[0].n_layers
        state = [m.state_dict() for m in models]
        stack = lambda k: torch.stack([s[k] for s in state]).to(device)
        hidden = range(self.n_layers) if self.n_layers > 1 else []
        self.linear = [(stack(f"fc{i}.weight"), stack(f"fc{i}.bias")) for i in hidden]
        self.norm = [(stack(f"norm{i}.weight"), stack(f"norm{i}.bias")) for i in hidden]
        self.stats = [
            (stack(f"norm{i}.running_mean"), stack(f"norm{i}.running_var"))
            for i in hidden
        ]
        self.out = (stack("fc_out.weight"), stack("fc_out.bias"))
        for p in self.params():
            p.requires_grad_(True)

    def params(self):
        return [p for pair in self.linear + self.norm + [self.out] for p in pair]

    def set_params(self, params):
        it = iter(params)
        pairs = lambda n: [(next(it), next(it)) for _ in range(n)]
        self.linear = pairs(len(self.linear))
        self.norm = pairs(len(self.norm))
        self.out = (next(it), next(it))

    def select(self, keep):
        self.set_params([p.detach()[keep].requires_grad_(True) for p in self.params()])
        self.stats = [(m[keep], v[keep]) for m, v in self.stats]

    def forward(self, x, drop=None, rows=slice(None)):
        """x: (n, B, in) for the replicas in `rows`. `drop(x)` applies training
        dropout (all replicas only); None means eval."""
        training = drop is not None
        if training:
            x = drop(x)
        for (w, b), (g, beta), (mean, var) in zip(self.linear, self.norm, self.stats):
            w, b, g, beta, mean, var = (t[rows] for t in (w, b, g, beta, mean, var))
            x = torch.baddbmm(b.unsqueeze(1), x, w.transpose(1, 2))
            if training:
                x = drop(x)
            x = F.relu(x)
            # BatchNorm1d per replica: with the replica axis folded into the
            # channels, each (replica, unit) is one channel normalized over the
            # batch, and the running statistics update in place through the views
            n, B, H = x.shape
            x = F.batch_norm(
                x.transpose(0, 1).reshape(B, n * H),
                mean.reshape(-1), var.reshape(-1), g.reshape(-1), beta.reshape(-1),
                training=training, momentum=0.1, eps=1e-5,
            ).reshape(B, n, H).transpose(0, 1)
        w, b = (t[rows] for t in self.out)
        return torch.baddbmm(b.unsqueeze(1), x, w.transpose(1, 2))


def replica_bytes(n_layer, hidden_dim, input_dim, output_dim, batch_size=512):
    """Rough GPU memory one replica needs while training: parameters with their
    gradients and AdamW moments, plus a step's activations."""
    hidden = [hidden_dim] * (n_layer if n_layer > 1 else 0)
    dims = [input_dim] + hidden + [output_dim]
    n_params = sum(a * b + b for a, b in zip(dims[:-1], dims[1:])) + 2 * sum(hidden)
    activations = batch_size * (4 * input_dim + 10 * sum(hidden) + 4 * output_dim)
    return 4 * (4 * n_params + activations)


def _train_stack(tasks, n_layer, hidden_dim, dropouts, seeds):
    """`train_model` for every (task, dropout, seed) at once, as one StackedMLP.

    A task is one cross-validation fold: (train_dataset, val_dataset), all with
    the same feature and label dimensions and the same number of 512-row batches
    per epoch, optionally with a test_dataset, which each replica scores at its
    best validation epoch. Each replica keeps what its sequential run gives it: the initial
    weights of its seed, the shuffle order of its seed on its fold (the two draws
    per epoch a DataLoader makes on the CPU generator, replayed from a per-(fold,
    seed) copy of that generator), its own AdamW state, ReduceLROnPlateau
    schedule and early stop. Dropout masks are drawn for all replicas in one
    call, so they are fresh draws rather than the sequential run's; the
    per-epoch validation macro AP is `macro_ap`.
    """
    device = tasks[0][1].tensors[0].device
    batch_size = 512
    X_train = torch.cat([tr.tensors[0] for tr, *_ in tasks])
    y_train = torch.cat([tr.tensors[1] for tr, *_ in tasks])
    n_train = [len(tr.tensors[0]) for tr, *_ in tasks]
    offset = np.cumsum([0] + n_train[:-1]).tolist()
    n_batches = n_train[0] // batch_size
    assert all(n // batch_size == n_batches for n in n_train)
    # validation folds padded to one length: a padded row scores -1, below any
    # sigmoid output, and has no positives, so it leaves the macro AP unchanged
    n_val = [len(va.tensors[0]) for _, va, *_ in tasks]
    n_max = max(n_val)
    D, C = X_train.shape[1], y_train.shape[1]
    X_val = torch.zeros(len(tasks), n_max, D, device=device)
    y_val = torch.zeros(len(tasks), n_max, C, dtype=y_train.dtype, device=device)
    valid = torch.zeros(len(tasks), n_max, dtype=torch.bool, device=device)
    for t, (_, va, *_) in enumerate(tasks):
        X_val[t, : n_val[t]], y_val[t, : n_val[t]] = va.tensors
        valid[t, : n_val[t]] = True
    test = [task[2].tensors[0] for task in tasks] if len(tasks[0]) == 3 else None
    if test is not None:
        n_test = [len(x) for x in test]
        X_test = torch.zeros(len(tasks), max(n_test), D, device=device)
        for t, x in enumerate(test):
            X_test[t, : n_test[t]] = x
    criterion = SigmoidFocalLoss(reduction="none")
    lr0, betas, eps, weight_decay = 0.01, (0.9, 0.999), 1e-8, 0.01

    seeds, dropouts = list(seeds), list(dropouts)
    models, init_state = [], []
    for seed in seeds:
        set_random_seed(seed)
        # dropout has no parameters, so the initial weights (and the generator
        # state after them) are the same for every dropout rate and fold
        models.append(
            MLPClassifier(
                input_dim=D, output_dim=C, hidden_dim=hidden_dim, n_layers=n_layer,
                dropout=0.0,
            )
        )
        init_state.append(torch.get_rng_state())
    cpu_gens = {}
    for t in range(len(tasks)):
        for s in range(len(seeds)):
            cpu_gens[t, s] = torch.Generator()
            cpu_gens[t, s].set_state(init_state[s])

    # replica r trains task t with dropouts[d] and seeds[s]
    replicas = [
        (t, d, s)
        for t in range(len(tasks))
        for d in range(len(dropouts))
        for s in range(len(seeds))
    ]
    n_replicas = len(replicas)
    net = StackedMLP([models[s] for _, _, s in replicas], device)
    rate = torch.tensor(
        [dropouts[d] for _, d, _ in replicas], dtype=torch.float32, device=device
    )
    task_of = torch.tensor([t for t, _, _ in replicas], device=device)
    schedulers = []
    for _ in replicas:
        # a stand-in optimizer, so the schedule is torch's own ReduceLROnPlateau
        dummy = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=lr0)
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                dummy, mode="max", factor=0.5, patience=5
            )
        )
    exp_avg = [torch.zeros_like(p) for p in net.params()]
    exp_avg_sq = [torch.zeros_like(p) for p in net.params()]
    step = 0

    active = torch.arange(n_replicas)
    best_map = [0.0] * n_replicas
    stale = [0] * n_replicas
    best_pred = torch.zeros(n_replicas, n_max, C, device=device)
    if test is not None:
        best_test = torch.zeros(n_replicas, max(n_test), C, device=device)
    # replicas scored per validation chunk, bounding the (chunk, n_max, D) input
    chunk = max(1, int(2**30 // (4 * n_max * max(D, hidden_dim))))

    def drop(x):
        p = rate[active_dev].view(-1, *([1] * (x.dim() - 1)))
        return x * (torch.rand_like(x) >= p) / (1 - p)

    for epoch in range(200):
        keep = [j for j, i in enumerate(active.tolist()) if stale[i] < 20]
        if not keep:
            break
        if len(keep) < len(active):
            net.select(keep)
            exp_avg = [m[keep] for m in exp_avg]
            exp_avg_sq = [v[keep] for v in exp_avg_sq]
            active = active[keep]
        active_dev = active.to(device)
        n_active = len(active)

        # one permutation per (fold, seed) that still has a replica training
        pairs = sorted({(replicas[i][0], replicas[i][2]) for i in active.tolist()})
        perms = {}
        for t, s in pairs:
            torch.empty((), dtype=torch.int64).random_(generator=cpu_gens[t, s])
            draw = int(torch.empty((), dtype=torch.int64).random_(generator=cpu_gens[t, s]).item())
            perm = torch.randperm(n_train[t], generator=torch.Generator().manual_seed(draw))
            perms[t, s] = perm[: n_batches * batch_size] + offset[t]
        order = torch.stack(
            [perms[replicas[i][0], replicas[i][2]] for i in active.tolist()]
        )
        batches = order.view(n_active, n_batches, batch_size).to(device)

        # AdamW, as torch.optim.AdamW(lr, betas, eps, weight_decay), with a
        # learning rate per replica. The per-replica scalars are formed in float64
        # and rounded once, as torch rounds its Python-float scalars.
        lrs = [schedulers[i].optimizer.param_groups[0]["lr"] for i in active.tolist()]
        decay = torch.tensor([1 - lr * weight_decay for lr in lrs], device=device)
        neg_lr = torch.tensor([-lr for lr in lrs], dtype=torch.float64, device=device)
        shape = lambda p: (n_active,) + (1,) * (p.dim() - 1)
        for b in range(n_batches):
            idx = batches[:, b]
            loss = criterion(net.forward(X_train[idx], drop), y_train[idx])
            params = net.params()
            grads = torch.autograd.grad(loss.mean(dim=(1, 2)).sum(), params)

            step += 1
            bias2_sqrt = (1 - betas[1] ** step) ** 0.5
            step_size = (neg_lr / (1 - betas[0] ** step)).float()
            with torch.no_grad():
                for p, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
                    p.mul_(decay.view(shape(p)))
                    m.lerp_(g, 1 - betas[0])
                    v.mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
                    denom = (v.sqrt() / bias2_sqrt).add_(eps)
                    p.addcmul_(m / denom, step_size.view(shape(p)))

        with torch.no_grad():
            maps, improved = [], []
            for j0 in range(0, n_active, chunk):
                rows = slice(j0, min(j0 + chunk, n_active))
                tasks_here = task_of[active_dev[rows]]
                scores = torch.sigmoid(net.forward(X_val[tasks_here], rows=rows))
                scores = scores.masked_fill(~valid[tasks_here].unsqueeze(-1), -1.0)
                chunk_maps = macro_ap(y_val[tasks_here], scores).tolist()
                for jj, value in enumerate(chunk_maps):
                    i = int(active[j0 + jj])
                    if value > best_map[i]:
                        best_map[i] = value
                        stale[i] = 0
                        improved.append(jj)
                    else:
                        stale[i] += 1
                    schedulers[i].step(best_map[i])
                if improved:
                    sel = torch.tensor(improved, device=device)
                    best_pred[active_dev[rows][sel]] = scores[sel]
                    if test is not None:
                        pos = torch.arange(rows.start, rows.stop, device=device)[sel]
                        best_test[active_dev[pos]] = torch.sigmoid(
                            net.forward(X_test[task_of[active_dev[pos]]], rows=pos)
                        )
                    improved = []

    best_pred = best_pred.cpu().numpy()
    if test is not None:
        best_test = best_test.cpu().numpy()
    n_seeds = len(seeds)
    out = []
    for t in range(len(tasks)):
        by_dropout = {}
        for d, dropout in enumerate(dropouts):
            first = (t * len(dropouts) + d) * n_seeds
            by_dropout[dropout] = np.mean(
                [best_pred[first + s, : n_val[t]] for s in range(n_seeds)], axis=0
            )
            if test is not None:
                by_dropout[dropout] = (
                    by_dropout[dropout],
                    np.mean(
                        [best_test[first + s, : n_test[t]] for s in range(n_seeds)],
                        axis=0,
                    ),
                )
        out.append(by_dropout)
    return out


def train_models_batched(
    tasks, n_layer, hidden_dim, dropouts, seeds=range(10), max_bytes=None
):
    """`_train_stack` over `tasks`, stacking as many folds together as fit in
    `max_bytes` of GPU memory (folds with a different number of batches per
    epoch go in separate stacks). Returns one {dropout: predictions} per task,
    the predictions being (validation, test) when the tasks have a test set.
    """
    D = tasks[0][0].tensors[0].shape[1]
    C = tasks[0][0].tensors[1].shape[1]
    per_task = len(list(dropouts)) * len(list(seeds)) * replica_bytes(n_layer, hidden_dim, D, C)
    fit = len(tasks) if max_bytes is None else max(1, int(max_bytes // per_task))
    groups = {}
    for t, (tr, *_) in enumerate(tasks):
        groups.setdefault(len(tr.tensors[0]) // 512, []).append(t)
    out = [None] * len(tasks)
    for members in groups.values():
        for k in range(0, len(members), fit):
            part = members[k : k + fit]
            for t, res in zip(part, _train_stack(
                [tasks[t] for t in part], n_layer, hidden_dim, dropouts, seeds
            )):
                out[t] = res
    return out


METRIC_NAMES = ["acc", "macro_f1", "micro_f1", "macro_ap", "micro_ap", "mlrap"]
RESULT_COLUMNS = ["n_layers", "hidden_dim", "dropout"] + METRIC_NAMES


def cross_val_mlp_folds(folds, parameters, n_jobs=-1, max_bytes=None):
    """`cross_val_mlp(..., batched=True)` for several folds at once.

    folds: [(X_train, y_train, X_test, y_test), ...] with the same feature and
    label dimensions, or [(X_train, y_train, X_val, y_val, X_test, y_test), ...]
    to select on the validation set and also score the test set at each
    replica's best validation epoch and the validation-chosen threshold
    (``compute_nested_metrics``, extra ``test_`` columns). Each architecture (n_layers, hidden_dim) is trained once
    for every fold, dropout rate and seed (`train_models_batched`), and `n_jobs`
    workers score the predictions with sklearn while the next architecture
    trains. A one-layer MLP has no hidden layer, so ignores hidden_dim; it is
    trained once and its row repeated for each hidden_dim (the sequential path
    computes those repeats and gets the same numbers). Returns one results frame
    per fold, rows in `cross_val_mlp` order.
    """
    from joblib.externals.loky import get_reusable_executor

    tasks = [
        tuple(
            torch.utils.data.TensorDataset(*f[i : i + 2]) for i in range(0, len(f), 2)
        )
        for f in folds
    ]
    combos = [
        (n_layer, hidden_dim, dropout)
        for hidden_dim in parameters["hidden_dim"]
        for dropout in parameters["dropout"]
        for n_layer in parameters["n_layers"]
    ]
    first_hidden = parameters["hidden_dim"][0]
    canonical = lambda c: (c[0], first_hidden, c[2]) if c[0] == 1 else c
    archs = list(dict.fromkeys(canonical(c)[:2] for c in combos))
    y_true = [f[3].cpu().numpy() for f in folds]
    y_test = [f[5].cpu().numpy() for f in folds] if len(folds[0]) == 6 else None
    names = METRIC_NAMES if y_test is None else METRIC_NAMES + [
        f"test_{n}" for n in METRIC_NAMES
    ]

    pool = get_reusable_executor(max_workers=max(1, n_jobs))
    scored = [{} for _ in folds]
    for n_layer, hidden_dim in tqdm(archs):
        per_task = train_models_batched(
            tasks, n_layer, hidden_dim, parameters["dropout"], max_bytes=max_bytes
        )
        for t, by_dropout in enumerate(per_task):
            for dropout, y_pred in by_dropout.items():
                scored[t][(n_layer, hidden_dim, dropout)] = (
                    pool.submit(compute_metrics, y_true[t], y_pred)
                    if y_test is None
                    else pool.submit(
                        compute_nested_metrics, y_true[t], *y_pred, y_test[t]
                    )
                )
    frames = []
    for t in range(len(folds)):
        metrics = {k: f.result() for k, f in scored[t].items()}
        rows = [c + tuple(metrics[canonical(c)][n] for n in names) for c in combos]
        frames.append(pd.DataFrame(rows, columns=RESULT_COLUMNS[:3] + names))
    return frames


def cross_val_mlp(
    X_train, y_train, X_test, y_test, parameters, n_jobs=-1, batched=False
):
    """Every hyperparameter combination, trained with `train_model`, or (with
    `batched`) with `cross_val_mlp_folds` for this one fold."""
    if batched:
        return cross_val_mlp_folds(
            [(X_train, y_train, X_test, y_test)], parameters, n_jobs=n_jobs
        )[0]

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    all_hyperparam_combinations = []
    for hidden_dim in parameters["hidden_dim"]:
        for dropout in parameters["dropout"]:
            for n_layer in parameters["n_layers"]:
                all_hyperparam_combinations.append((n_layer, hidden_dim, dropout))

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_model)(train_dataset, val_dataset, *params)
        for params in tqdm(all_hyperparam_combinations)
    )
    return pd.DataFrame(results, columns=RESULT_COLUMNS)
