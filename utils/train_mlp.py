import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


def balanced_weights(train_y):
    freq = train_y.sum(axis=0) / train_y.shape[0]
    weights_per_class = 1 / freq

    weights = []
    for idx in range(train_y.shape[0]):
        idx_label = train_y[idx].bool()
        weights.append(weights_per_class[idx_label].max())
    weights = torch.tensor(weights)
    return weights


def train_mlp(
    train_x, train_y, val_x, val_y, test_x, test_y, device, unique_cats, save_folder
):

    model_path = f"{save_folder}/mlp_best_map_final.pth"

    train_dataloader = DataLoader(
        TensorDataset(
            train_x.float().to(device),
            train_y.float().to(device),
        ),
        batch_size=8192,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        TensorDataset(
            val_x.float().to(device),
            val_y.float().to(device),
        ),
        batch_size=8192,
    )
    test_dataloader = DataLoader(
        TensorDataset(
            test_x.float().to(device),
            test_y.float().to(device),
        ),
        batch_size=8192,
    )

    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(unique_cats)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5
    )

    criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    best_val_map = 0
    epochs_without_improvement = 0

    for epoch in range(1, 201):
        if epochs_without_improvement >= 20:
            print("No improvement after 20 epochs, stopping")
            break

        _train_epoch(model, train_dataloader, optimizer, criterion, epoch)
        val_map = _val_epoch(model, val_dataloader, criterion, epoch)
        lr_scheduler.step(val_map)
        print(f"Epoch: {epoch} LR: {optimizer.param_groups[0]['lr']}", flush=True)

        if val_map >= best_val_map:
            best_val_map = val_map
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_folder}/mlp_best_map.pth")
        else:
            epochs_without_improvement += 1
            print(
                f"Epochs without improvement: {epochs_without_improvement}, Best Val MAP: {best_val_map}",
                flush=True,
            )

    model.load_state_dict(torch.load(f"{save_folder}/mlp_best_map.pth"))
    torch.save(model.state_dict(), model_path)
    return model


def _train_epoch(model, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch} Batch: {i+1} Loss: {loss.item():.5f}", flush=True)


def _val_epoch(model, val_dataloader, criterion, epoch):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        val_loss = 0
        for i, (x, y) in enumerate(val_dataloader):
            out = model(x)
            loss = criterion(out, y)
            out_onehot = F.sigmoid(out)
            y_pred.append(out_onehot.cpu())
            y_true.append(y.cpu())

            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()

        mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")

    print(
        f"Epoch: {epoch} Val Loss: {val_loss:.5f} Mean Avg Precision: {mean_avg_precision:.5f}",
        flush=True,
    )
    return mean_avg_precision


def _test_epoch(model, test_dataloader, unique_cats):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        for i, (x, y) in enumerate(test_dataloader):
            out = model(x)
            out_sigmoid = F.sigmoid(out)
            y_pred.append(out_sigmoid.cpu())
            y_true.append(y.cpu())

        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()

        mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")

    print(f"Mean Avg Precision: {mean_avg_precision:.5f}")

    df_pred = pd.DataFrame(y_pred, columns=unique_cats)
    df_true = pd.DataFrame(y_true, columns=unique_cats)

    df = pd.merge(
        df_true,
        df_pred,
        suffixes=("_true", "_pred"),
        left_index=True,
        right_index=True,
    )
    return df


def eval_model(x, y, unique_cats, model, seed, device):
    dataloader = DataLoader(
        TensorDataset(
            x.float().to(device),
            y.float().to(device),
        ),
        batch_size=16384,
    )
    df_res = _test_epoch(model, dataloader, unique_cats)
    df_res = df_res.reindex(sorted(df_res.columns), axis=1)
    df_res["Seed"] = seed
    return df_res
