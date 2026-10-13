"""Build an encoder from a ``model`` block of a file in ``configs/`` and load its weights.

Each ``get_*_model`` takes that block (``args``, ``weight_path``, and for Cell-DINO
``repo_path`` and ``entrypoint``, with ``${NAME}`` roots already expanded by
``config.load_yaml``) and returns the model, ready to be put in ``eval`` mode.
``subcell.utils.inference.get_model`` dispatches on the config ``name``.
"""
import sys

import numpy as np
import torch

from subcell.models.cls_inception_v3 import InceptionV3
from subcell.models.cell_dino import CellDINO
from subcell.models.vision_transformer import VisionTransformer
from subcell.models.vit_pool import ViTPoolModel


def remove_prefix(load_state_dict, name):
    return {
        (key.replace(name, "") if key.startswith(name) else key): value
        for key, value in load_state_dict.items()
    }


def load_pretrained_state_dict(net, load_state_dict, strict=False, can_print=True):
    """Load a checkpoint into *net*, skipping keys the model does not have and
    copying the overlapping part of parameters whose size differs (same rank)."""
    if "epoch" in load_state_dict and can_print:
        print(f"load epoch:{load_state_dict['epoch']:.2f}")
    for key in ("state_dict", "model_state_dict", "model"):
        if key in load_state_dict:
            load_state_dict = load_state_dict[key]
            break
    state_dict = net.state_dict()

    load_state_dict = remove_prefix(load_state_dict, "module.")
    load_state_dict = remove_prefix(load_state_dict, "base_model.")

    skipped = []
    for key in list(load_state_dict.keys()):
        if key not in state_dict:
            if strict:
                raise Exception(f"not in {key}")
            skipped.append(key)
            continue
        if load_state_dict[key].size() != state_dict[key].size():
            if strict or (
                len(load_state_dict[key].size()) != len(state_dict[key].size())
            ):
                raise Exception(
                    f"size not the same {key}: {load_state_dict[key].size()} -> {state_dict[key].size()}"
                )
            state_slice = tuple(
                slice(s)
                for s in np.minimum(
                    np.array(load_state_dict[key].size()),
                    np.array(state_dict[key].size()),
                )
            )
            state_dict[key][state_slice] = load_state_dict[key][state_slice]
            continue
        state_dict[key] = load_state_dict[key]

    if skipped and can_print:
        print(f"{len(skipped)} checkpoint keys are not in the model")
    net.load_state_dict(state_dict)
    return net


def load_pretrained(net, pretrained_file, strict=False, can_print=False):
    load_state_dict = torch.load(pretrained_file, map_location=torch.device("cpu"))
    return load_pretrained_state_dict(
        net, load_state_dict, strict=strict, can_print=can_print
    )


def get_bestfitting_model(config):
    model = InceptionV3(config["args"], att_type="cbam")
    return load_pretrained(model, config["weight_path"], strict=False, can_print=True)


def get_dino_model(config):
    model = VisionTransformer(**config["args"])
    state_dict = torch.load(
        config["weight_path"], map_location="cpu", weights_only=False
    )
    teacher = state_dict["teacher"]
    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
    teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}

    # the teacher checkpoint also holds the DINO projection head, which is not loaded
    msg = model.load_state_dict(teacher, strict=False)
    if msg.missing_keys:
        print(f"missing keys: {msg.missing_keys}")

    return model


def get_cell_dino_model(config):
    # Cell-DINO is defined in the dinov2 repository (``config["repo_path"]``)
    sys.path.insert(0, config["repo_path"])
    from dinov2.hub.cell_dino import backbones

    backbone = getattr(backbones, config["entrypoint"])(
        pretrained_path=config["weight_path"], in_channels=4
    )
    return CellDINO(backbone, output_attentions=config.get("output_attentions", False))


def get_subcell_model(config):
    model = ViTPoolModel(
        config["args"]["vit_model"],
        config["args"]["pool_model"],
        output_attentions=config.get("output_attentions", True),
    )
    state_dict = torch.load(
        config["weight_path"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    return model
