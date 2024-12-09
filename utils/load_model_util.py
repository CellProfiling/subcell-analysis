import sys
from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from models.cls_inception_v3 import InceptionV3
from models.vision_transformer import VisionTransformer
from models.vit_pool import ViTPoolModel


def remove_prefix(load_state_dict, name):
    new_load_state_dict = dict()
    for key in load_state_dict.keys():
        if key.startswith(name):
            dst_key = key.replace(name, "")
        else:
            dst_key = key
        new_load_state_dict[dst_key] = load_state_dict[key]
    load_state_dict = new_load_state_dict
    return load_state_dict


# load_pretrained ------------------------------------
def load_pretrained_state_dict(net, load_state_dict, strict=False, can_print=True):
    if "epoch" in load_state_dict and can_print:
        epoch = load_state_dict["epoch"]
        print(f"load epoch:{epoch:.2f}")
    if "state_dict" in load_state_dict:
        load_state_dict = load_state_dict["state_dict"]
    elif "model_state_dict" in load_state_dict:
        load_state_dict = load_state_dict["model_state_dict"]
    elif "model" in load_state_dict:
        load_state_dict = load_state_dict["model"]
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    load_state_dict = remove_prefix(load_state_dict, "module.")
    load_state_dict = remove_prefix(load_state_dict, "base_model.")

    for key in list(load_state_dict.keys()):
        if key not in state_dict:
            if strict:
                raise Exception(f"not in {key}")
            if can_print:
                print("not in", key)
            continue
        if load_state_dict[key].size() != state_dict[key].size():
            if strict or (
                len(load_state_dict[key].size()) != len(state_dict[key].size())
            ):
                raise Exception(
                    f"size not the same {key}: {load_state_dict[key].size()} -> {state_dict[key].size()}"
                )
            if can_print:
                print(
                    f"{key} {load_state_dict[key].size()} -> {state_dict[key].size()}"
                )
            state_slice = [
                slice(s)
                for s in np.minimum(
                    np.array(load_state_dict[key].size()),
                    np.array(state_dict[key].size()),
                )
            ]
            state_dict[key][state_slice] = load_state_dict[key][state_slice]
            continue
        state_dict[key] = load_state_dict[key]

    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net.module.load_state_dict(state_dict)
    else:
        msg = net.load_state_dict(state_dict)
        if can_print:
            print(msg)
    return net


def load_pretrained(net, pretrained_file, strict=False, can_print=False):
    if can_print:
        print(f"load pretrained file: {pretrained_file}")
    load_state_dict = torch.load(pretrained_file, map_location=torch.device("cpu"))
    net = load_pretrained_state_dict(
        net, load_state_dict, strict=strict, can_print=can_print
    )
    return net


def get_bestfitting_model(config):
    model = InceptionV3(config["args"], feature_net="inception_v3", att_type="cbam")
    weight_path = config.get("weight_path")
    model = load_pretrained(model, weight_path, strict=False, can_print=True)
    return model


def get_dino_model(config):
    model = VisionTransformer(**config["args"])
    state_dict = torch.load(config["weight_path"], map_location="cpu")
    teacher = state_dict["teacher"]
    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
    teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}

    msg = model.load_state_dict(teacher, strict=False)
    print(msg)

    return model


def get_subcell_model(config):
    model = ViTPoolModel(config["args"]["vit_model"], config["args"]["pool_model"])
    state_dict = torch.load(config["weight_path"], map_location="cpu")

    msg = model.load_state_dict(state_dict)
    print(msg)
    return model
