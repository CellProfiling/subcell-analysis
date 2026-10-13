"""Base class of the MIRAGE model: devices, schedulers, losses and checkpoints."""

# Adapted from pytorch-CycleGAN-and-pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park, BSD 2-Clause licence.

import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from . import networks


class BaseModel(ABC):
    """Abstract base class for models.

    A subclass implements ``__init__`` (calling ``BaseModel.__init__`` first, then
    defining ``loss_names``, ``model_names`` and appending to ``optimizers``),
    ``set_input``, ``forward`` and ``optimize_parameters``.
    """

    def __init__(self, config):
        """
        Parameters:
            config -- experiment settings (attribute access), see ``subcell.multimodal.mirage``
        """
        self.config = config
        # no CUDA (or no GPU ids): run on the CPU, with the networks built for it
        use_gpu = bool(self.config.gpu_ids) and torch.cuda.is_available()
        self.gpu_ids = list(self.config.gpu_ids) if use_gpu else []
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}" if use_gpu else "cpu")
        self.save_dir = config.checkpoints_dir  # save all the checkpoints to save_dir

        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): the batch, one tensor per modality.
        """

    @abstractmethod
    def forward(self):
        """Run forward pass; called by ``optimize_parameters``."""

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

    def setup(self, opt):
        """Create the schedulers and print the networks.

        Parameters:
            opt -- experiment settings
        """
        self.schedulers = [
            networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
        ]
        self.print_networks(self.config.verbose)

    def eval(self):
        """Put the networks in eval mode, for inference."""
        for name in self.model_names:
            getattr(self, name).eval()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the beginning of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.config.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate %.7f -> %.7f" % (old_lr, lr))
        return lr

    def get_current_losses(self):
        """Return the training losses named in ``loss_names``, as floats."""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            save_filename = "%s_net_%s.pth" % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)

            torch.save(net.cpu().state_dict(), save_path)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                net.cuda(self.gpu_ids[0])

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            net = getattr(self, name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print(
                "[Network %s] Total number of parameters : %.3f M"
                % (name, num_params / 1e6)
            )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all parameters of the given networks

        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
