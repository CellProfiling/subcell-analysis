"""MIRAGE: adversarial co-embedding of several modalities of the same proteins.

Each modality has an encoder to a shared-size, l2-normalised latent space, a decoder
back to the modality, and a discriminator. A latent code is decoded into another
modality's space (a "fake" sample the discriminator must accept) and re-encoded, and
the objective combines the GAN loss with a reconstruction ("short cycle") and a
latent ("latent cycle") consistency term. Used for the MIRAGE columns of Extended
Data Fig. 7b; Nasser et al., Cell Syst. 16, 101444 (2025).
"""

# Adapted from pytorch-CycleGAN-and-pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park, BSD 2-Clause licence.

import itertools
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import loss, networks
from .base_model import BaseModel
from .image_pool import ImagePool


class GanToTensor:
    """Convert an ndarray to a float tensor."""

    def __call__(self, sample):
        return torch.tensor(sample).float()


class AlignedDataset(Dataset):
    def __init__(self, config):
        self.file_names = config.input_files
        self.modalities = config.modalities
        self.transform = GanToTensor()
        self.dataframes = []
        self.overlapping_indices = None
        self.load_dataframes()

    def load_dataframes(self):
        self.dims = {}
        all_indices = set()
        first_iteration = True

        for i, file in enumerate(self.file_names):
            df = pd.read_table(
                file,
                sep="\t",
                index_col=0,
            )
            self.dataframes.append(df)
            self.dims[self.modalities[i]] = df.shape[1]

            # Find overlapping indices
            if first_iteration:
                all_indices = set(df.index)
                first_iteration = False
            else:
                all_indices = all_indices.intersection(set(df.index))

        # Convert to list and sort for consistency
        self.overlapping_indices = sorted(list(all_indices))

        # Filter dataframes to keep only overlapping indices
        for i in range(len(self.dataframes)):
            self.dataframes[i] = self.dataframes[i].loc[self.overlapping_indices]

        self.size = len(self.overlapping_indices)

    def __getitem__(self, item):
        data = {}
        for modality, df in zip(self.modalities, self.dataframes):
            data[modality] = self.transform(df.iloc[item].values)

        return data

    def __len__(self):
        return self.size


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        latent_dim,
        gpu_ids=[],
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )
        self.encoder = networks.init_net(self.encoder, init_type, init_gain, gpu_ids=gpu_ids)

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_size,
        output_size,
        gpu_ids=[],
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = networks.init_net(self.decoder, init_type, init_gain, gpu_ids=gpu_ids)

    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(
        self, input_size, hidden_size, gpu_ids=[], init_type="normal", init_gain=0.02
    ):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.discriminator = networks.init_net(
            self.discriminator, init_type, init_gain, gpu_ids=gpu_ids
        )

    def forward(self, x):
        return self.discriminator(x)


class MirageModel(BaseModel):
    """Encoders, decoders and discriminators for each modality, trained jointly.

    See the module docstring. ``config`` needs ``modalities``, ``dims`` (input
    dimension per modality), ``hidden_dim``, ``latent_dim``, ``gpu_ids``,
    ``init_type``, ``pool_size``, ``gan_mode``, the optimizer and loss-weight
    settings of ``subcell.multimodal.mirage.DEFAULT_CONFIG`` and ``checkpoints_dir``.
    """

    def __init__(self, config):
        BaseModel.__init__(self, config)
        # an encoder for each modality
        self.modalities = config.modalities
        for modality in self.modalities:
            setattr(
                self,
                f"enc_{modality}",
                Encoder(
                    self.config.dims[modality],
                    self.config.hidden_dim,
                    self.config.latent_dim,
                    self.gpu_ids,
                    init_type=self.config.init_type,
                ),
            )
        # and a decoder
        for modality in self.modalities:
            setattr(
                self,
                f"dec_{modality}",
                Decoder(
                    self.config.latent_dim,
                    self.config.hidden_dim,
                    self.config.dims[modality],
                    self.gpu_ids,
                    init_type=self.config.init_type,
                ),
            )

        # training losses reported by <BaseModel.get_current_losses>
        self.loss_names = (
            [f"G_{modality}" for modality in self.modalities]
            + [f"short_cycle_{modality}" for modality in self.modalities]
            + [f"latent_cycle_{modality}" for modality in self.modalities]
            + [f"D_{modality}" for modality in self.modalities]
        )

        self.model_names = (
            [f"enc_{modality}" for modality in self.modalities]
            + [f"dec_{modality}" for modality in self.modalities]
            + [f"disc_{modality}" for modality in self.modalities]
        )

        for modality in self.modalities:
            setattr(
                self,
                f"disc_{modality}",
                Discriminator(
                    self.config.dims[modality],
                    self.config.hidden_dim,
                    self.gpu_ids,
                    init_type=self.config.init_type,
                ),
            )

        # buffers of previously generated samples for the discriminators
        for modality in self.modalities:
            setattr(self, f"fake_{modality}_pool", ImagePool(self.config.pool_size))

        # define loss functions
        self.criterionGAN = loss.GANLoss(self.config.gan_mode).to(self.device)
        self.criterionCycle = torch.nn.MSELoss()
        # optimizers; the schedulers are created by <BaseModel.setup>
        gen_parameters = [
            getattr(self, f"enc_{modality}").parameters()
            for modality in self.modalities
        ] + [
            getattr(self, f"dec_{modality}").parameters()
            for modality in self.modalities
        ]
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(*gen_parameters),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
        )

        disc_parameters = [
            getattr(self, f"disc_{modality}").parameters()
            for modality in self.modalities
        ]
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(*disc_parameters),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
        )

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Move the batch (one tensor per modality) to the device."""
        for modality in self.modalities:
            setattr(
                self,
                f"real_{modality}",
                input[modality].to(self.device),
            )

    def forward(self):
        """Encode every modality, decode it into itself and into another modality, and re-encode."""
        target_pool = self.modalities.copy()
        for i, modality in enumerate(self.modalities):
            real_data = getattr(self, f"real_{modality}")
            enc = getattr(self, f"enc_{modality}")
            dec = getattr(self, f"dec_{modality}")

            latent = enc(real_data)
            setattr(self, f"latent_{modality}", latent)
            # Short reconstruction
            short_rec = dec(latent)
            setattr(self, f"short_rec_{modality}", short_rec)

            # fake + full reconstruction
            target = random.choice(target_pool)
            target_pool.remove(target)
            tar_enc = getattr(self, f"enc_{target}")
            tar_dec = getattr(self, f"dec_{target}")
            fake = tar_dec(latent)
            setattr(self, f"fake_{target}", fake)
            setattr(self, f"latent_rec_{modality}", tar_enc(fake))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real samples
            fake (tensor array) -- samples generated by a decoder

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Gradient penalty https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf#page=7.39
        loss_gp = 0.0
        if self.config.lambda_gp > 0:
            loss_gp, _ = networks.cal_gradient_penalty(
                netD,
                real,
                fake.detach(),
                device=self.device,
                type=self.config.gp_type,
                constant=self.config.gp_constant,
                lambda_gp=self.config.lambda_gp,
            )

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + loss_gp
        loss_D.backward()
        return loss_D

    def backward_D_ALL(self):
        """Calculate GAN loss for all discriminators"""
        for modality in self.modalities:
            real = getattr(self, f"real_{modality}")
            disc = getattr(self, f"disc_{modality}")
            fake = getattr(self, f"fake_{modality}")
            fake = getattr(self, f"fake_{modality}_pool").query(fake)
            loss_D = self.backward_D_basic(disc, real, fake)
            setattr(
                self,
                f"loss_D_{modality}",
                loss_D,
            )

    def backward_G(self):
        """Calculate the losses of the encoders and decoders, and their gradients"""
        for modality in self.modalities:
            real = getattr(self, f"real_{modality}")
            latent = getattr(self, f"latent_{modality}")
            short_rec = getattr(self, f"short_rec_{modality}")
            latent_rec = getattr(self, f"latent_rec_{modality}")
            fake = getattr(self, f"fake_{modality}")
            disc = getattr(self, f"disc_{modality}")
            # GAN loss
            setattr(
                self,
                f"loss_G_{modality}",
                self.criterionGAN(disc(fake), True),
            )
            # short cycle loss || Dec(Enc(x)) - x ||
            setattr(
                self,
                f"loss_short_cycle_{modality}",
                self.criterionCycle(short_rec, real) * self.config.lambda_short_cycle,
            )
            setattr(
                self,
                f"loss_latent_cycle_{modality}",
                self.criterionCycle(latent_rec, latent)
                * self.config.lambda_latent_cycle,
            )

        # combined loss and calculate gradients
        self.loss_G = sum(
            getattr(self, f"loss_G_{modality}")
            + getattr(self, f"loss_short_cycle_{modality}")
            + getattr(self, f"loss_latent_cycle_{modality}")
            for modality in self.modalities
        )

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.set_requires_grad(
            [getattr(self, f"disc_{modality}") for modality in self.modalities], False
        )  # discriminators need no gradients when optimizing the encoders/decoders
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad(
            [getattr(self, f"disc_{modality}") for modality in self.modalities], True
        )
        self.optimizer_D.zero_grad()
        self.backward_D_ALL()
        self.optimizer_D.step()
