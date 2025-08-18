import torch
import itertools
from .image_pool import ImagePool
from .base_model import BaseModel
from . import loss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import random
from torch.nn import init
from . import networks


class GanToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        return torch.tensor(sample).float()


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_normal_(m.weight.data)
#         #   nn.init.kaiming_normal(m.weight.data, nonlinearity='relu')
#         nn.init.constant_(m.bias.data, 0)


# def init_weights_d(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight.data)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Aligned_Dataset(Dataset):
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
        # index = self.overlapping_indices[item]
        for modality, df in zip(self.modalities, self.dataframes):
            data[modality] = self.transform(df.iloc[item].values)
        # data["index"] = index

        return data

    def __len__(self):
        return self.size


class Unaligned_Dataset(Dataset):
    def __init__(self, config):
        self.file_names = config.input_files
        self.modalities = config.modalities
        self.transform = GanToTensor()
        self.size = 0
        self.dataframes = []
        self.load_dataframes()

    def load_dataframes(self):
        self.dims = {}
        for i, file in enumerate(self.file_names):
            df = pd.read_table(
                file,
                sep="\t",
                index_col=0,
            )
            self.dataframes.append(df)
            self.size = max(self.size, df.shape[0])
            self.dims[self.modalities[i]] = df.shape[1]

    def __getitem__(self, item):
        data = {}
        for modality, value in zip(self.modalities, self.dataframes):
            index = random.randint(0, value.shape[0] - 1)
            data[modality] = self.transform(value.iloc[index])

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
        dropout=0.25,
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )
        self.encoder = init_net(self.encoder, init_type, init_gain, gpu_ids=gpu_ids)
        # self.encoder.apply(init_weights)

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_size,
        output_size,
        gpu_ids=[],
        dropout=0.25,
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = init_net(self.decoder, init_type, init_gain, gpu_ids=gpu_ids)
        # self.decoder.apply(init_weights_d)

    def forward(self, x):
        return self.decoder(x)


# TODO: Test different architectures for the discriminator (patchGAN, pixelGAN, etc.)
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
        self.discriminator = init_net(
            self.discriminator, init_type, init_gain, gpu_ids=gpu_ids
        )
        # self.discriminator.apply(init_weights_d)

    def forward(self, x):
        return self.discriminator(x)


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, config):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, config)
        # Create a encoder for each modality
        self.modalities = config.modalities
        for modality in self.modalities:
            setattr(
                self,
                f"enc_{modality}",
                Encoder(
                    self.config.dims[modality],
                    self.config.hidden_dim,
                    self.config.latent_dim,
                    self.config.gpu_ids,
                    self.config.dropout,
                    init_type=self.config.init_type,
                ),
            )
        # Create a decoder for each modality
        for modality in self.modalities:
            setattr(
                self,
                f"dec_{modality}",
                Decoder(
                    self.config.latent_dim,
                    self.config.hidden_dim,
                    self.config.dims[modality],
                    self.config.gpu_ids,
                    self.config.dropout,
                    init_type=self.config.init_type,
                ),
            )

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = (
            [f"G_{modality}" for modality in self.modalities]
            + [f"short_cycle_{modality}" for modality in self.modalities]
            + [f"latent_cycle_{modality}" for modality in self.modalities]
            + [f"D_{modality}" for modality in self.modalities]
        )

        if self.isTrain:
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
                        self.config.gpu_ids,
                        init_type=self.config.init_type,
                    ),
                )

            # create image buffer to store previously generated images
            for modality in self.modalities:
                setattr(self, f"fake_{modality}_pool", ImagePool(self.config.pool_size))

            # define loss functions
            self.criterionGAN = loss.GANLoss(self.config.gan_mode).to(
                self.device
            )  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # Create a list of parameter iterables from all encoders
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
        else:  # during test time, only load Gs
            self.model_names = [f"enc_{modality}" for modality in self.modalities] + [
                f"dec_{modality}" for modality in self.modalities
            ]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        for modality in self.modalities:
            setattr(
                self,
                f"real_{modality}",
                input[modality].to(self.device),
            )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
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
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

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
            # self.loss_gp.backward(retain_graph=True)

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

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.disc_B, self.real_B, fake_B)

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.disc_A, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # losses
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
            # Short cycle loss || Dec_A(Enc(A)) - A||
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
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [getattr(self, f"disc_{modality}") for modality in self.modalities], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(
            [getattr(self, f"disc_{modality}") for modality in self.modalities], True
        )
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_ALL()
        # self.backward_D_A()  # calculate gradients for D_A
        # self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
