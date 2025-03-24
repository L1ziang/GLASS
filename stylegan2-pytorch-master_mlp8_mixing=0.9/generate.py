import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision import transforms
from torch import nn

class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Normalized Tensor image.

        Returns:
            Tensor: Denormalized Tensor.
        """
        return self._denormalize(tensor)

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)

        return tensor


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            # print(sample.shape) #torch.Size([10, 3, 64, 64])
            # exit()
            # print(sample[0].shape) #torch.Size([3, 64, 64])
            # print(sample[0].max)
            # norm = DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # s = norm(sample[0])
            
            # min=-1
            # max=1
            # sample[0].clamp_(min=min, max=max)
            # sample[0].add_(-min).div_(max - min + 1e-5)
            # vutils.save_image(sample[0], 'test1.png', normalize=False)
            # exit()

            utils.save_image(
                sample,
                f"sample_g/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=64, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=50, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096, # 没用到
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoint/040000.pt",
        # default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 4

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        # print("exit")
        # exit()
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
