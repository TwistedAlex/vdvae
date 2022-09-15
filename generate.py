import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import set_up_data
from utils import get_cpu_stats_over_ranks
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
import PIL.Image


def main():
    H, logprint = set_up_hyperparams()
    H = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)

    for idx in range(1000):

        img = ema_vae.forward_uncond_samples(1, t=1)
        PIL.Image.fromarray(img[0], 'RGB').save(f'{idx:05d}.png')

if __name__ == "__main__":
    main()