from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM
from mindiffusion.dataset import load_data
import torchaudio

#from dotenv import load_dotenv

#load_dotenv("./.env")
#CELEBA_PATH = os.getenv("CELEBA_PATH")


def train_celeba(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(2, 2, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    ddpm.load_state_dict(torch.load("/content/ddpm_audio.pth"))

    ddpm.to(device)

#    tf = transforms.Compose(  # resize to 512 x 512, convert to tensor, normalize
#        [
#            transforms.Resize((128, 128)),
#            transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#        ]
#    )

    dataloader = load_data(data_dir="/content/data/wav/chunks", batch_size=32, audio_size=4096)


    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(1, (2, 4096), device)
        xh[0]=xh[0].clamp(0,1)
        print(xh[0].shape)
        torchaudio.save("/content/ddpm_sample_out"+".wav", xh[0], 22025)

if __name__ == "__main__":
    train_celeba()
