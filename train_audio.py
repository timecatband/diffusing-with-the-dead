from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm
import torchaudio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from deaddream.ddpm import DDPM
from deaddream.dataset import load_data
from deaddream.audio_utilities import reconstruct_signal_griffin_lim

from deaddream.vqvae import VQVAE



#from dotenv import load_dotenv

#load_dotenv("./.env")
#DEAD_PATH = os.getenv("DEAD_PATH")


def train_dead(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:
    vqmodel = VQVAE(in_channel=2)
    ddpm = DDPM(eps_model=vqmodel, betas=(1e-4, 0.02), n_T=1000)
    

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_dead.pth"))

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

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        j = 0
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            print("shape: " + str(x.shape))
            j=j+1
            if j%20 == 0:
                torch.save(ddpm.state_dict(), f"./ddpm_dead.pth")
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(1, (2, 50, 80), device)
            xh[0]=xh[0].clamp(0,1)
            print(xh[0].shape)
            xh = xh.to("cpu")
            lol = torch.narrow(xh[0], 2, 0, 51)
            wav = reconstruct_signal_griffin_lim(lol[0], 100, 80, 100)
            wav = torch.from_numpy(wav)
            wav = wav.unsqueeze(0).type(torch.float32)
            print("wav shape: " + str(wav.dtype))
            torchaudio.save("/content/ddpm_sample_out"+".wav", wav, 44050)

            # save model



if __name__ == "__main__":
    train_dead()
