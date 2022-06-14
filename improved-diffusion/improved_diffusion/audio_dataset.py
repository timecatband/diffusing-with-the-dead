
import blobfile as bf
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch
from torchaudio.transforms import Spectrogram
from torchaudio.transforms import InverseSpectrogram
from torchaudio.transforms import AmplitudeToDB

from torchaudio.transforms import InverseMelScale
from torchaudio.transforms import MelScale
import matplotlib.pyplot as plt
import torchaudio.functional as FA
import librosa
import torch.nn.functional as NF

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.savefig("/content/lol.png"

)

def load_data(
    *, data_dir, batch_size, audio_size, class_cond=False, deterministic=False
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_wav_files_recursively(data_dir)
    print("Loading dataset: " + data_dir + str(len(all_files)))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = AudioDataset(
        audio_size,
        all_files

    )
    print("dataset len: " + str(len(dataset)))
    print("Loaded datset" + str(len(all_files)))
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
        while True:
            yield from loader



def _list_wav_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]       
        if bf.isdir(full_path):
            results.extend(_list_wav_files_recursively(full_path))
        if os.path.getsize(full_path) > 770000/4:
            results.append(full_path)
        else:
            print("Skipping file: " + full_path)

    return results

from numpy import ndarray
class AudioDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.spec = Spectrogram(512, onesided=True, normalized=True, power=None)
        self.scale = MelScale(256, int(20050/4), n_stft=256, norm="slaney")
      #  self.a2db = AmplitudeToDB()
    
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        waveform, sample_rate = torchaudio.load(path)
      #  print("waveform: " + str(waveform.shape))
        waveform = torch.narrow(waveform, 1, 0, int(262144/4))
      #  waveform = FA.resample(waveform ,22050, int(22050/4))     #   waveform = FA.lowpass_biquad(waveform, 22050, 8000)

        spec = self.spec(waveform)
       # print("Spec.size: " + str(spec.shape))
        #print("Spec data: " + str(spec.max()))


        spec = spec.narrow(1, 0, 256)
        spec = spec.narrow(2, 0, 256)
     #   plot_spectrogram(spec[0])

        spec = torch.view_as_real(spec)
        spec = spec.permute(0,3,1,2)
        spec = spec[0]
#        spec[0] = self.a2db(spec[0])


      #  spec[0] = self.scale(spec[0])
    #    plot_spectrogram(spec[0])

    #    spec = (spec-spec.mean())/spec.std(unbiased=True)
     #   spec = spec/(spec.max())
      #  inverse_mel = InverseMelScale(64, 64, 22050)
 
     #   plot_spectrogram(inverse_mel(spec).detach()[0])
      #  print("eamn: " + str(spec.mean()))
      #  print("var: " + str(spec.var()))
     #   print("spec max:" + str(spec.max()) + ", " + str(spec.min()))
        #spec = self.scale(spec)
        return spec, {}
