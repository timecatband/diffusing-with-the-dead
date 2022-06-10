import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch


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
    print("Loaded datset")
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader


def _list_wav_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]       
        if bf.isdir(full_path):
            results.extend(_list_wav_files_recursively(full_path))
        if os.path.getsize(full_path) > 26500:
            results.append(full_path)

    return results

from numpy import ndarray
class AudioDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        waveform, sample_rate = torchaudio.load(path)
        waveform = torch.narrow(waveform, 1, 0, 4096)
#        torchaudio.save("/content/lol.wav", waveform, 22025)
        return waveform, ()
