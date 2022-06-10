import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from tcn import TCNModel
from data import LibriMixDataset

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--root_dir", type=str, default="./data")
parser.add_argument("--sample_rate", type=int, default=8000)
parser.add_argument("--train_subset", type=str, default="train")
parser.add_argument("--val_subset", type=str, default="val")
parser.add_argument("--train_length", type=int, default=16384)
parser.add_argument("--eval_length", type=int, default=32768)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=0)

# add model specific args
parser = TCNModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# parse them args
args = parser.parse_args()

# init the trainer and model
trainer = pl.Trainer.from_argparse_args(args)

# setup the dataloaders
train_dataset = LibriMixDataset(
    args.root_dir, subset=args.train_subset, length=args.train_length
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)

val_dataset = LibriMixDataset(
    args.root_dir, subset=args.val_subset, length=args.eval_length
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)

dict_args = vars(args)
model = TCNModel(**dict_args)


# find proper learning rate
trainer.tune(model, train_dataloader)

# train!
trainer.fit(model, train_dataloader, val_dataloader)
