import os
import argparse
import timeit
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tools.train_utils import evaluate_epoch

from models.models import PyanNet
from models.utils import load_model_config, pyannote_target_fn
from dataio.dataloader import CustomAudioDataset
from dataio.utils import collate_fn




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--model_config_path",
                required=True,
                type=str,
                help="config of model for evaluation")

ap.add_argument("-i", "--saved_model_path",
                required=True,
                type=str,
                help="path to saved model")

args = vars(ap.parse_args())


save_model_path = args["saved_model_path"]
model_config_path = args["model_config_path"]



np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# use cuda if cuda available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Device name : {torch.cuda.get_device_name(0)}")



model_configs = load_model_config(model_config_path)

if model_configs["model"]["name"] == "Pyannote":
    model = PyanNet(model_configs["model"])
    target_fn = partial(pyannote_target_fn, model_configs=model_configs["model"])
else :
    raise ValueError("the name of the VAD model is not supported!!")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of model's parameters : {total_params}")



data_base_path = model_configs["data"]["data_base_path"]
batch_size = model_configs["train"]["batch_size"]
num_workers = model_configs["data"]["num_workers"]
pin_memory = model_configs["data"]["pin_memory"]
target_rate = model_configs["data"]["target_rate"]
post_proc = model_configs["data"]["post_proc"]

model = model.to(device)
model.load_state_dict(torch.load(save_model_path))

if model_configs["train"]["loss_fn"] == "BCE":
    loss_fn = torch.nn.BCELoss()
else:
    raise ValueError("Loss function is not supported!!")


label_path = os.path.join(data_base_path,"VadLabel")
test_path = pd.read_csv(os.path.join(data_base_path,"test_filenames.csv"))['speech_path']
print(f"sample test: {len(test_path)}")


# for reading and preparing test dataset
test_loader = DataLoader(
    CustomAudioDataset(test_path, label_path, data_base_path, post_proc=post_proc, target_rate=target_rate),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


start = timeit.default_timer()
test_loss, test_fscore, test_mcc = evaluate_epoch(model, test_loader, loss_fn, target_fn, device)
print(f"\nTest accuracy on Best model. Test_loss: {test_loss:.4f}, Test_Fscore: {test_fscore:.3f}, Test_MCC: {test_mcc:.3f}")
print('\ntotal Time (Min) : ', round((timeit.default_timer() - start) / 60, 3))
