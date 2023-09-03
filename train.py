import os
import json
import argparse
import timeit
from functools import partial

import numpy as np
import pandas as pd
import torch

from tools.train_utils import run

from models.models import PyanNet
from models.utils import load_model_config, pyannote_target_fn
from dataio.dataset import data_loader




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--model_config_path",
                required=True,
                type=str,
                help="config of model for training")

ap.add_argument("-o", "--saving_model_path",
                required=True,
                type=str,
                help="path for saving model")

args = vars(ap.parse_args())


save_model_path = args["saving_model_path"]
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

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of model's parameters : {total_params}")



data_base_path = model_configs["data"]["data_base_path"]
batch_size = model_configs["train"]["batch_size"]
epoch = model_configs["train"]["epoch"]
num_workers = model_configs["data"]["num_workers"]
pin_memory = model_configs["data"]["pin_memory"]
target_rate = model_configs["data"]["target_rate"]
step_show = model_configs["train"]["step_show"] 
post_proc = model_configs["data"]["post_proc"]

model = model.to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1, rho=0.95, eps=1e-08)

if model_configs["train"]["loss_fn"] == "BCE":
    loss_fn = torch.nn.BCELoss()
else:
    raise ValueError("Loss function is not supported!!")


label_path = os.path.join(data_base_path,"VadLabel")

train_path = pd.read_csv(os.path.join(data_base_path,"train_filenames.csv"))['speech_path']
validation_path = pd.read_csv(os.path.join(data_base_path,"valid_filenames.csv"))['speech_path']
test_path = pd.read_csv(os.path.join(data_base_path,"test_filenames.csv"))['speech_path']
print(f"sample train : {len(train_path)}, sample valid: {len(validation_path)}, sample test: {len(test_path)}")


train_loader, validation_loader, test_loader = data_loader(train_path, 
                                                           validation_path,
                                                           test_path,
                                                           label_path,
                                                           data_base_path,
                                                           target_rate,
                                                           batch_size,
                                                           num_workers,
                                                           pin_memory,
                                                           post_proc=post_proc)



start = timeit.default_timer()
train_losses, val_losses, val_fscores, val_mccs, test_loss, test_fscore, test_mcc = run(
    model,
    train_loader,
    validation_loader,
    test_loader,
    optimizer,
    loss_fn,
    target_fn,
    device,
    save_model_path,
    step_show,
    epoch,
)

result_metrics = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_fscores": val_fscores,
    "val_mccs": val_mccs,
    "test_loss": test_loss,
    "test_fscore": test_fscore,
    "test_mcc": test_mcc,
}
with open(save_model_path[:-4] + ".json", "w") as f:
    json.dumps(f, indent=4)

print('\ntotal Time (Hour) : ', round((timeit.default_timer() - start) / 3600, 3))
