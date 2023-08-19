import os
import argparse
import timeit
from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from metrics import F1_Score, MCC

from base.models import PyanNet
from base.utils import load_model_config, pyannote_target_fn
from data.dataset import CustomAudioDataset
from data.utils import collate_fn




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--model_config_path",
                required=True,
                type=str,
                help="config of model for evaluation")

ap.add_argument("-s", "--save_model_path",
                required=True,
                type=str,
                help="path for saved model")

args = vars(ap.parse_args())


save_model_path = args["save_model_path"]
model_config_path = args["model_config_path"]



np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# use cuda if cuda available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Device name : {torch.cuda.get_device_name(0)}")



# Evaluate model with loss, F1-Score and MCC
def evaluate_epoch(model, data_loader, loss_fn, target_fn, device):
    model.eval()

    loss = 0
    TP = 0 # pred 1, actual 1
    FP = 0 # pred 1, actual 0
    TN = 0 # pred 0, actual 0
    FN = 0 # pred 0, actual 1
    counter = 0

    with torch.no_grad():  
        for data, target in tqdm(data_loader):
            target = target_fn(target)
            
            output = model(data.to(device)).cpu()
            loss += loss_fn(output, target)

            ind_pred = output > 0.5
            ind_target = target > 0.5
            
            # Calculate TP, FP, FN, TN
            TP += len(target[ind_pred * ind_target])
            FP += len(target[ind_pred * ~ind_target])
            FN += len(target[~ind_pred * ind_target])
            TN += len(target[~ind_pred * ~ind_target])

            del data, target,  ind_pred, ind_target
            counter += 1

    f1 = F1_Score(TP, FP, TN, FN)
    mcc = MCC(TP, FP, TN, FN)
    loss = loss.cpu().item() / counter

    return round(loss, 5), round(f1, 3), round(mcc, 3)




model_configs = load_model_config(model_config_path)

if model_configs["model"]["name"] == "Pyannote":
    model = PyanNet(model_configs["model"])
    target_fn = partial(pyannote_target_fn, model_configs=model_configs["model"])

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