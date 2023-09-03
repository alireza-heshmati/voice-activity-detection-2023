import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchaudio.transforms import Resample

from models.models import PyanNet
from models.utils import load_model_config




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--wav_path",
                required=True,
                type=str,
                help="path of audio file")

ap.add_argument("-cfg", "--model_config_path",
                required=True,
                type=str,
                help="config of model for training and inference")

ap.add_argument("-i", "--saved_model_path",
                required=True,
                type=str,
                help="path to saved model")

args = vars(ap.parse_args())


wav_path = args["wav_path"]
save_model_path = args["saved_model_path"]
model_config_path = args["model_config_path"]


# use cuda if cuda available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Device name : {torch.cuda.get_device_name(0)}")



model_configs = load_model_config(model_config_path)

if model_configs["model"]["name"] == "Pyannote":
    model = PyanNet(model_configs["model"])


target_rate = model_configs["data"]["target_rate"]

model = model.to(device)
model.load_state_dict(torch.load(save_model_path))
model.eval()


raw_audio, rate = torchaudio.load(wav_path)
# resample the speech
if rate != target_rate:
    transform = Resample(rate, target_rate)
    raw_audio = transform(raw_audio)
raw_audio = raw_audio.to(device)


predict = model(raw_audio)
predict = predict.detach().cpu().numpy()
predict = (predict.squeeze() > 0.5).astype(np.int32)

raw_audio = raw_audio.cpu().numpy().squeeze()
sample_per_frame = len(raw_audio) // len(predict)

mask = []
for c in predict:
    if c == 0:
        mask = mask + [0] * sample_per_frame
    else:
        mask = mask + [1] * sample_per_frame
mask = np.array(mask, np.float32)

if len(mask) > len(raw_audio):
    mask = mask[:len(raw_audio)]
else:
    raw_audio = raw_audio[:len(mask)]

plt.figure()
plt.plot(raw_audio)
plt.plot(mask)
plt.legend(["audio", "vad mask"])
plt.show()
