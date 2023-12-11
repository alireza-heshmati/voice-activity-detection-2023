import time
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchaudio.transforms import Resample

import onnxruntime as ort


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--wav_path",
                required=True,
                type=str,
                help="path of audio file")

ap.add_argument("-i", "--saved_model_path",
                required=True,
                type=str,
                help="path to saved onnx model")

ap.add_argument("-t", "--target_rate",
                default=16000,
                type=int,
                help="sample rate for input model")

args = vars(ap.parse_args())


wav_path = args["wav_path"]
save_model_path = args["saved_model_path"]
target_rate = args["target_rate"]


# use cuda if cuda available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Device name : {torch.cuda.get_device_name(0)}")


raw_audio, rate = torchaudio.load(wav_path)
# resample the speech
if rate != target_rate:
    transform = Resample(rate, target_rate)
    raw_audio = transform(raw_audio)
raw_audio = raw_audio.cpu().numpy()



if device == "cuda":
    provider = ["CUDAExecutionProvider"]
else:
    provider = ["CPUExecutionProvider"]

sess = ort.InferenceSession(save_model_path, providers=provider)
model_input = sess.get_inputs()[0]


predict = sess.run(None, {model_input.name: raw_audio})[0]
predict = (predict.squeeze() > 0.5).astype(np.int32)

raw_audio = raw_audio.squeeze()
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
