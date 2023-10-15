import numpy as np

import torch
import torchaudio
from torchaudio.transforms import Resample
from torch.nn.utils.rnn import pad_sequence

from dataio.postprocessing import remove_short_space


def read_audio_label(speech_path, 
                      label_path, 
                      post_proc = False, 
                      target_rate = 16000,
                      frame_length = 20.): 
    """read and align data and labels from the path.

    Arguments
    ---------
    speech_path : str
        Path of the audio
    label_path : str
        Path of the label
    post_proc : bool
        Change the labels of short non-speech between two speech, for example 200 ms gap between two speech.
    target_rate : int
        Sampling rate of audio.
    frame_length : float
        frame length in millisecond

    Returns
    -------
    raw_audio : float (torch.float)
        readed audio

    time_label : float (torch.float)
        readed label

    label : float (torch.float)
        readed 20ms framed label

    """
    
    frame_length = int(frame_length * target_rate / 1000)

    raw_audio, rate = torchaudio.load(speech_path)
    
    # resample the speech
    if rate != target_rate:
        transform = Resample(rate, target_rate)
        raw_audio = transform(raw_audio)
    
    # fetch label
    with open(label_path, "r") as f:
        label = f.read().split(" ")
    label = np.array([int(c) for c in label])
    
    # post processing for vad labels 
    if post_proc:
        label = remove_short_space(label)

    time_label = torch.repeat_interleave(torch.tensor(label), 320, dim = -1)
    
    # align length of label with the signal
    if raw_audio.shape[-1] < len(time_label):
        time_label = time_label[:len(raw_audio)]
    else:
        raw_audio = raw_audio[:,:len(time_label)]
        
    return raw_audio, torch.tensor(time_label), torch.tensor(label).float()
        


def collate_fn(batch):
    """help torch.loader give data and labels. indeed preparing readed dataset for network with padding.

    Arguments
    ---------
    batch : str
        Path of the audio

    Returns
    -------
    tensors : float (torch.float)
        padded audios

    targets : float (torch.float)
        padded labels

    frmd_targets: float (torch.float)
        padded framed label

    """

    tensors, targets, frmd_targets = [], [], []
    for log_mel, label, frmd_label in batch:
        tensors.append(log_mel.squeeze())
        targets.append(label.squeeze())
        frmd_targets.append(frmd_label.squeeze())

    tensors = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    frmd_targets = pad_sequence(frmd_targets, batch_first=True, padding_value=0.0)

    return tensors, targets, frmd_targets