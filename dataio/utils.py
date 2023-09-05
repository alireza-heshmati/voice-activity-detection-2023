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

    post_label : float (torch.float)
        readed label

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
    post_label = np.array([int(c) for c in label])
    
    # post processing for vad labels 
    if post_proc:
        post_label = remove_short_space(post_label)

    buffer = []
    for c in post_label:
        if c == 0:
            buffer += [0] * frame_length # 320 : the Number of samples allocated for each frame 
        else:
            buffer += [1] * frame_length
    post_label = np.array(buffer)
    
    # align length of label with the signal
    if raw_audio.shape[-1] < len(post_label):
        post_label = post_label[:len(raw_audio)]
    else:
        raw_audio = raw_audio[:,:len(post_label)]
        
    return raw_audio, torch.tensor(post_label)


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

    """

    tensors, targets = [], []
    for tensor, label in batch:
        tensors.append(tensor.squeeze())
        targets.append(label.squeeze())

    tensors = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    return tensors, targets