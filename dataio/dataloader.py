

import os

import torch
from torch.utils.data import Dataset, DataLoader

from dataio.utils import read_audio_label, collate_fn


class CustomAudioDataset(Dataset):
    """Tis class read data and a label.

    Arguments
    ---------
    speech_filenames : str
        List of names of train, validation and test audio files.
    label_path : str
        Path of label file for example os.path.join(data_base_path,"VadLabel").
    data_base_path : str
        Path of dataset file for example dataio/dataset.
    post_proc : bool
        Change the labels of short non-speech between two speech, for example 200 ms gap between two speech.
    target_rate : int
        Sampling rate of audio.

    Returns
    -------
    data
        Readed audio output as torch

    label
        The label relative to data.

    """
    def __init__(self,
                 speech_filenames,
                 label_path,
                 data_base_path,
                 post_proc = False,
                 target_rate = 16000):
        
        self.speech_filenames = speech_filenames
        self.label_path = label_path
        self.post_proc = post_proc
        self.data_base_path = data_base_path
        self.target_rate = target_rate
        
    def __len__(self):
        return len(self.speech_filenames)

    def __getitem__(self, idx):
        speech_filename = os.path.join(self.data_base_path, self.speech_filenames[idx])
        
        label_filename = os.path.basename(speech_filename)[:-4].split("SPLIT")[0] + ".txt"
        label_filename = os.path.join(self.label_path, label_filename)
        data, label = read_audio_label(speech_filename, label_filename, post_proc=self.post_proc, target_rate=self.target_rate)
        
        return data, label
        

# for reading and preparing dataset
def data_loader(train_path,
                validation_path,
                test_path,
                label_path, 
                data_base_path, 
                target_rate,
                batch_size, 
                num_workers, 
                pin_memory, 
                post_proc = False):
    
    """Tis class prepare dataset for train, validation, test.

    Arguments
    ---------
    train_path : str
        List of names of train audio files.
    validation_path : str
        List of names of validation audio files.
    test_path : str
        List of names of test audio files.
    label_path : str
        Path of label file for example os.path.join(data_base_path,"VadLabel").
    data_base_path : str
        Path of dataset file for example dataio/dataset.
    target_rate : int
        Sampling rate of audio.
    batch_size : int
        Size of batch size for training and evaluating.
    num_workers : int
        Number of cpu used.
    pin_memory : bool
        This is good for using GPU to collect data.
    post_proc : bool
        Change the labels of short non-speech between two speech, for example 200 ms gap between two speech.

    Returns
    -------
    train_loader
        loader of train set

    validation_loader
        loader of validat set

    test_loader
        loader of test set

    """

    # for reading and preparing train dataset
    train_loader = DataLoader(
        CustomAudioDataset(train_path, label_path, data_base_path, post_proc=post_proc, target_rate=target_rate),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # for reading and preparing validation dataset
    validation_loader = DataLoader(
        CustomAudioDataset(validation_path, label_path, data_base_path, post_proc=post_proc, target_rate=target_rate),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

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
    return train_loader, validation_loader, test_loader