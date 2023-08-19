import os

import torch
from torch.utils.data import Dataset, DataLoader

from data.utils import read_audio_label, collate_fn



# prepare dataset for train, validation, test
class CustomAudioDataset(Dataset):
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