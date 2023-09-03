import numpy as np


# post processing for vad labels 
# remove 200 ms gap
def remove_short_space(label):
    """Post processing for vad labels to remove the labels of short non-speech between
      two speech, for example 200 ms gap between two speech.

    Arguments
    ---------
    label : numpy
        Label for postprocessing

    Returns
    -------
    data
        Readed audio output as torch

    new_label : numpy
        New label after postprocessing.

    """
    non_zero = np.where(label != 0)[0]
    new_label = np.zeros_like(label)
    for counter in range(len(non_zero) - 1):
        c1, c2 = non_zero[counter], non_zero[counter+1]
        if ( c2 - c1 ) <= 10:
            new_label[c1-1:c2+2] = 1
    return new_label