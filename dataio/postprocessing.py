import numpy as np


# post processing for vad labels 
# remove 200 ms gap
def remove_short_space(label):
    non_zero = np.where(label != 0)[0]
    new_label = np.zeros_like(label)
    for counter in range(len(non_zero) - 1):
        c1, c2 = non_zero[counter], non_zero[counter+1]
        if ( c2 - c1 ) <= 10:
            new_label[c1-1:c2+2] = 1
    return new_label