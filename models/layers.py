import torch
import torch.nn as nn

from asteroid_filterbanks.enc_dec import Filterbank, Encoder
from asteroid_filterbanks.param_sinc_fb import ParamSincFB



class SincNet(nn.Module):
    """Filtering and convolutional part of Pyannote

    Arguments
    ---------
    n_filters : list, int
        List consist of number of each convolution kernel
    stride_ : in 
        Stride of ParamSincFB fliltering.

    Returns
    -------
    Sincnet model: class

    """
    
    def __init__(self, 
                 n_filters = [80,60,60],
                 stride_ = 10,
                 ):
        super(SincNet,self).__init__()
        

        sincnet_list = nn.ModuleList(
            [
                nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                Encoder(ParamSincFB(n_filters=n_filters[0], kernel_size=251, stride=stride_)),
                nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
                nn.InstanceNorm1d(n_filters[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            ]
        )
        for counter in range(len(n_filters) - 1):
            sincnet_list.append(nn.Conv1d(n_filters[counter], n_filters[counter+1], kernel_size=(5,), stride=(1,)))
            sincnet_list.append(nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False))
            sincnet_list.append(nn.InstanceNorm1d(n_filters[counter+1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))

        self.sincnet_layer = nn.Sequential(*sincnet_list)

    def forward(self, x):
        """This method should implement forwarding operation in the SincNet model.

        Arguments
        ---------
        x : float (Tensor)
            The input of SincNet model.

        Returns
        -------
        out : float (Tensor)
            The output of SincNet model.
        """
        out = self.sincnet_layer(x)
        return out