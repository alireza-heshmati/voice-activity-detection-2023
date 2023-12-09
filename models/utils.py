import json
import copy

def load_model_config(config_path):
    """Load model config

    Arguments
    ---------
    config_path : str
        Path of config

    Returns
    -------
    configs : dict, str
        Loaded config

    """
    with open(config_path, "r") as f:
        configs = json.load(f)
    return configs



def cal_frame_sample_pyannote(wav_length,
                              sinc_step=10,
                              sinc_filter=251,
                              n_conv=2,
                              conv_filter= 5,
                              max_pool=3):
    """Define the number and the length of frames according to Pyannote model

    Arguments
    ---------
    wav_length : int
        Length of wave
    sinc_step : int
        Frame shift
    sinc_filter : int
        Length of sincnet filter
    n_conv : int
        Number of convolutional layers
    conv_filter : int
        Length of convolution filter
    max_pool : int
        Lenght of maxpooling
    
    Returns
    -------
    n_frame : float
        The number of frames according to Pyannote model
    sample_per_frame : float
        The length of frames according to Pyannote model

    """

    n_frame = (wav_length - (sinc_filter - sinc_step)) // sinc_step
    n_frame = n_frame // max_pool

    for _ in range(n_conv):
        n_frame = n_frame - (conv_filter - 1)
        n_frame = n_frame // max_pool

    sample_per_frame = wav_length // n_frame

    return n_frame, sample_per_frame


def wav_label_to_frame_label_pyannote(label, num_frame, frame_shift):
    """Create framed label from sampled label

    Arguments
    ---------
    label : float (Tensor)
        Sampled label
    num_frame : int
        number of frames in the audio
    frame_shift : int
        Length of frame
    
    Returns
    -------
    final_label : float (Tensor)
        Decided framed label

    label : float (Tensor)
        Framed label

    """

    LEN = num_frame * frame_shift
    label = label[..., :LEN]
    label = label.reshape(label.shape[0], num_frame, frame_shift)

    label = label.float().mean(-1, True)
    final_label = copy.deepcopy(label)
    final_label[final_label > 0.5] = 1
    final_label[final_label <= 0.5] = 0

    return final_label, label

# make label frames from label samples
def pyannote_target_fn(target, model_configs):
    """Make framed label from sampeled label

    Arguments
    ---------
    target : float (Tensor)
        Sampled label

    model_configs : dict, str
        For sincnet_filters
    
    Returns
    -------
    output : float (Tensor)
        framed label

    """
    n_conv = len(model_configs["sincnet_filters"]) - 1
    sincnet_stride = model_configs["sincnet_stride"]
    num_frame, len_frame = cal_frame_sample_pyannote(target.shape[-1], sinc_step= sincnet_stride, n_conv=n_conv)
    target,_ = wav_label_to_frame_label_pyannote(target, num_frame, len_frame)
    return target