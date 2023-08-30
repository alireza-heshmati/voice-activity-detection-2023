import json


def load_model_config(config_path):
    with open(config_path, "r") as f:
        configs = json.load(f)
    return configs


# define the number and length of frames according to Pyannote
def cal_frame_sample_pyannote(wav_length,
                              sinc_step=10,
                              sinc_filter=251,
                              n_conv=2,
                              conv_filter= 5,
                              max_pool=3):
    

    n_frame = (wav_length - (sinc_filter - sinc_step)) // sinc_step
    n_frame = n_frame // max_pool

    for _ in range(n_conv):
        n_frame = n_frame - (conv_filter - 1)
        n_frame = n_frame // max_pool

    sample_per_frame = wav_length // n_frame

    return n_frame, sample_per_frame


def wav_label_to_frame_label_pyannote(label, num_frame, frame_shift):
    LEN = num_frame * frame_shift
    label = label[..., :LEN]
    label = label.reshape(label.shape[0], num_frame, frame_shift)

    label = label.float().mean(-1, True)
    label[label > 0.5] = 1
    label[label <= 0.5] = 0

    return label

# make label frames from label samples
def pyannote_target_fn(target, model_configs):
    n_conv = len(model_configs["sincnet_filters"]) - 1

    num_frame, len_frame = cal_frame_sample_pyannote(target.shape[-1], n_conv=n_conv)
    return wav_label_to_frame_label_pyannote(target, num_frame, len_frame)