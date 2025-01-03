# Configuration Structure

This [**folder**](../recipes) include all of the configuration files related to the models. 
In this file you can set some hyperparameter of the model, data, process of the training.


#### The configuration file of the pyannote_v2.2 is shown below: ####

```bash
{
    "model": {
        "name": "Pyannote",
        "sincnet_filters": [80,60,60],
        "sincnet_stride": 30,
        "sequence_type": "gru",
        "sequence_nlayers": 2,
        "sequence_neuron": 64,
        "sequence_drop_out": 0.1,
        "sequence_bidirectional": true,
        "linear_blocks": [128,128,1]
    },
    "data": {
        "num_workers": 20,
        "pin_memory": false,
        "target_rate": 16000,
        "data_base_path": "datasets",
        "post_proc": false
    },
    "train": {
        "batch_size": 256,
        "epoch": 20,
        "step_show": 2500,
        "loss_fn": "BCE"
    }
}

```





