import os
import argparse

import torch
import torch.onnx

from models.utils import load_model_config
from models.models import PyanNet


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--model_config_path",
                required=True,
                type=str,
                help="config of model for training and inference")

ap.add_argument("-i", "--saved_model_path",
                required=True,
                type=str,
                help="path for loading raw model")

ap.add_argument("-o", "--export_model_path",
                required=True,
                type=str,
                help="path for saving exported model")

args = vars(ap.parse_args())


saved_model_path = args["saved_model_path"]
export_model_path = args["export_model_path"]
model_config_path = args["model_config_path"]



# use cuda if cuda available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Device name : {torch.cuda.get_device_name(0)}")


model_configs = load_model_config(model_config_path)

if model_configs["model"]["name"] == "Pyannote":
    model = PyanNet(model_configs["model"]).to(device)
model.load_state_dict(torch.load(saved_model_path),strict=True)

model.eval()
dummy_input = torch.randn(2, 25000, requires_grad=True).to(device)

torch.onnx.export(model,                                               # model being run
                  dummy_input ,                                        # model input (or a tuple for multiple inputs)
                  export_model_path,                                   # where to save the model
                  export_params=True,                                  # store the trained parameter weights inside the model file
                  opset_version=16,                                    # the ONNX version to export the model to
                  do_constant_folding=True,                            # whether to execute constant folding for optimization
                  input_names=['waves_input'],                         # the model's input names
                  output_names=['output'],                             # the model's output names,
                  dynamic_axes={                                       # dynamic axis      
                          'waves_input': {0:'batch_size', 1:'len'},
                          'output' : {0 : 'batch_size',1:'frame' }
                  },
)