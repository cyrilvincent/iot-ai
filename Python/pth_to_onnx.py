import torch
from torch_mlp_cancer import MLP
import onnx
import onnxscript  #pip install onnxscript

name = "data/breast-cancer/cancer_mlp.pth"
model = MLP(30)
model.load_state_dict(torch.load(name))
model.eval()

input_tensor = torch.rand(1, 30)

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    name.replace(".pth", ".onnx"),        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)
