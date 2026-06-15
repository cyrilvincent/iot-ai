import torch
import litert_torch
from torch_mlp_cancer import MLP

name = "data/breast-cancer/cancer_mlp.pth"
model = MLP(30)
model.load_state_dict(torch.load(name))

input_tensor = torch.rand(1, 30)

edge_model = litert_torch.convert(model.eval(), input_tensor)
edge_model.export(model + ".tflite")

# Sur WSL2 Ubuntu
# cd /mnt/c/Users/conta/git-CVC/Formation/IoT/git-iot-ai/python
# pip install torch