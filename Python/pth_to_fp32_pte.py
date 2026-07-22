import torch
from executorch.exir import to_edge
from torch.export import export
from torch_mlp_cancer import MLP

name = "data/breast-cancer/cancer_mlp.pth"
model = MLP(30)
model.load_state_dict(torch.load(name))
model.eval()

shape = torch.randn(1, 30)  # 1 = batch_size toujours 1, 30 = nb features

# Export
exported = export(model, (shape,))
et_program = to_edge(exported).to_executorch()

with open(name.replace(".pth", ".pte"), "wb") as f:
    f.write(et_program.buffer)

print(f"Taille : {len(et_program.buffer) / 1024:.1f} KB")
