import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config
)
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from torch_mlp_cancer import MLP

model = MLP(30)
model.load_state_dict(torch.load("data/breast-cancer/cancer_mlp.pth"))
model.eval()

shape = (torch.randn(1, 30),)

exported = export(model, shape)
m = exported.module()

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)
with torch.no_grad():
    for _ in range(200):
        m(shape[0])
m = convert_pt2e(m)

et_program = to_edge_transform_and_lower(
    export(m, shape),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("data/breast-cancer/cancer_mlp_int8.pte", "wb") as f:
    f.write(et_program.buffer)

print(f"Taille : {len(et_program.buffer) / 1024:.1f} KB")