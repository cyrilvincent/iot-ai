import onnxruntime as ort
import numpy as np

model_path = "data/breast-cancer/cancer_mlp_int8_quark.onnx"

providers = ["VitisAIExecutionProvider"]  # DPU à comparer avec CPUExecutionProvider

provider_options = [{
    "config_file": "data/breast-cancer/vitisai_config.json",
    "cache_dir": "./cache",
    "cache_key": "model",
    "target": "VAIML"
}]

session = ort.InferenceSession(
    model_path,
    providers=providers,
    provider_options=provider_options
)

# Input exemple (adapter à ton modèle)
input_name = session.get_inputs()[0].name
x = np.random.rand(1, 30).astype(np.float32)

output = session.run(None, {input_name: x})

print(output)
