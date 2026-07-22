import tensorflow as tf
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("==== TensorFlow ====")
print("Version TF :", tf.__version__)
print("Built with CUDA :", tf.test.is_built_with_cuda())

# =========================
# 1. GPU (TensorFlow)
# =========================
print("\n==== GPU TensorFlow ====")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print("Nom GPU :", details.get("device_name", "Unknown"))
    print("✅ GPU détecté")
else:
    print("❌ Aucun GPU détecté")

# =========================
# 2. CUDA version (TF build)
# =========================
print("\n==== CUDA (TensorFlow) ====")

try:
    build_info = tf.sysconfig.get_build_info()
    print("CUDA utilisé par TF :", build_info.get("cuda_version", "Unknown"))
except:
    print("Impossible de récupérer la version CUDA TF")

# =========================
# 3. CUDNN version
# =========================
print("\n==== cuDNN ====")

try:
    build_info = tf.sysconfig.get_build_info()
    print("cuDNN utilisé par TF :", build_info.get("cudnn_version", "Unknown"))
except:
    print("Impossible de récupérer la version cuDNN")

# =========================
# 4. GPU RAM via nvidia-smi
# =========================
print("\n==== Mémoire GPU ====")

try:
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader"
        ],
        encoding="utf-8"
    )

    for i, line in enumerate(result.strip().split("\n")):
        name, total, used, free = [x.strip() for x in line.split(",")]

        print(f"\nGPU {i}: {name}")
        print(f"  Total : {total} MB")
        print(f"  Utilisée : {used} MB")
        print(f"  Libre : {free} MB")

except Exception as e:
    print("⚠️ nvidia-smi indisponible :", e)

