import os
import sys
import numpy as np

# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================
# Fix for tf2onnx: map missing np.object to the built-in object
if not hasattr(np, "object"):
    np.object = object # type: ignore

# Add project root to path for local imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.append(project_root)

import tensorflow as tf
import tf2onnx
import joblib
from pathlib import Path
from src.vae.vae import CSTVariationalAutoencoder

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
RUN_ID = "20260201-233018"    # Folder ID for the trained model
EPOCH = 500                   # Epoch checkpoint to load
NPV = 12                      # Number of CST parameters per side
LATENT_DIM = 16               # Latent space dimensions

MODELS_DIR = Path(project_root) / "models" / "cstvae" / RUN_ID
WEIGHTS_PATH = MODELS_DIR / "weights" / f"vae_weights_epoch_{EPOCH}.weights.h5"
SCALER_PATH = MODELS_DIR / "scaler" / "scaler.pkl"

OUTPUT_DIR = Path(project_root) / "exported_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_OUTPUT_PATH = OUTPUT_DIR / "vae_decoder.onnx"
SCALER_OUTPUT_PATH = OUTPUT_DIR / "vae_scaler.pkl"

print("\n" + "="*50)
print(f"🚀 EXPORTING DECODER TO ONNX")
print("="*50)

# ============================================================================
# 2. LOAD MODEL
# ============================================================================
print(f"\n[1/4] Loading Scaler...")
scaler = joblib.load(SCALER_PATH)
print("   ✓ OK.")

print(f"\n[2/4] Instantiating VAE...")
vae = CSTVariationalAutoencoder(scaler=scaler, npv=NPV, latent_dim=LATENT_DIM)
dummy_input = tf.zeros((1, (2 * NPV) + 2))
_ = vae(dummy_input) # Initialize the computational graph
print("   ✓ OK.")

print(f"\n[3/4] Loading Weights (Epoch {EPOCH})...")
vae.load_weights(WEIGHTS_PATH)
print("   ✓ OK.")

# ============================================================================
# 3. ONNX EXPORT
# ============================================================================
print(f"\n[4/4] Generating ONNX File...")

# Export a copy of the scaler for the RL environment
joblib.dump(scaler, SCALER_OUTPUT_PATH)

try:
    # 1. Define explicit input shape
    latent_input = tf.keras.Input(shape=(LATENT_DIM,), dtype=tf.float32, name="input_latent") # type: ignore
    
    # 2. Pass the input through the decoder
    decoder_outputs = vae.decoder(latent_input)
    
    # 3. Wrap in a static Functional Keras Model
    export_model = tf.keras.Model( # type: ignore
        inputs=latent_input, 
        outputs=decoder_outputs, 
        name="onnx_decoder"
    )
    
    # 4. Convert and save
    input_signature = (tf.TensorSpec((None, LATENT_DIM), tf.float32, name="input_latent"),) # type: ignore
    
    model_proto, _ = tf2onnx.convert.from_keras(
        export_model,
        input_signature=input_signature,
        opset=13,
        output_path=str(ONNX_OUTPUT_PATH)
    )
    print(f"   ✓ Success! ONNX saved at: exported_models/{ONNX_OUTPUT_PATH.name}")

except Exception as e:
    print(f"   ❌ Export error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ MODEL READY")
print("="*50)