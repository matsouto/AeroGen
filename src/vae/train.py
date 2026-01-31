# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import random
from pathlib import Path
from aerosandbox import Airfoil, KulfanAirfoil

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from src import vae
from src.vae import CSTVariationalAutoencoder
from src.utils import plot_original_and_reconstruction
from src.layers.airfoil_scaler import AirfoilScaler

from src.airfoil import airfoil_modifications

# ============================================================================
# CONFIGURATION AND RANDOM SEEDS
# ============================================================================
# Fixed seed for reproducibility across all random operations
SEED = 42
# Number of validation airfoils to visualize during training
AIRFOILS_TO_PLOT = 9

# Set seeds for all libraries to ensure reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
# Training configuration
EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
LATENT_DIM = 16  # Dimensionality of the latent space
NPV = 12  # Number of CST coefficients per surface (MUST match dataset generation)
LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
CLIPNORM = 1.0  # Gradient clipping norm to prevent exploding gradients
WARMUP_EPOCHS = 100  # Number of epochs for KL annealing warm-up
TARGET_BETA = 0.01  # Final weight for KL Divergence loss (reached after warmup)

HYPERPARAMETERS = {
    'epochs': EPOCHS,
    'latent_dim': LATENT_DIM,
    'learning_rate': LEARNING_RATE,
    'target_beta': TARGET_BETA,
    'warmup_epochs': WARMUP_EPOCHS,
    'batch_size': BATCH_SIZE,
    'clipnorm': CLIPNORM,
}

PROJECT_PATH = "./"  # Project root directory

# ============================================================================
# DATASET LOADING AND PREPARATION
# ============================================================================
# Load the Kulfan parameter dataset
dataset_path = Path(PROJECT_PATH) / "data" / "processed" / "kulfan_dataset_75.json"
print("Loading dataset...")
airfoil_dataset = pd.read_json(dataset_path)

# Convert coordinate strings to numpy arrays
airfoil_dataset["coordinates"] = airfoil_dataset["coordinates"].apply(lambda coords: np.array(coords))

# Extract and concatenate Kulfan parameters:
# [lower_weights(12), upper_weights(12), TE_thickness(1), leading_edge_weight(1)]
# Total: 26 parameters per airfoil
airfoil_data = airfoil_dataset["kulfan_parameters"].apply(
  lambda p: np.concatenate([
    p["lower_weights"],  # Lower surface CST weights
    p["upper_weights"],  # Upper surface CST weights
    [p["TE_thickness"]],  # Trailing edge thickness
    [p["leading_edge_weight"]]  # Leading edge weight
    ], axis=0)).to_numpy()

airfoil_data = np.stack(airfoil_data, axis=0).astype(np.float32)

# Split weights and parameters for normalization
raw_weights = airfoil_data[:, :-2]  # All CST coefficients (24 total)
raw_params = airfoil_data[:, -2:]  # TE thickness and leading edge weight

# Fit scaler on clean data
scaler = AirfoilScaler()
scaler.fit(raw_weights, raw_params)
print(f"Max Weight Value: {np.max(scaler.w_max)}")
print(f"Max Param Value: {np.max(scaler.p_max)}")

# Normalize the data to [-1, 1] range
normalized_data = scaler.transform(raw_weights, raw_params)

# Create TensorFlow dataset pipeline with shuffling and batching
train_dataset = tf.data.Dataset.from_tensor_slices(normalized_data)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
print(f"Dataset loaded and normalized: {len(normalized_data)} samples")
print(
    f"Data Range Check -> Min: {normalized_data.min():.2f}, Max: {normalized_data.max():.2f}"
)

# ============================================================================
# VALIDATION AIRFOILS PREPARATION
# ============================================================================
# Select first N airfoils for validation visualization during training
validation_airfoils_df = airfoil_dataset.iloc[:AIRFOILS_TO_PLOT].reset_index(drop=True)
# Create Airfoil objects for plotting reference
validation_airfoils = [Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"]) 
                       for af in validation_airfoils_df.to_dict(orient="records")]

# Extract Kulfan parameters from validation airfoils
validation_input = validation_airfoils_df["kulfan_parameters"].apply(
  lambda p: np.concatenate([
    p["lower_weights"],  # Lower surface weights
    p["upper_weights"],  # Upper surface weights
    [p["TE_thickness"]],  # Trailing edge thickness
    [p["leading_edge_weight"]]  # Leading edge weight
    ], axis=0)).to_list()

# Convert to tensor and normalize using the fitted scaler
validation_input = tf.convert_to_tensor(validation_input, dtype=tf.float32)
weights = validation_input[:, :24]  # Extract weights
params = validation_input[:, 24:]  # Extract parameters
validation_input = scaler.transform(weights, params)  # Normalize to [-1, 1]

# ============================================================================
# MODEL, OPTIMIZER, AND LOSS INITIALIZATION
# ============================================================================
# Initialize the VAE model
vae = CSTVariationalAutoencoder(scaler, npv=NPV, latent_dim=LATENT_DIM)

# Variables for early stopping and learning rate scheduling (optional)
best_loss = float('inf')
wait = 0
current_lr = LEARNING_RATE

# Adam optimizer with gradient clipping to prevent exploding gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, clipnorm=CLIPNORM)

# Reconstruction loss function (MAE is more robust than MSE for this task)
# reconstruction_loss = tf.keras.losses.MeanSquaredError()  # Alternative: MSE
reconstruction_loss = tf.keras.losses.MeanAbsoluteError()

# ============================================================================
# TRAINING STEP FUNCTION
# ============================================================================
@tf.function
def train_step(data, beta):
    """
    Runs one training step with Sum-Squared Error reconstruction loss.
    
    Args:
        data: Normalized airfoil parameters (Batch, 26)
        beta: Current KL divergence weight for annealing
    
    Returns:
        total_loss: Combined reconstruction + KL loss
        reco_loss: Reconstruction loss only
        kl_loss: KL divergence loss only
    """
    with tf.GradientTape() as tape:
        # Forward pass: training=True enables stochastic sampling from latent distribution
        reconstruction = vae(data, training=True)
        _, pred_weights, pred_params = reconstruction
        
        # Split ground truth: data shape is (Batch, 26)
        true_weights, true_params = tf.split(data, [2 * NPV, 2], axis=1)

        # Flatten predicted weights from (Batch, 2, 12) to (Batch, 24) for loss computation
        pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * NPV])

        # Compute reconstruction losses using Sum of Squared Errors
        loss_weights = tf.reduce_mean(tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1))
        loss_params = tf.reduce_mean(tf.reduce_sum(tf.square(true_params - pred_params), axis=1))
        
        # Combined reconstruction loss
        reco_loss = loss_weights + loss_params
        
        # KL divergence loss (computed via self.add_loss in the model)
        kl_loss = sum(vae.losses)

        # Total loss: reconstruction + annealed KL divergence
        total_loss = reco_loss + (beta * kl_loss)

    # Backpropagation
    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    
    return total_loss, reco_loss, kl_loss

# ============================================================================
# WEIGHTS & BIASES INITIALIZATION
# ============================================================================
# Initialize WandB for experiment tracking and logging
wandb.init(
    project="CSTVAE",
    config=HYPERPARAMETERS,
    name=f"VAE_{time.strftime('%Y%m%d-%H%M%S')}",  # Unique run name with timestamp
    notes="Dense Arch + Linear Output + Sum Loss + Scaler"
)

# Verbosity level for training output
VERBOSE = 1  # 0: silent, 1: print epoch info

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
# Create timestamped directories for model checkpoints and visualization images
models_path = Path(PROJECT_PATH) / "models" / "cstvae" / time.strftime("%Y%m%d-%H%M%S")
images_path = Path(PROJECT_PATH) / "images" / "cstvae" / time.strftime("%Y%m%d-%H%M%S")
os.makedirs(models_path, exist_ok=True)  # Create model save directory
os.makedirs(images_path, exist_ok=True)  # Create visualization output directory

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Initialize epoch metrics
    epoch_total_loss = tf.keras.metrics.Mean()
    epoch_reco_loss = tf.keras.metrics.Mean()
    epoch_kl_loss = tf.keras.metrics.Mean()

    # KL Annealing: gradually increase beta from 0 to TARGET_BETA over WARMUP_EPOCHS
    # This helps prevent posterior collapse and improves training stability
    if epoch < WARMUP_EPOCHS:
        BETA = TARGET_BETA * (epoch / WARMUP_EPOCHS)
    else:
        BETA = TARGET_BETA
    
    wandb.log({'beta': BETA})

    # Train on all batches for this epoch
    for x_batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # Run one training step and get losses
        total_loss, reco_loss, kl_loss = train_step(x_batch, BETA)
        
        # Update epoch-level metrics
        epoch_total_loss.update_state(total_loss)
        epoch_reco_loss.update_state(reco_loss)
        epoch_kl_loss.update_state(kl_loss)

    # Log epoch results
    elapsed_time = time.time() - start_time
    
    if VERBOSE > 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Time: {elapsed_time:.2f}s, "
              f"Total Loss: {epoch_total_loss.result():.4f}, "
              f"Reco Loss: {epoch_reco_loss.result():.4f}, "
              f"KL Loss: {epoch_kl_loss.result():.4f}")
    
    # Log metrics to WandB for monitoring
    wandb.log({
        'epoch_total_loss': epoch_total_loss.result(),
        'epoch_reconstruction_loss': epoch_reco_loss.result(),
        'epoch_kl_loss': epoch_kl_loss.result(),
    })

    # Validation and visualization: run inference on validation set
    val_input_tensor = tf.convert_to_tensor(validation_input)
    # training=False disables stochastic sampling for deterministic reconstruction
    _, reco_weights_norm, reco_params_norm = vae(val_input_tensor, training=False)

    # Denormalize weights and parameters back to physical range
    real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
        reco_weights_norm.numpy(), 
        reco_params_norm.numpy()
    )

    # Generate airfoil coordinates from denormalized Kulfan parameters
    w_tensor = tf.convert_to_tensor(real_reco_weights, dtype=tf.float32)
    p_tensor = tf.convert_to_tensor(real_reco_params, dtype=tf.float32)
    # CST transform produces (Batch, Points, 2) coordinates
    reco_coords = vae.decoder.cst_transform(w_tensor, p_tensor).numpy()

    # Convert coordinates to Airfoil objects for visualization
    reconstructed_airfoils = []
    for coords in reco_coords:
        reconstructed_airfoils.append(Airfoil(coordinates=coords))

    # Save visualization and model checkpoints periodically
    if (epoch + 1) % 10 == 0:
        # Optional: Save model weights at checkpoints
        # vae.save_weights(f"{models_path}/model_epoch_{epoch+1}.weights.h5")

        # Generate and save comparison plots of original vs reconstructed airfoils
        plot_original_and_reconstruction(
            validation_airfoils,  # Reference airfoils
            reconstructed_airfoils,  # VAE reconstructions
            text_label=f"Epoch: {epoch+1} / Elapsed Time: {elapsed_time:.2f}s",
            save_path=images_path,
            filename=f"reconstruction_epoch_{epoch+1}.png",
            show=False
        )