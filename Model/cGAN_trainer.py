import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from Model.Models import Generator, Discriminator
from Model.cGAN import cGAN

from Model.utils import save_best_generator, evaluate_predicted_gbsa


def TrainAndGenerate(pretrained_discriminator_model_path=None,
                     dataset_path="./matrices/padded_dataset.npz",
                     epochs=100, batch_size=32,
                     save_dir="./saved_models"):
    best_predicted_gbsa = 1000.0  # Initialize with a high value for minimization

    # Load and preprocess data
    print("Loading data")
    data = np.load(dataset_path, allow_pickle=True)
    ab = data['ab']  # shape (batch, 92, 5, 30)
    ag = data['ag']  # shape (batch, 97, 5, 30)
    gbsa = data['gbsa'].reshape(-1, 1)
    print("PRE SCALING")
    print(ab[0][-1])

    # Create masks for non-padded positions BEFORE scaling
    # A position is considered padded if ALL features are zero
    ab_mask = np.sum(np.abs(ab), axis=-1) > 0  # shape (batch, 92, 5)
    ag_mask = np.sum(np.abs(ag), axis=-1) > 0  # shape (batch, 97, 5)

    # Scale only x, y, z coordinates, but only for NON-PADDED positions
    continuous_idx = slice(0, 3)

    # Extract coordinates from non-padded positions only
    ab_coords_nonpadded = ab[..., continuous_idx][ab_mask]  # 1D array of valid coordinates
    ag_coords_nonpadded = ag[..., continuous_idx][ag_mask]  # 1D array of valid coordinates

    # Fit scaler only on valid (non-padded) coordinates
    feature_scaler = StandardScaler()
    all_valid_coords = np.vstack([ab_coords_nonpadded, ag_coords_nonpadded])
    feature_scaler.fit(all_valid_coords)

    # Apply scaling only to non-padded positions
    # Create copies to avoid modifying original data
    ab_scaled = ab.copy()
    ag_scaled = ag.copy()

    # Scale AB coordinates
    ab_coords_flat = ab[..., continuous_idx].reshape(-1, 3)
    ab_mask_flat = ab_mask.reshape(-1)
    ab_coords_flat[ab_mask_flat] = feature_scaler.transform(ab_coords_flat[ab_mask_flat])
    ab_scaled[..., continuous_idx] = ab_coords_flat.reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)

    # Scale AG coordinates
    ag_coords_flat = ag[..., continuous_idx].reshape(-1, 3)
    ag_mask_flat = ag_mask.reshape(-1)
    ag_coords_flat[ag_mask_flat] = feature_scaler.transform(ag_coords_flat[ag_mask_flat])
    ag_scaled[..., continuous_idx] = ag_coords_flat.reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

    # Update ab and ag with scaled versions
    ab = ab_scaled
    ag = ag_scaled

    print("AFTER SCALING")
    print(ab[0][-1])
    label_scaler = StandardScaler()
    gbsa_scaled = label_scaler.fit_transform(gbsa)

    # Create two datasets, one with scaled features for training, one with original for saving
    dataset_scaled = tf.data.Dataset.from_tensor_slices(((ab, ag), gbsa_scaled)).shuffle(512).batch(
        batch_size).prefetch(
        tf.data.AUTOTUNE)

    # Initialize models
    discriminator = Discriminator()
    generator = Generator()

    if pretrained_discriminator_model_path:
        discriminator.load_weights(pretrained_discriminator_model_path)

    # Instantiate and compile cGAN - Pass the label_scaler here!
    print("Instantiating the cGAN")
    gan = cGAN(discriminator=discriminator, generator=generator, label_scaler=label_scaler)
    print("Now compiling cGAN")
    gan.compile(
        d_optimizer=Adam(learning_rate=0.0001),
        g_optimizer=Adam(learning_rate=0.0001),
    )

    os.makedirs(save_dir, exist_ok=True)

    # Use the scaled dataset for evaluation
    evaluation_dataset_scaled = dataset_scaled.take(10)  # For passing to evaluate_predicted_gbsa

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reset metrics at the start of each epoch
        gan.d_loss_metric.reset_state()
        gan.g_loss_metric.reset_state()
        gan.d_gbsa_mae_metric.reset_state()
        gan.g_fake_gbsa_mean_scaled_metric.reset_state()
        gan.g_fake_gbsa_mean_denormalized_metric.reset_state()

        for step, data_batch in enumerate(dataset_scaled):  # Use scaled dataset for training
            logs = gan.train_step(data_batch)

            # Print current step logs (if you want more frequent updates)
            if step % 100 == 0:
                print(f"Step {int(step)}: "
                      f"d_loss={logs['d_loss']:.4f} "
                      f"g_gbsa_loss={logs['g_gbsa_loss']:.4f} "
                      f"inter_batch_diversity={logs['inter_batch_diversity']:.4f} ")

        # End of epoch evaluation
        print(f"Evaluating predicted GBSA at end of epoch {epoch + 1}...")

        # Get mean GBSA and the best sample found during evaluation
        current_predicted_gbsa_denormalized_mean, best_eval_sample = evaluate_predicted_gbsa(
            generator, discriminator, evaluation_dataset_scaled, label_scaler
        )
        print(
            f"Mean predicted GBSA for generated antibodies (Denormalized): {current_predicted_gbsa_denormalized_mean:.4f}")

        # Save if this is the best, passing the best_eval_sample
        old_best_gbsa = best_predicted_gbsa
        best_predicted_gbsa = save_best_generator(generator_=generator, epoch_=epoch,
                                                  predicted_gbsa_score=current_predicted_gbsa_denormalized_mean,
                                                  current_best_gbsa=best_predicted_gbsa, sample_data=best_eval_sample,
                                                  save_dir_=save_dir, feature_scaler_=feature_scaler)

        if best_predicted_gbsa == old_best_gbsa:
            print(f"No improvement. Best predicted GBSA (Denormalized) remains: {best_predicted_gbsa:.4f}")

    print(f"\nTraining completed!")
    print(f"Best predicted GBSA score achieved (Denormalized): {best_predicted_gbsa:.4f}")
