import os
import numpy as np


def evaluate_predicted_gbsa(generator_, discriminator_, evaluation_data, scaler):
    """
    Evaluate the predicted GBSA scores for generated antibodies.
    Returns the mean predicted GBSA score IN ORIGINAL UNITS.
    """
    predicted_gbsa_scores = []
    # Keep track of the input and generated examples if a new best is found
    best_evalued_samples = {
        'ab_input': None,
        'ag_input': None,
        'generated_ab': None,
        'predicted_gbsa': float('inf')  # For tracking the best in this evaluation batch
    }

    # Iterate through the evaluation data
    for batch_idx, data_batch_ in enumerate(evaluation_data):
        if batch_idx >= 5:  # Limit to 5 batches for evaluation
            break

        (ab_data, ag_data), _ = data_batch_

        # Generate mutated antibodies
        generated_ab, generated_variety = generator_([ab_data, ag_data], training=False)

        # Get predicted GBSA from discriminator (still in scaled units)
        predicted_gbsa_scaled, _ = discriminator_([generated_ab, ag_data], training=False)

        # De-normalize the predicted GBSA scores
        predicted_gbsa_denormalized = scaler.inverse_transform(
            predicted_gbsa_scaled.numpy().reshape(-1, 1)).flatten()

        predicted_gbsa_scores.extend(predicted_gbsa_denormalized)

        # Find the best sample within this evaluation batch
        # Note: This takes the BEST from the 5 evaluation batches.
        # You might want to grab a random sample instead if the very best is not needed.
        min_gbsa_in_batch_idx = np.argmin(predicted_gbsa_denormalized)
        min_gbsa_in_batch = predicted_gbsa_denormalized[min_gbsa_in_batch_idx]

        if min_gbsa_in_batch < best_evalued_samples['predicted_gbsa']:
            best_evalued_samples['predicted_gbsa'] = min_gbsa_in_batch
            # Ensure these are numpy arrays for saving
            best_evalued_samples['ab_input'] = ab_data.numpy()[min_gbsa_in_batch_idx]
            best_evalued_samples['ag_input'] = ag_data.numpy()[min_gbsa_in_batch_idx]
            best_evalued_samples['generated_ab'] = generated_ab.numpy()[min_gbsa_in_batch_idx]

    mean_gbsa = np.mean(predicted_gbsa_scores) if predicted_gbsa_scores else float('inf')
    return mean_gbsa, best_evalued_samples


def save_best_generator(generator_, epoch_, predicted_gbsa_score, current_best_gbsa,
                        sample_data, save_dir_, feature_scaler_):  # Added sample_data and feature_scaler
    """
    Save full generator model and a sample if this is the best (most negative) predicted GBSA score.
    Returns the updated best_predicted_gbsa value.
    """
    if predicted_gbsa_score < current_best_gbsa:
        updated_best_gbsa = predicted_gbsa_score

        # Save the full model in Keras format
        best_model_path = os.path.join(save_dir_, 'best_generator.keras')
        generator_.save(best_model_path)

        print(f"NEW BEST Predicted GBSA (Denormalized): {updated_best_gbsa:.4f} - Full generator model saved!")

        # --- Save the sample antibody and antigen data ---
        sample_filepath = os.path.join(save_dir_, f'best_gbsa_sample_epoch_{epoch_}.npz')

        # Inverse transform coordinates for the sample data if they were scaled
        # The 'evaluate_predicted_gbsa' function passes in the scaled data.
        # So, we need to inverse transform ab_input, ag_input, generated_ab's coords.

        # Create copies to avoid modifying the original tensors/numpy arrays
        ab_input_original_coords = np.copy(sample_data['ab_input'])
        ag_input_original_coords = np.copy(sample_data['ag_input'])
        generated_ab_original_coords = np.copy(sample_data['generated_ab'])

        # Only inverse transform the first 3 features (x,y,z coordinates)
        ab_input_original_coords[..., :3] = feature_scaler_.inverse_transform(
            ab_input_original_coords[..., :3].reshape(-1, 3)
        ).reshape(ab_input_original_coords[..., :3].shape)

        ag_input_original_coords[..., :3] = feature_scaler_.inverse_transform(
            ag_input_original_coords[..., :3].reshape(-1, 3)
        ).reshape(ag_input_original_coords[..., :3].shape)

        generated_ab_original_coords[..., :3] = feature_scaler_.inverse_transform(
            generated_ab_original_coords[..., :3].reshape(-1, 3)
        ).reshape(generated_ab_original_coords[..., :3].shape)

        np.savez_compressed(
            sample_filepath,
            ab_input=ab_input_original_coords,
            ag_input=ag_input_original_coords,
            generated_ab=generated_ab_original_coords,
            predicted_gbsa=sample_data['predicted_gbsa']  # This is already denormalized
        )
        print(f"Sample data for best GBSA saved to {sample_filepath}")

        # Save info
        info_path = os.path.join(save_dir_, 'best_model_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Best Predicted GBSA Score (Denormalized): {updated_best_gbsa:.4f}\n")
            f.write(f"Achieved at Epoch: {epoch_ + 1}\n")
            f.write(f"Model saved to: {best_model_path}\n")
            f.write(f"Sample data saved to: {sample_filepath}\n")

        return updated_best_gbsa

    else:
        return current_best_gbsa
