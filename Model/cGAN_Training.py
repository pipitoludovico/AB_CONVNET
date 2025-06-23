import numpy as np
import tensorflow as tf
from keras import optimizers, losses
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorflow.keras.models import load_model
import logging

from Model.Models import Generator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def TrainAndGenerate(pretrained_discriminator_model_path="./best_model.keras"):
    # --- Load and preprocess ---
    data = np.load("./matrices/padded_dataset.npz", allow_pickle=True)
    ab = data['ab']
    ag = data['ag']
    gbsa = data['gbsa'].reshape(-1, 1)

    continuous_idx = slice(0, 3)
    ab_cont = ab[..., continuous_idx].reshape(-1, 3)
    ag_cont = ag[..., continuous_idx].reshape(-1, 3)

    feature_scaler = StandardScaler()
    feature_scaler.fit(np.vstack([ab_cont, ag_cont]))

    ab[..., continuous_idx] = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
    ag[..., continuous_idx] = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

    ab = np.expand_dims(ab, axis=3)
    ag = np.expand_dims(ag, axis=3)

    label_scaler = StandardScaler()
    gbsa_scaled = label_scaler.fit_transform(gbsa)

    validity_labels = np.ones_like(gbsa_scaled)

    # --- Dataset ---
    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices((
        {'ab_input': ab, 'ag_input': ag},
        {'gbsa_prediction': gbsa_scaled, 'validity': validity_labels}
    ))
    train_dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    generator = Generator()

    # Load or create discriminator
    discriminator = None
    if pretrained_discriminator_model_path:
        try:
            discriminator = load_model(pretrained_discriminator_model_path, compile=False, safe_mode=False)
            discriminator.trainable = False
            logger.info("Loaded pretrained discriminator")
        except Exception as e:
            logger.warning(f"Failed to load discriminator: {e}, creating new one")
            from Model.Models import Discriminator
            discriminator = Discriminator(ab_shape=ab.shape[1:], ag_shape=ag.shape[1:])
    else:
        from Model.Models import Discriminator
        discriminator = Discriminator(ab_shape=ab.shape[1:], ag_shape=ag.shape[1:])

    # More balanced learning rates
    gen_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
    disc_optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    # Reduced label smoothing
    bce_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)
    mse_loss = losses.MeanSquaredError()

    # Remove handicaps - use proper GAN balance instead
    # Track training dynamics for adaptive scheduling
    d_loss_history = []
    g_loss_history = []
    warmup_steps = tf.Variable(0, trainable=False, dtype=tf.int32)

    def check_for_nan(tensor, name):
        """Check for NaN values and log warnings"""
        if tf.reduce_any(tf.math.is_nan(tensor)):
            logger.error(f"NaN detected in {name}")
            return True
        return False

    def add_noise_to_discriminator_inputs(real_ab, fake_ab, noise_level=0.05):
        """Add controlled noise to discriminator inputs"""
        noise_real = tf.random.normal(tf.shape(real_ab)) * noise_level
        noise_fake = tf.random.normal(tf.shape(fake_ab)) * noise_level
        return real_ab + noise_real, fake_ab + noise_fake

    def train_generator_step(batch_ag_, batch_ab_real, batch_gbsa_):
        """Generator training with authentic challenge"""
        with tf.GradientTape() as gen_tape:
            ab_fake = generator(batch_ag_, batch_ab_real, training=True)
            gbsa_fake_pred, validity_fake = discriminator([ab_fake, batch_ag_], training=False)

            if check_for_nan(ab_fake, "Generated antibodies"):
                return None

            # Adversarial loss - no artificial boost
            target_labels = tf.ones_like(validity_fake) * 0.9
            gen_adv_loss = bce_loss(target_labels, validity_fake)

            # GBSA optimization - this is your actual objective
            gen_gbsa_loss = -tf.reduce_mean(gbsa_fake_pred)
            real_gbsa_mean = tf.reduce_mean(batch_gbsa_)
            gbsa_improvement = tf.reduce_mean(gbsa_fake_pred) - real_gbsa_mean

            # Feature matching for structural plausibility
            ab_real_flat = tf.reshape(batch_ab_real, [-1, 34])
            ab_fake_flat = tf.reshape(ab_fake, [-1, 34])

            real_mean = tf.reduce_mean(ab_real_flat, axis=0)
            fake_mean = tf.reduce_mean(ab_fake_flat, axis=0)
            real_std = tf.math.reduce_std(ab_real_flat, axis=0) + 1e-8
            fake_std = tf.math.reduce_std(ab_fake_flat, axis=0) + 1e-8

            feature_matching_loss = (tf.reduce_mean(tf.square(real_mean - fake_mean)) +
                                     tf.reduce_mean(tf.square(real_std - fake_std)))

            # Diversity loss
            diversity_loss = tf.cond(
                tf.shape(ab_fake)[0] > 1,
                lambda: compute_diversity_loss(tf.reshape(ab_fake, [tf.shape(ab_fake)[0], -1])),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

            # Balanced loss - GBSA improvement is primary objective
            gen_loss = (0.6 * gen_adv_loss +  # Moderate adversarial pressure
                        1.0 * gen_gbsa_loss +  # Primary objective
                        0.3 * feature_matching_loss +  # Structural validity
                        0.1 * diversity_loss)  # Encourage variety

            if check_for_nan(gen_loss, "Generator loss"):
                return None

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gen_gradients]

        valid_gradients = [(g, v) for g, v in zip(gen_gradients, generator.trainable_variables) if g is not None]
        if valid_gradients:
            gen_optimizer.apply_gradients(valid_gradients)

        return {
            "gen_loss": gen_loss,
            "gen_realism": tf.reduce_mean(validity_fake),
            "gbsa_improvement": gbsa_improvement,
            "gen_adv_loss": gen_adv_loss,
            "feature_matching": feature_matching_loss,
            "gen_gbsa_loss": gen_gbsa_loss
        }

    def compute_diversity_loss(ab_fake_batch):
        """Compute diversity loss with numerical stability"""
        expanded_a = tf.expand_dims(ab_fake_batch, 1)
        expanded_b = tf.expand_dims(ab_fake_batch, 0)
        pairwise_diff = expanded_a - expanded_b
        pairwise_dist = tf.reduce_sum(tf.square(pairwise_diff), axis=2)

        # Add small epsilon to prevent division by zero
        mean_dist = tf.reduce_mean(pairwise_dist) + 1e-8
        return -tf.math.log(mean_dist)

    def train_discriminator_step(batch_ag_, batch_ab_real, batch_gbsa_):
        """Discriminator training step with controlled difficulty"""
        batch_size_ = tf.shape(batch_ag_)[0]

        # Generate fake samples
        ab_fake = generator(batch_ag_, batch_ab_real, training=False)
        ab_fake = tf.stop_gradient(ab_fake)

        # Add moderate noise
        ab_real_noisy, ab_fake_noisy = add_noise_to_discriminator_inputs(
            batch_ab_real, ab_fake, noise_level=0.1
        )

        # More reasonable label smoothing
        real_labels = tf.ones((batch_size_, 1)) * 0.9
        fake_labels = tf.zeros((batch_size_, 1)) * 0.1

        # Occasional label flipping (reduced probability)
        label_flip_prob = 0.1
        should_flip = tf.random.uniform([batch_size_, 1]) < label_flip_prob
        real_labels = tf.where(should_flip, fake_labels, real_labels)

        with tf.GradientTape() as disc_tape:
            gbsa_real_pred, validity_real = discriminator([ab_real_noisy, batch_ag_], training=True)
            gbsa_fake_pred, validity_fake = discriminator([ab_fake_noisy, batch_ag_], training=True)

            # Check for NaN in discriminator outputs
            if check_for_nan(validity_real, "Discriminator real validity") or \
                    check_for_nan(validity_fake, "Discriminator fake validity"):
                return None

            # Discriminator losses with handicap
            disc_loss_real = bce_loss(real_labels, validity_real)
            disc_loss_fake = bce_loss(fake_labels, validity_fake)
            disc_gbsa_loss = mse_loss(batch_gbsa_, gbsa_real_pred)

            disc_loss = disc_loss_real + disc_loss_fake + disc_gbsa_loss

            if check_for_nan(disc_loss, "Discriminator loss"):
                return None

        # Apply gradients with conservative clipping
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_gradients = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in disc_gradients]

        valid_gradients = [(g, v) for g, v in zip(disc_gradients, discriminator.trainable_variables) if g is not None]
        if valid_gradients:
            disc_optimizer.apply_gradients(valid_gradients)

        # Calculate metrics
        disc_real_accuracy = tf.reduce_mean(tf.cast(validity_real > 0.5, tf.float32))
        disc_fake_accuracy = tf.reduce_mean(tf.cast(validity_fake < 0.5, tf.float32))
        disc_overall_accuracy = (disc_real_accuracy + disc_fake_accuracy) / 2.0

        return {
            "disc_loss": disc_loss,
            "disc_accuracy": disc_overall_accuracy,
            "disc_real_acc": disc_real_accuracy,
            "disc_fake_acc": disc_fake_accuracy,
            "gbsa_pred_rmse": tf.sqrt(mse_loss(batch_gbsa_, gbsa_real_pred))
        }

    def should_train_discriminator(d_loss_history, g_loss_history, step):
        """Adaptive training schedule based on loss dynamics"""
        if step < 100:
            return step % 3 == 0  # Train discriminator every 3rd step initially

        if len(d_loss_history) < 10 or len(g_loss_history) < 10:
            return step % 2 == 0

        # Check recent loss trends
        recent_d_loss = np.mean(d_loss_history[-10:])
        recent_g_loss = np.mean(g_loss_history[-10:])

        # If discriminator is losing badly, train it more
        if recent_d_loss > recent_g_loss * 2:
            return True
        # If generator is struggling, train discriminator less
        elif recent_g_loss > recent_d_loss * 2:
            return step % 3 == 0
        else:
            return step % 2 == 0

    # Training loop
    epochs = 50
    best_realism = 0.0
    step_count = 0
    consecutive_nan_count = 0
    max_consecutive_nans = 5

    logger.info("Starting improved GAN training...")

    for epoch in range(epochs):
        logger.info(f"EPOCH {epoch + 1}/{epochs}")

        train_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}", ncols=180)
        epoch_gen_losses = []
        epoch_disc_losses = []

        for batch in train_bar:
            inputs, targets = batch
            batch_ab = inputs['ab_input']
            batch_ag = inputs['ag_input']
            batch_gbsa = targets['gbsa_prediction']

            step_count += 1

            # Train generator every step
            gen_metrics = train_generator_step(batch_ag, batch_ab, batch_gbsa)
            if gen_metrics is not None:
                epoch_gen_losses.append(float(gen_metrics["gen_loss"]))
                g_loss_history.append(float(gen_metrics["gen_loss"]))
                consecutive_nan_count = 0
            else:
                consecutive_nan_count += 1
                if consecutive_nan_count >= max_consecutive_nans:
                    logger.error("Too many consecutive NaN values, stopping training")
                    return generator, discriminator

            # Train discriminator based on adaptive schedule
            disc_metrics = None
            if should_train_discriminator(d_loss_history, g_loss_history, step_count):
                disc_metrics = train_discriminator_step(batch_ag, batch_ab, batch_gbsa)
                if disc_metrics is not None:
                    epoch_disc_losses.append(float(disc_metrics["disc_loss"]))
                    d_loss_history.append(float(disc_metrics["disc_loss"]))

            if disc_metrics is None:
                # Get metrics without training
                ab_fake = generator(batch_ag, batch_ab, training=False)
                _, validity_real = discriminator([batch_ab, batch_ag], training=False)
                _, validity_fake = discriminator([ab_fake, batch_ag], training=False)

                disc_metrics = {
                    "disc_accuracy": (tf.reduce_mean(tf.cast(validity_real > 0.5, tf.float32)) +
                                      tf.reduce_mean(tf.cast(validity_fake < 0.5, tf.float32))) / 2.0,
                    "disc_loss": tf.constant(0.0),
                    "gbsa_pred_rmse": tf.constant(0.0)
                }

            # Keep history manageable
            if len(d_loss_history) > 100:
                d_loss_history = d_loss_history[-50:]
            if len(g_loss_history) > 100:
                g_loss_history = g_loss_history[-50:]

            # Update progress bar - focus on GBSA improvement as primary metric
            if gen_metrics is not None:
                train_bar.set_postfix({
                    "GBSA_Imp": f"{float(gen_metrics['gbsa_improvement']):.3f}",
                    "G_Real": f"{float(gen_metrics['gen_realism']):.3f}",
                    "D_Acc": f"{float(disc_metrics['disc_accuracy']):.3f}",
                    "G_GBSA": f"{float(gen_metrics['gen_gbsa_loss']):.3f}",
                    "Step": step_count
                })

                # Save best generator based on GBSA improvement, not fake realism
                current_gbsa_improvement = float(gen_metrics['gbsa_improvement'])
                if current_gbsa_improvement > best_realism:  # Reusing variable name
                    generator.save('best_generator.keras', overwrite=True)
                    best_realism = current_gbsa_improvement
                    logger.info(f"New best GBSA improvement: {best_realism:.4f} at step {step_count}")

        # Epoch summary
        if epoch_gen_losses:
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses) if epoch_disc_losses else 0
            logger.info(f"Epoch {epoch + 1} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

    logger.info(f"Training completed! Best GBSA improvement: {best_realism:.4f}")
    return generator, discriminator
