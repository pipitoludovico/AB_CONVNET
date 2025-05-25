import numpy as np
import tensorflow as tf
from keras import optimizers, losses
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorflow.keras.models import load_model

from Model.Models import AutoregressiveGeneratorConditioned


def TrainAndGenerate(pretrained_discriminator_model_path="./best_model.keras"):
    # --- Load e preprocess ---
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

    label_scaler = StandardScaler()
    gbsa_scaled = label_scaler.fit_transform(gbsa)

    validity_labels = np.ones_like(gbsa_scaled)

    # --- Dataset ---
    batch_size = 64
    dataset = tf.data.Dataset.from_tensor_slices((
        {'ab_input': ab, 'ag_input': ag},
        {'gbsa_prediction': gbsa_scaled, 'validity': validity_labels}
    ))
    train_dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    generator = AutoregressiveGeneratorConditioned(embedding_dim=128)

    # check if discriminator exists else load the one I pretrained
    discriminator = None
    if pretrained_discriminator_model_path:
        discriminator = load_model(pretrained_discriminator_model_path, compile=False, safe_mode=False)
        discriminator.trainable = False
    else:
        from Model.Models import Discriminator
        discriminator = Discriminator(ab_shape=ab.shape[1:], ag_shape=ag.shape[1:])

    # trying with two optimizers...to tune eventually?
    gen_optimizer = optimizers.Adam(1e-2)
    disc_optimizer = optimizers.Adam(1e-2)

    # losses
    bce_loss = losses.BinaryCrossentropy(from_logits=False)
    mse_loss = losses.MeanSquaredError()

    # @tf.function
    # def train_step(batch_ag_, batch_ab_real, batch_gbsa_):
    #     batch_size_ = tf.shape(batch_ag_)[0]
    #     # valid = tf.ones((batch_size_, 1))
    #     # fake = tf.zeros((batch_size_, 1))
    #     valid = tf.ones((batch_size_, 1)) * 0.9
    #     fake = tf.zeros((batch_size_, 1)) + 0.1
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         ab_fake = generator(batch_ag_, batch_ab_real, training=True)
    #
    #         # Always compute predictions
    #         gbsa_real_pred, validity_real = discriminator([batch_ab_real, batch_ag_], training=False)
    #         gbsa_fake_pred, validity_fake = discriminator([ab_fake, batch_ag_], training=False)
    #
    #         # Compute Discriminator loss
    #         disc_loss_real = bce_loss(valid, validity_real)
    #         disc_loss_fake = bce_loss(fake, validity_fake)
    #         disc_gbsa_loss = mse_loss(batch_gbsa_, gbsa_real_pred)
    #         disc_loss = disc_loss_real + disc_loss_fake + disc_gbsa_loss
    #         # should I split these into more?
    #
    #         # Compute Generator loss
    #         gen_adv_loss = bce_loss(valid, validity_fake)
    #
    #         # Want to encourage more negative GBSA (lower = better), so loss is positive gbsa
    #         gen_gbsa_loss = tf.reduce_mean(gbsa_fake_pred)  # average over batch
    #         gen_loss = gen_adv_loss + 0.1 * gen_gbsa_loss  # adjust 0.1 as a weight hyperparameter
    #
    #     # Apply Generator gradients
    #     gradients_gen = tape.gradient(gen_loss, generator.trainable_variables)
    #     gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    #
    #     if not pretrained_discriminator_model_path:
    #         gradients_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
    #         disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
    #
    #     return {
    #         "gen_loss": gen_loss,
    #         "disc_loss_total": disc_loss,
    #         "disc_loss_real": disc_loss_real,
    #         "disc_loss_fake": disc_loss_fake,
    #         "disc_gbsa_loss": disc_gbsa_loss,
    #         "gen_adv_loss": gen_adv_loss
    #     }
    def train_step(batch_ag_, batch_ab_real, batch_gbsa_):
        batch_size_ = tf.shape(batch_ag_)[0]

        # Label smoothing for better training stability
        valid = tf.ones((batch_size_, 1)) * 0.9
        fake = tf.zeros((batch_size_, 1)) + 0.1

        with tf.GradientTape(persistent=True) as tape:
            ab_fake = generator(batch_ag_, batch_ab_real, training=True)

            # Add noise to discriminator inputs for regularization
            noise_std = 0.05
            ab_real_noisy = batch_ab_real + tf.random.normal(tf.shape(batch_ab_real), stddev=noise_std)
            ab_fake_noisy = ab_fake + tf.random.normal(tf.shape(ab_fake), stddev=noise_std)

            gbsa_real_pred, validity_real = discriminator([ab_real_noisy, batch_ag_], training=False)
            gbsa_fake_pred, validity_fake = discriminator([ab_fake_noisy, batch_ag_], training=False)

            # Discriminator losses with gradient penalty
            disc_loss_real = bce_loss(valid, validity_real)
            disc_loss_fake = bce_loss(fake, validity_fake)
            disc_gbsa_loss = mse_loss(batch_gbsa_, gbsa_real_pred)
            disc_loss = disc_loss_real + disc_loss_fake + disc_gbsa_loss

            # Improved generator loss with multiple objectives
            gen_adv_loss = bce_loss(valid, validity_fake)

            # GBSA optimization (encourage lower values)
            gen_gbsa_loss = tf.reduce_mean(gbsa_fake_pred)

            # Feature matching loss (compare statistics of real vs fake)
            real_features = tf.reduce_mean(batch_ab_real, axis=[1, 2])
            fake_features = tf.reduce_mean(ab_fake, axis=[1, 2])
            feature_matching_loss = mse_loss(real_features, fake_features)

            # Diversity loss to prevent mode collapse
            batch_diversity = tf.reduce_mean(tf.nn.l2_normalize(ab_fake, axis=-1))
            diversity_loss = -tf.reduce_mean(tf.square(batch_diversity - 0.5))

            # Combined generator loss
            gen_loss = (gen_adv_loss +
                        0.1 * gen_gbsa_loss +
                        0.05 * feature_matching_loss +
                        0.02 * diversity_loss)

        # Apply gradients with gradient clipping
        gradients_gen = tape.gradient(gen_loss, generator.trainable_variables)
        gradients_gen = [tf.clip_by_norm(g, 1.0) for g in gradients_gen]
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

        if not pretrained_discriminator_model_path:
            gradients_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
            gradients_disc = [tf.clip_by_norm(g, 1.0) for g in gradients_disc]
            disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

        return {
            "gen_loss": gen_loss,
            "disc_loss_total": disc_loss,
            "gen_adv_loss": gen_adv_loss,
            "gen_gbsa_loss": gen_gbsa_loss,
            "feature_matching_loss": feature_matching_loss,
            "diversity_loss": diversity_loss
        }

    #  training
    epochs = 50
    best_score = np.inf
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        gen_losses = []
        disc_losses = []

        train_bar = tqdm(train_dataset, desc="Training", unit="batch")
        for batch in train_bar:
            inputs, targets = batch
            batch_ab = inputs['ab_input']
            batch_ag = inputs['ag_input']
            batch_gbsa = targets['gbsa_prediction']

            losses_dict = train_step(batch_ag, batch_ab, batch_gbsa)
            g_loss_val = losses_dict['gen_loss'].numpy()
            d_loss_val = losses_dict['disc_loss_total'].numpy()

            gen_losses.append(g_loss_val)
            disc_losses.append(d_loss_val)

            train_bar.set_postfix({
                key: f"{value.numpy():.4f}" for key, value in losses_dict.items()
            })

            if losses_dict['gen_adv_loss'].numpy() < best_score:
                generator.save('best_generator.keras', overwrite=True)
                best_score = losses_dict['gen_adv_loss'].numpy()

        print(f"Train - Gen loss: {np.mean(gen_losses):.4f} - Disc loss: {np.mean(disc_losses):.4f}")
