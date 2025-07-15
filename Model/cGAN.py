from keras import Model, metrics
import tensorflow as tf


class cGAN(Model):
    # Pass label_scaler as an argument to __init__
    def __init__(self, discriminator, generator, label_scaler):
        super().__init__()
        self.loss_fn = None
        self.discriminator = discriminator
        self.generator = generator
        self.g_optimizer = None
        self.d_optimizer = None
        self.validity_weight = 5

        # Additional metrics for discriminator
        self.d_gbsa_loss_metric = tf.keras.metrics.Mean(name="d_gbsa_loss")
        self.d_validity_loss_metric = tf.keras.metrics.Mean(name="d_validity_loss")
        self.d_validity_real_metric = tf.keras.metrics.Mean(name="d_validity_real")

        # Additional metrics for generator
        self.g_gbsa_loss_metric = tf.keras.metrics.Mean(name="g_gbsa_loss")
        self.g_validity_loss_metric = tf.keras.metrics.Mean(name="g_validity_loss")
        self.g_validity_fake_metric = tf.keras.metrics.Mean(name="gen_validity_fake")

        # Store the label_scaler for de-normalization
        self.label_scaler = label_scaler

        # Metrics for training progress
        self.d_loss_metric = metrics.Mean(name='d_loss')
        self.g_loss_metric = metrics.Mean(name='g_loss')
        self.d_gbsa_mae_metric = metrics.MeanAbsoluteError(name='d_gbsa_mae')

        # Metric for the scaled GBSA (what the model directly optimizes)
        self.g_fake_gbsa_mean_scaled_metric = metrics.Mean(name='gen_fake_gbsa_mean_scaled')

        # NEW: Metric for the DENORMALIZED GBSA score
        self.g_fake_gbsa_mean_denormalized_metric = metrics.Mean(name='gen_fake_gbsa_mean_denormalized')

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        # Return all metrics you want to track
        return [self.d_loss_metric,
                self.g_loss_metric,
                self.d_gbsa_mae_metric,
                self.g_fake_gbsa_mean_scaled_metric,
                self.g_fake_gbsa_mean_denormalized_metric]  # Include the new metric

    @tf.function
    def train_step(self, data):
        (real_ab, real_ag), real_gbsa = data  # real_gbsa here is already scaled by label_scaler

        # --- Train Discriminator (GBSA Predictor) ---
        with tf.GradientTape() as tape:
            predicted_gbsa_real, validity_real = self.discriminator((real_ab, real_ag), training=True)

            # GBSA prediction loss
            d_gbsa_loss = self.loss_fn(real_gbsa, predicted_gbsa_real)

            # Validity loss (real antibodies should have high validity ~1.0)
            real_validity_targets = tf.ones_like(validity_real)  # Target validity = 1.0 for real antibodies
            d_validity_loss = self.loss_fn(real_validity_targets, validity_real)

            # Combined discriminator loss
            d_loss = d_gbsa_loss + self.validity_weight * d_validity_loss

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Update discriminator metrics
        self.d_loss_metric.update_state(d_loss)
        self.d_gbsa_loss_metric.update_state(d_gbsa_loss)
        self.d_validity_loss_metric.update_state(d_validity_loss)
        self.d_gbsa_mae_metric.update_state(real_gbsa, predicted_gbsa_real)
        self.d_validity_real_metric.update_state(validity_real)

        # --- Train Generator ---
        with tf.GradientTape() as genTape:
            fake_ab = self.generator([real_ab, real_ag], training=True)
            predicted_gbsa_fake, validity_fake = self.discriminator([fake_ab, real_ag], training=False)

            # GBSA improvement loss (maximize predicted GBSA)
            g_gbsa_loss = tf.reduce_mean(predicted_gbsa_fake)

            # Validity loss (generated antibodies should also have high validity)
            fake_validity_targets = tf.ones_like(validity_fake)
            g_validity_loss = -self.loss_fn(fake_validity_targets,
                                            validity_fake)  # Negative because we want to minimize this

            # Combined generator loss
            g_loss = g_gbsa_loss + self.validity_weight * g_validity_loss

        g_grads = genTape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Update generator metrics (scaled)
        self.g_loss_metric.update_state(g_loss)
        self.g_gbsa_loss_metric.update_state(g_gbsa_loss)
        self.g_validity_loss_metric.update_state(g_validity_loss)
        self.g_fake_gbsa_mean_scaled_metric.update_state(predicted_gbsa_fake)
        self.g_validity_fake_metric.update_state(validity_fake)

        # Denormalize GBSA for interpretability
        denormalized_gbsa_fake = tf.numpy_function(
            func=lambda x: self.label_scaler.inverse_transform(x.reshape(-1, 1)).flatten(),
            inp=[predicted_gbsa_fake],
            Tout=tf.float32,
            name='denormalize_gbsa_fake_op',
        )
        denormalized_gbsa_fake.set_shape([None])

        self.g_fake_gbsa_mean_denormalized_metric.update_state(denormalized_gbsa_fake)

        return {
            "d_loss": self.d_loss_metric.result(),
            "d_gbsa_loss": self.d_gbsa_loss_metric.result(),
            "d_validity_loss": self.d_validity_loss_metric.result(),
            "d_gbsa_mae": self.d_gbsa_mae_metric.result(),
            "d_validity_real": self.d_validity_real_metric.result(),

            "g_loss": self.g_loss_metric.result(),
            "g_gbsa_loss": self.g_gbsa_loss_metric.result(),
            "g_validity_loss": self.g_validity_loss_metric.result(),
            "gen_fake_gbsa_mean_scaled": self.g_fake_gbsa_mean_scaled_metric.result(),
            "gen_fake_gbsa_mean_denormalized": self.g_fake_gbsa_mean_denormalized_metric.result(),
            "gen_validity_fake": self.g_validity_fake_metric.result(),
        }
