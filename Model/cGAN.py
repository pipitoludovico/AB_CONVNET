from keras import Model, metrics
import tensorflow as tf
from .Custom_Layers import DiversityCalculator, Separator

from keras import Model, metrics
import tensorflow as tf
from .Custom_Layers import DiversityCalculator, Separator


class cGAN(Model):
    def __init__(self, discriminator, generator, label_scaler,
                 diversity_weight=1.0, property_weight=0.5, inter_batch_diversity_weight=1.0):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.label_scaler = label_scaler

        # Loss weights
        self.diversity_weight = diversity_weight
        self.property_weight = property_weight
        self.inter_batch_diversity_weight = inter_batch_diversity_weight

        # Loss function
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()

        # Optimizers will be set during compile
        self.g_optimizer = None
        self.d_optimizer = None

        # Create metric containers
        self._build_metrics()

    def _build_metrics(self):
        # Discriminator metrics
        self.d_loss_metric = metrics.Mean(name='d_loss')
        self.d_gbsa_loss_metric = metrics.Mean(name='d_gbsa_loss')
        self.d_diversity_loss_metric = metrics.Mean(name='d_diversity_loss')
        self.d_gbsa_mae_metric = metrics.MeanAbsoluteError(name='d_gbsa_mae')

        # Generator metrics
        self.g_loss_metric = metrics.Mean(name='g_loss')
        self.g_gbsa_loss_metric = metrics.Mean(name='g_gbsa_loss')
        self.g_diversity_loss_metric = metrics.Mean(name='g_diversity_loss')
        self.g_property_loss_metric = metrics.Mean(name='g_property_loss')

        # GBSA metrics (scaled and denormalized)
        self.g_fake_gbsa_mean_scaled_metric = metrics.Mean(name='g_fake_gbsa_scaled')
        self.g_fake_gbsa_mean_denormalized_metric = metrics.Mean(name='g_fake_gbsa_denormalized')

        # Diversity metrics
        self.real_diversity_metric = metrics.Mean(name='real_diversity')
        self.fake_diversity_metric = metrics.Mean(name='fake_diversity')
        self.fake_entropy_metric = metrics.Mean(name='fake_entropy')
        self.fake_unique_ratio_metric = metrics.Mean(name='fake_unique_ratio')
        self.fake_evenness_metric = metrics.Mean(name='fake_evenness')

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_gbsa_loss_metric,
            self.d_diversity_loss_metric,
            self.d_gbsa_mae_metric,
            self.g_loss_metric,
            self.g_gbsa_loss_metric,
            self.g_diversity_loss_metric,
            self.g_property_loss_metric,
            self.g_fake_gbsa_mean_scaled_metric,
            self.g_fake_gbsa_mean_denormalized_metric,
            self.real_diversity_metric,
            self.fake_diversity_metric,
            self.fake_entropy_metric,
            self.fake_unique_ratio_metric,
            self.fake_evenness_metric,
        ]

    def train_step(self, data):
        (real_ab, real_ag), real_gbsa = data

        # === Discriminator step ===
        with tf.GradientTape() as d_tape:
            predicted_gbsa_real, diversity_score = self.discriminator([real_ab, real_ag], training=True)
            d_gbsa_loss = self.loss_fn(real_gbsa, predicted_gbsa_real)

        d_grads = d_tape.gradient(d_gbsa_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # === Generator step ===
        with tf.GradientTape() as g_tape:
            fake_ab, fake_diversity = self.generator([real_ab, real_ag], training=True)
            predicted_gbsa_fake, _ = self.discriminator([fake_ab, real_ag], training=False)  # Fixed: fake*ab -> fake_ab

            # GBSA loss
            g_gbsa_loss = -tf.reduce_mean(predicted_gbsa_fake)

            # Diversity loss (the higher the better, so we minimize negative diversity)
            diversity_loss = -tf.reduce_mean(fake_diversity)  # Fixed: changed to negative for proper minimization

            # Inter-batch similarity (cosine)
            fake_ab_last22 = fake_ab[..., 8:]  # (B, R, 5, 22)
            batch_size = tf.shape(fake_ab)[0]
            flat = tf.reshape(fake_ab_last22, [batch_size, -1])  # (B, R*5*22)
            normed = tf.nn.l2_normalize(flat, axis=1)
            sim_matrix = tf.matmul(normed, normed, transpose_b=True)
            mask = tf.linalg.band_part(tf.ones_like(sim_matrix), 0, -1) - tf.eye(batch_size)
            sim_values = tf.boolean_mask(sim_matrix, tf.cast(mask, tf.bool))
            inter_batch_diversity_loss = tf.reduce_mean(sim_values)

            # Total generator loss
            g_loss = (
                    g_gbsa_loss
                    + self.diversity_weight * diversity_loss
                    + self.inter_batch_diversity_weight * inter_batch_diversity_loss
            )

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Optional: compute inter-batch diversity metric (1 - similarity)
        inter_batch_diversity = 1.0 - tf.reduce_mean(sim_values)

        return {
            "d_loss": d_gbsa_loss,
            "g_loss": g_loss,
            "g_gbsa_loss": g_gbsa_loss,
            "diversity_loss": diversity_loss,
            "inter_batch_diversity_loss": inter_batch_diversity_loss,
            "inter_batch_diversity": inter_batch_diversity
        }
