import numpy as np
import joblib
import tensorflow as tf
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.optimizers import Adam
from Model.Generator import ConditionalGenerator
from Model.Models import Net as ConditionalDiscriminator
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

eleTypes = ['N.3', 'N.am', 'N.4', 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


class ConditionalGAN:
    def __init__(self, noise_dim=100, generator_kwargs=None):
        self.noise_dim = noise_dim
        self.gen = ConditionalGenerator(noise_dim=noise_dim, **generator_kwargs)
        self.disc = ConditionalDiscriminator()
        self.cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mse = MeanSquaredError()
        self.gen_optimizer = Adam(3e-4)
        self.disc_optimizer = Adam(5e-5)

    def generator_loss(self, valid_pred, gbsa_pred, target_label):
        adv_loss = self.cross_entropy(tf.ones_like(valid_pred) * 0.9, valid_pred)
        reg_loss = self.mse(target_label, gbsa_pred)
        return adv_loss + 0.05 * reg_loss  # Further reduce MSE influence

    def discriminator_loss(self, valid_real, valid_fake, gbsa_pred_real, real_labels):
        # Label smoothing and flipping parameters
        real_label_smooth_range = (0.9, 1.0)
        fake_label_smooth_range = (0.0, 0.2)
        label_flip_prob = 0.05

        batch_size = tf.shape(valid_real)[0]

        flip_mask = tf.random.uniform(tf.shape(valid_real)) < label_flip_prob
        real_targets = tf.where(
            flip_mask,
            tf.zeros_like(valid_real),
            tf.random.uniform(tf.shape(valid_real), *real_label_smooth_range)
        )

        # Fake labels: 0-20% of 1.0
        fake_targets = tf.random.uniform(
            tf.shape(valid_fake),
            *fake_label_smooth_range
        )

        # Loss calculations
        real_loss = self.cross_entropy(real_targets, valid_real)
        fake_loss = self.cross_entropy(fake_targets, valid_fake)
        reg_loss = self.mse(real_labels, gbsa_pred_real)

        # Combine losses
        return tf.reduce_mean(real_loss + fake_loss) + reg_loss

    # @tf.function
    def train_step(self, ab_real, ag_real, gbsa_real):
        batch_size = tf.shape(ab_real)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        target_labels = gbsa_real + tf.random.normal(tf.shape(gbsa_real), stddev=0.1)
        # target_labels = gbsa_real  # Now using actual GBSA values for conditioning

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            ag_fake = self.gen(noise, target_labels)

            real_input = {
                'ab_input': ab_real,
                'ag_input': ag_real,
                'gbsa_input': gbsa_real
            }

            fake_input = {
                'ab_input': ab_real,
                'ag_input': ag_fake,
                'gbsa_input': target_labels
            }

            valid_real, gbsa_pred_real = self.disc(real_input)
            valid_fake, gbsa_pred_fake = self.disc(fake_input)

            gen_loss = self.generator_loss(valid_fake, gbsa_pred_fake, target_labels)
            disc_loss = self.discriminator_loss(valid_real, valid_fake, gbsa_pred_real, gbsa_real)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))

        return gen_loss, disc_loss

    def train(self, ab, ag, gbsa, epochs=100, batch_size=32, val_split=0.1):
        # === Train/val split ===
        ab_train, ab_val, ag_train, ag_val, gbsa_train, gbsa_val = train_test_split(ab, ag, gbsa, test_size=val_split,
                                                                                    random_state=42)

        # === Build datasets ===
        train_dataset = tf.data.Dataset.from_tensor_slices((ab_train, ag_train, gbsa_train))
        train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((ab_val, ag_val, gbsa_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            g_loss_total, d_loss_total = [], []

            # === Training loop ===
            for step, (ab_batch, ag_batch, gbsa_batch) in enumerate(train_dataset):
                g_loss, d_loss = self.train_step(ab_batch, ag_batch, gbsa_batch)
                g_loss_total.append(g_loss.numpy())
                d_loss_total.append(d_loss.numpy())

                if step % 10 == 0:
                    print(f"  Step {step} | Gen Loss: {g_loss.numpy():.4f} | Disc Loss: {d_loss.numpy():.4f}")

            # === Validation loop (no training) ===
            val_g_loss_total, val_d_loss_total = [], []

            for ab_val_batch, ag_val_batch, gbsa_val_batch in val_dataset:
                batch_size_val = tf.shape(ab_val_batch)[0]
                noise = tf.random.normal([batch_size_val, self.noise_dim])
                target = tf.fill([batch_size_val, 1], -abs(tf.reduce_min(gbsa_val_batch)))

                ag_fake = self.gen(noise, target, training=False)

                fake_input = {
                    'ab_input': ab_val_batch,
                    'ag_input': ag_fake,
                    'gbsa_input': target
                }
                real_input = {
                    'ab_input': ab_val_batch,
                    'ag_input': ag_val_batch,
                    'gbsa_input': gbsa_val_batch
                }

                valid_real, gbsa_pred_real = self.disc(real_input, training=False)
                valid_fake, gbsa_pred_fake = self.disc(fake_input, training=False)

                val_g_loss = self.generator_loss(valid_fake, gbsa_pred_fake, target)
                val_d_loss = self.discriminator_loss(valid_real, valid_fake, gbsa_pred_real, gbsa_val_batch)

                val_g_loss_total.append(val_g_loss.numpy())
                val_d_loss_total.append(val_d_loss.numpy())

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Gen Loss: {np.mean(g_loss_total):.4f} | Train Disc Loss: {np.mean(d_loss_total):.4f}")
            print(
                f"  Val   Gen Loss: {np.mean(val_g_loss_total):.4f} | Val   Disc Loss: {np.mean(val_d_loss_total):.4f}")


def TrainAndGenerate(args):
    # === Load data ===
    data = np.load('matrices/padded_dataset.npz', allow_pickle=True)
    ab = data['ab']
    ag = data['ag']
    gbsa = data['gbsa'].reshape(-1, 1)
    print("Loaded")

    # === Load scalers ===
    feature_scaler = joblib.load('feature_scaler.pkl')
    label_scaler = joblib.load('label_scaler.pkl')
    print("Scalers loaded")
    # === Scale only x,y,z (first 3 channels) ===
    continuous_idx = slice(0, 3)

    ab_cont = ab[..., continuous_idx].reshape(-1, 3)
    ag_cont = ag[..., continuous_idx].reshape(-1, 3)

    ab[..., continuous_idx] = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
    ag[..., continuous_idx] = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)
    gbsa_scaled = label_scaler.transform(gbsa)
    print("Scaled stuff")
    # === Initialize and train GAN ===
    gan = ConditionalGAN(
        noise_dim=512,
        generator_kwargs={
            'eleTypes': eleTypes,
            'amino_acids': amino_acids,
        }
    )

    print("Gan Built")
    gan.train(ab, ag, gbsa_scaled, epochs=args['epoch'], batch_size=args['batch'])
    print("Finished GAN training")
    noise = tf.random.normal([10, gan.noise_dim])
    target = tf.fill([10, 1], -1.5)  # Provide a meaningful scaled GBSA value, e.g., -1.5
    generated_ag = gan.gen(noise, target).numpy()

    gen_ag_cont = generated_ag[..., :3].reshape(-1, 3)
    gen_ag_cont_rescaled = feature_scaler.inverse_transform(gen_ag_cont)
    generated_ag[..., :3] = gen_ag_cont_rescaled.reshape(generated_ag.shape[0], generated_ag.shape[1],
                                                         generated_ag.shape[2], 3)

    np.save('generated_ag.npy', generated_ag)
