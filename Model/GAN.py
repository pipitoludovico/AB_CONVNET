import numpy as np
import joblib
import tensorflow as tf
from Model.Generator import CNNConditionalGenerator
from Model.Models import Net as ConditionalDiscriminator

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

eleTypes = ['N.3', 'N.am', 'N.4', 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


class ConditionalGAN:
    def __init__(self, max_residues=100):
        # Discriminator da te
        self.discriminator = ConditionalDiscriminator(ab_shape=(max_residues, 5, 34), ag_shape=(None, 5, 34))

        # Generator
        self.generator = CNNConditionalGenerator(max_residues=max_residues)

        # Loss e ottimizzatori
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.gen_opt = tf.keras.optimizers.Adam(3e-4)
        self.disc_opt = tf.keras.optimizers.Adam(5e-5)

    def count_residues(self, ab_batch):
        # Considera residui validi se somma ass > 0 (ignora padding)
        residue_sum = tf.reduce_sum(tf.abs(ab_batch), axis=[2, 3])
        mask = tf.cast(residue_sum > 1e-5, tf.int32)
        return tf.reduce_sum(mask, axis=1)

    @tf.function
    def train_step(self, ag_batch, ab_batch, gbsa_batch):
        K = self.count_residues(ab_batch)

        with tf.GradientTape(persistent=True) as tape:
            fake_ab, _ = self.generator([ag_batch, K])

            valid_real, gbsa_pred_real = self.discriminator([ab_batch, ag_batch, tf.expand_dims(gbsa_batch, -1)])
            valid_fake, gbsa_pred_fake = self.discriminator([fake_ab, ag_batch, tf.expand_dims(gbsa_batch, -1)])

            # Loss discriminator
            disc_loss_real = self.bce(tf.ones_like(valid_real), valid_real)
            disc_loss_fake = self.bce(tf.zeros_like(valid_fake), valid_fake)
            disc_loss_gbsa = self.mse(gbsa_batch, tf.squeeze(gbsa_pred_real))
            disc_loss = disc_loss_real + disc_loss_fake + disc_loss_gbsa

            # Loss generator
            gen_adv_loss = self.bce(tf.ones_like(valid_fake), valid_fake)
            gen_gbsa_loss = self.mse(gbsa_batch, tf.squeeze(gbsa_pred_fake))
            gen_loss = gen_adv_loss + gen_gbsa_loss

        # Grad e update pesi
        grads_gen = tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        self.disc_opt.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))
        del tape

        return gen_loss, disc_loss


def TrainAndGenerate(args):
    data = np.load('matrices/padded_dataset.npz', allow_pickle=True)
    ab, ag, gbsa = data['ab'], data['ag'], data['gbsa'].reshape(-1, 1)

    f_scaler = joblib.load('feature_scaler.pkl')
    l_scaler = joblib.load('label_scaler.pkl')

    # Normalizza coordinate xyz (prime 3 feature)
    ab[..., :3] = f_scaler.transform(ab[..., :3].reshape(-1, 3)).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
    ag[..., :3] = f_scaler.transform(ag[..., :3].reshape(-1, 3)).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

    gbsa_scaled = l_scaler.transform(gbsa)

    gan = ConditionalGAN(max_residues=ab.shape[1])  # max_residues = max residui antibody

    # Training loop
    dataset = tf.data.Dataset.from_tensor_slices((ag, ab, gbsa_scaled))
    dataset = dataset.shuffle(1024).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

    for epoch in range(args['epoch']):
        print(f"Epoch {epoch + 1}/{args['epoch']}")
        for step, (ag_batch, ab_batch, gbsa_batch) in enumerate(dataset):
            gen_loss, disc_loss = gan.train_step(ag_batch, ab_batch, gbsa_batch)
            if step % 10 == 0:
                print(f"  Step {step}: Gen Loss={gen_loss:.4f}, Disc Loss={disc_loss:.4f}")

    # Generazione: usa primo esempio, conta residui reali da ab
    K = gan.count_residues(ab[0:1])  # batch dimension 1

    generated_ab, _ = gan.generator([ag[0:1], K])

    generated_ab = generated_ab.numpy()
    # Denormalizza le coordinate xyz
    generated_ab[..., :3] = f_scaler.inverse_transform(
        generated_ab[..., :3].reshape(-1, 3)
    ).reshape(generated_ab.shape[0], generated_ab.shape[1], generated_ab.shape[2], 3)

    np.save('generated_ab.npy', generated_ab)
