from keras import layers, Model, Input
import tensorflow as tf

# Constants
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
               "HIS", "HIE", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL"]
ELE_TYPES = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
MAX_RESIDUES = 92
NUM_ATOMS = 5
XYZQ_DIM = 4  # x, y, z, charge
ATOM_TYPE_DIM = 8
RESNAME_DIM = 22
OUTPUT_DIM = XYZQ_DIM + ATOM_TYPE_DIM + RESNAME_DIM


def Discriminator(ab_shape, ag_shape):
    ab_input = Input(shape=ab_shape, name="ab_input")
    ag_input = Input(shape=ag_shape, name="ag_input")

    def encode_entity(entity):
        entity = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(64, activation='relu')))(entity)
        entity = layers.Reshape((entity.shape[1], -1))(entity)
        entity = layers.Bidirectional(layers.GRU(128, return_sequences=True))(entity)
        entity = layers.GlobalAveragePooling1D()(entity)
        return entity

    x_ab = encode_entity(ab_input)
    x_ag = encode_entity(ag_input)
    x = layers.Concatenate()([x_ab, x_ag])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    gbsa_out = layers.Dense(1, name="gbsa_prediction")(x)
    validity = layers.Dense(1, activation="sigmoid", name="validity")(x)

    return Model(inputs=[ab_input, ag_input], outputs=[gbsa_out, validity])


def Generator(ag_shape):
    ag_input = Input(shape=ag_shape, name="ag_input")

    # Encode antigen (e.g., using GRU or Transformer)
    x = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(64, activation='relu')))(ag_input)
    x = layers.Reshape((MAX_RESIDUES, -1))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Broadcast latent vector to residues
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(MAX_RESIDUES * 128, activation='relu')(x)
    x = layers.Reshape((MAX_RESIDUES, 128))(x)

    # Residue identity prediction (softmax)
    resname_logits = layers.Dense(RESNAME_DIM, activation='softmax', name="resname_logits")(x)

    # Generate atomic coordinates and charge: (MAX_RESIDUES, NUM_ATOMS, 4)
    x_coords = layers.Dense(NUM_ATOMS * XYZQ_DIM)(x)
    x_coords = layers.Reshape((MAX_RESIDUES, NUM_ATOMS, XYZQ_DIM), name="xyzq")(x_coords)

    # Create full feature vector placeholder, concatenate later in logic
    generator_model = Model(inputs=ag_input, outputs=[resname_logits, x_coords], name="Generator")
    return generator_model


def postprocess_generator_output(resname_logits, xyzq, atom_type_encoding, resname_encoding):
    """
    resname_logits: Tensor of shape (batch, MAX_RESIDUES, 22)
    xyzq: Tensor of shape (batch, MAX_RESIDUES, 5, 4)
    atom_type_encoding: one-hot vectors for ['N', 'CA', 'CB', 'C', 'O']
    resname_encoding: one-hot vectors for AMINO_ACIDS (22)
    """
    resname_idx = tf.argmax(resname_logits, axis=-1)  # shape: (batch, MAX_RESIDUES)

    # Create one-hot encodings for residue names
    resname_onehot = tf.one_hot(resname_idx, depth=RESNAME_DIM)  # (batch, MAX_RESIDUES, 22)

    # Tile the one-hot encoding across 5 atoms
    resname_onehot_tiled = tf.tile(resname_onehot[:, :, tf.newaxis, :], [1, 1, NUM_ATOMS, 1])

    # Broadcast fixed atom type one-hots
    atom_type_tensor = tf.constant(atom_type_encoding, dtype=tf.float32)  # (5, 8)
    atom_type_tensor = tf.reshape(atom_type_tensor, (1, 1, NUM_ATOMS, ATOM_TYPE_DIM))
    atom_type_broadcast = tf.tile(atom_type_tensor, [tf.shape(xyzq)[0], MAX_RESIDUES, 1, 1])

    # Combine all features
    features = tf.concat([xyzq, atom_type_broadcast, resname_onehot_tiled], axis=-1)  # (batch, MAX_RESIDUES, 5, 34)

    # Apply GLY mask: zero out last row if resname == 8
    gly_mask = tf.equal(resname_idx, 8)  # (batch, MAX_RESIDUES)
    gly_mask = tf.cast(gly_mask, tf.float32)

    # Expand to mask last atom only
    zero_patch = tf.zeros((tf.shape(features)[0], MAX_RESIDUES, 1, OUTPUT_DIM), dtype=tf.float32)

    # Remove the last row and append zero row if GLY
    features_main = features[:, :, :-1, :]
    features = tf.concat([features_main,
                          features[:, :, -1:, :] * (1.0 - gly_mask[:, :, tf.newaxis, tf.newaxis])], axis=2)

    return features  # (batch, MAX_RESIDUES, 5, 34)


class GANTrainer(Model):
    def __init__(self, generator, discriminator, atom_type_encoding, resname_encoding):
        super().__init__()
        self.valid_loss_fn = None
        self.gbsa_loss_fn = None
        self.disc_optimizer = None
        self.gen_optimizer = None
        self.generator = generator
        self.discriminator = discriminator
        self.atom_type_encoding = atom_type_encoding
        self.resname_encoding = resname_encoding

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    def compile(self, gen_optimizer, disc_optimizer, gbsa_loss_fn, valid_loss_fn):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gbsa_loss_fn = gbsa_loss_fn
        self.valid_loss_fn = valid_loss_fn

    def train_step(self, data):
        real_inputs, real_labels = data  # dict of inputs, dict of outputs
        real_ab = real_inputs['ab_input']
        real_ag = real_inputs['ag_input']
        real_gbsa = real_labels['gbsa_prediction']

        batch_size = tf.shape(real_ag)[0]

        # === GENERATOR FORWARD PASS ===
        with tf.GradientTape() as gen_tape:
            resname_logits, xyzq = self.generator(real_ag, training=True)
            fake_ab = postprocess_generator_output(resname_logits, xyzq, self.atom_type_encoding, self.resname_encoding)

            pred_gbsa, pred_validity = self.discriminator([fake_ab, real_ag], training=False)

            # Mask for "realistic" predictions (validity > 0.9)
            valid_mask = tf.cast(pred_validity > 0.9, tf.float32)
            gbsa_loss = self.gbsa_loss_fn(real_gbsa, pred_gbsa)  # per-sample loss
            masked_gbsa_loss = tf.reduce_mean(valid_mask * gbsa_loss)

            # Fool discriminator (label as valid)
            valid_labels = tf.ones_like(pred_validity)
            gen_valid_loss = self.valid_loss_fn(valid_labels, pred_validity)

            total_gen_loss = masked_gbsa_loss + gen_valid_loss

        gen_grads = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        # === DISCRIMINATOR FORWARD PASS ===
        with tf.GradientTape() as disc_tape:
            # Evaluate real
            pred_gbsa_real, pred_valid_real = self.discriminator([real_ab, real_ag], training=True)
            gbsa_loss_real = tf.reduce_mean(self.gbsa_loss_fn(real_gbsa, pred_gbsa_real))
            valid_loss_real = self.valid_loss_fn(tf.ones_like(pred_valid_real), pred_valid_real)

            # Evaluate fake
            _, pred_valid_fake = self.discriminator([fake_ab, real_ag], training=True)
            valid_loss_fake = self.valid_loss_fn(tf.zeros_like(pred_valid_fake), pred_valid_fake)

            total_disc_loss = gbsa_loss_real + valid_loss_real + valid_loss_fake

        disc_grads = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        self.gen_loss_tracker.update_state(total_gen_loss)
        self.disc_loss_tracker.update_state(total_disc_loss)

        return {
            "generator_loss": self.gen_loss_tracker.result(),
            "discriminator_loss": self.disc_loss_tracker.result(),
        }

    def test_step(self, data):
        # Optional: add for evaluation
        return self.train_step(data)

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
