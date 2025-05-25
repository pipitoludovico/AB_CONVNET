import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']
eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0,1) """
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature=0.5):
    """
    ST Gumbel-Softmax:
    logits: [batch_size, ..., n_class]
    Returns differentiable approximation of one-hot vectors.
    """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
    # Straight-through estimator
    y = tf.stop_gradient(y_hard - y) + y
    return y

def Generator(max_ab_len=92, max_ag_len=97, temperature=0.5):
    features_coords = 4
    features_ele = len(eleTypes)  # 8
    features_res = len(amino_acids)  # 22

    ag_input = layers.Input(shape=(max_ag_len, 5, features_coords + features_ele + features_res), name='ag_input')

    x = layers.Reshape((max_ag_len * 5, features_coords + features_ele + features_res))(ag_input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    coords = layers.Dense(max_ab_len * 5 * features_coords, activation='linear')(x)
    coords = layers.Reshape((max_ab_len, 5, features_coords), name='coords')(coords)

    ele_logits = layers.Dense(max_ab_len * 5 * features_ele)(x)
    ele_logits = layers.Reshape((max_ab_len, 5, features_ele))(ele_logits)

    res_logits = layers.Dense(max_ab_len * 5 * features_res)(x)
    res_logits = layers.Reshape((max_ab_len, 5, features_res))(res_logits)

    # Usa Lambda per applicare Gumbel-Softmax che è una operazione personalizzata
    ele = layers.Lambda(lambda l: gumbel_softmax(l, temperature), name='ele_gumbel_softmax')(ele_logits)
    res = layers.Lambda(lambda l: gumbel_softmax(l, temperature), name='res_gumbel_softmax')(res_logits)

    output = layers.Concatenate(axis=-1)([coords, ele, res])

    return Model(inputs=ag_input, outputs=output, name='Generator')


# Test
max_ab_len = 92
max_ag_len = 97
model = Generator(max_ab_len=max_ab_len, max_ag_len=max_ag_len, temperature=0.5)

batch_size = 1
input_shape = (max_ag_len, 5, 4 + len(eleTypes) + len(amino_acids))
dummy_input = tf.random.normal((batch_size,) + input_shape)
output = model(dummy_input)

print("Output shape:", output.shape)
print("Example coords output (first residue, first atom):", output[0, 0, 0, :4].numpy())
print("Sum of ele_gumbel_softmax (should be 1):", tf.reduce_sum(output[0, 0, 0, 4:12]).numpy())
print("Sum of res_gumbel_softmax (should be 1):", tf.reduce_sum(output[0, 0, 0, 12:]).numpy())
print("Ele Gumbel sample argmax:", tf.argmax(output[0, 0, 0, 4:12]).numpy())
print("Res Gumbel sample argmax:", tf.argmax(output[0, 0, 0, 12:]).numpy())
