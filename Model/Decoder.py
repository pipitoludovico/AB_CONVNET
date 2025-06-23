import numpy as np
from tensorflow.keras.models import load_model

accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']
resnames = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

atom_names_one_hot = np.eye(len(accepted_atoms))
resnames_one_hot = np.eye(len(resnames))

max_ab_len = 92
max_ag_len = 97

antigen_residues = 34
antibody_residues = 16
number_of_atoms = 5


def GenerateSample(number_of_residues: int, what: str):
    if what == "ag":
        sample = np.zeros(shape=(max_ag_len, 5, 30))
    else:
        sample = np.zeros(shape=(max_ab_len, 5, 30))
    for x in range(0, number_of_residues):
        amminoacido_scelto = np.random.choice(len(resnames))
        for atomo in range(len(accepted_atoms)):
            if resnames[amminoacido_scelto] == "GLY" and atomo == 2:
                continue
            coordinates = np.random.uniform(size=(3,), low=-100, high=180).round(2)
            zio = np.concatenate((coordinates, atom_names_one_hot[atomo], resnames_one_hot[amminoacido_scelto]), axis=0)
            sample[x][atomo] = zio
    return sample


ag = GenerateSample(48, 'ag')
ab = GenerateSample(42, 'ab')

ab = np.expand_dims(ab, axis=0)  # batch dimension
ab = np.expand_dims(ab, axis=-2)  # add channel dimension before features

ag = np.expand_dims(ag, axis=0)
ag = np.expand_dims(ag, axis=-2)

z = np.random.normal(size=(1, 128))


# from Model.Models import Generator, GeneticGenerator

# generator = Generator(128, ag.shape, ab.shape)
# ab_generated = generator.predict([ag, z])
#
# print(ab_generated.shape)
# print(ab_generated[0])


def GeneticMutate(ab: np.ndarray, resnames_onehot):
    # ab shape: (1, 92, 5, 1, 30)
    ab_clean = ab[0]  # (92, 5, 1, 30)

    mask = np.any(ab_clean != 0, axis=(1, 3)).squeeze()  # shape (92,)

    ab_stripped = ab_clean[mask]  # shape (num_valid_residues, 5, 1, 30)
    if ab_stripped.shape[0] == 0:
        raise ValueError("Nessun residuo valido trovato per mutazione.")

    idx_residuo = np.random.randint(ab_stripped.shape[0])

    num_resnames = resnames_onehot.shape[0]
    eye = np.eye(num_resnames)

    old_onehot = ab_stripped[idx_residuo][0, 0, -22:]
    old_idx = np.argmax(old_onehot)

    possibili = list(set(range(num_resnames)) - {old_idx})
    nuovo_idx = np.random.choice(possibili)
    nuovo_onehot = eye[nuovo_idx]

    for atom in range(5):
        ab_stripped[idx_residuo][atom, 0, -22:] = nuovo_onehot

    print(f"shape dello strippato: {ab_stripped.shape}")
    return ab_stripped


def pad_ab(ab_stripped, max_len=92):
    padded = np.zeros((1, max_len, 5, 1, 30), dtype=ab_stripped.dtype)
    n = ab_stripped.shape[0]
    padded[0, :n] = ab_stripped
    return padded


def MutagenesiGuidata(discriminatore, ab, ag, resnames_onehot):
    ab_current = ab.copy()
    initial_mask = np.any(ab_current != 0, axis=(2, 3, 4))[0]
    initial_valid_residues = np.sum(initial_mask)
    print(f"Residui validi: {initial_valid_residues}")

    gbsa_attuale = discriminatore.predict([ab_current, ag])[0]
    gbsa_attuale_scalar = gbsa_attuale.item()
    print(f"GBSA INIZIALE: {gbsa_attuale_scalar:.4f}")

    for i in range(initial_valid_residues):
        ab_clean = ab_current[0]
        mask = np.any(ab_clean != 0, axis=(1, 3)).squeeze()
        ab_stripped = ab_clean[mask]

        ab_mutato_stripped = GeneticMutate(np.expand_dims(ab_stripped, axis=0), resnames_onehot)  # (num_valid,5,1,30)

        # Riaggiungo padding e batch
        ab_mutato = pad_ab(ab_mutato_stripped, max_len=ab_current.shape[1])

        gbsa_mutato = discriminatore.predict([ab_mutato, ag])[0]
        gbsa_mutato_scalar = gbsa_mutato.item()
        print(f"[Mutazione {i + 1}] GBSA mutato: {gbsa_mutato_scalar:.4f}")

        if gbsa_mutato_scalar < gbsa_attuale_scalar:
            print(f"Mutazione accettata. Miglioramento: {gbsa_attuale_scalar:.4f} → {gbsa_mutato_scalar:.4f}")
            ab_current = ab_mutato
            gbsa_attuale_scalar = gbsa_mutato_scalar
        else:
            print("Mutazione scartata.")

    print("CURRENT SHAPE", ab_current.shape)
    return ab_current, gbsa_attuale_scalar


discriminator = load_model("./best_model.keras", compile=False, safe_mode=False)
discriminator.trainable = False

ab_ottimizzato, gbsa_finale = MutagenesiGuidata(
    discriminatore=discriminator,
    ab=ab,
    ag=ag,
    resnames_onehot=resnames_one_hot,
)
