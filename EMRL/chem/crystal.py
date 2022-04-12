import numpy
import pandas
import random
import torch
import chem.base as cb
from tqdm import tqdm
from pymatgen.core.structure import Structure
from chem.base import get_elem_feats
from chem.data import TuplewiseCrystal


def load_dataset(path_cif, id_target_file, target_idx, radius=4, shuffle=False, ext='cif'):
    list_crys = list()
    id_target = numpy.array(pandas.read_excel(id_target_file))
    elem_feats = get_elem_feats()

    for i in tqdm(range(0, id_target.shape[0])):
        crys = read_cif(elem_feats, path_cif, str(id_target[i, 0]), id_target[i, target_idx], radius, idx=i, ext=ext)

        if crys is not None:
            list_crys.append(crys)

    if shuffle:
        random.shuffle(list_crys)

    return list_crys


def read_cif(elem_feats, path, m_id, target, radius, idx, ext):
    crys = Structure.from_file(path + '/' + m_id + '.' + ext)
    atoms = crys.atomic_numbers
    list_nbrs = crys.get_all_neighbors(radius, include_index=True)
    rbf_means = cb.even_samples(3, radius, cb.n_bond_feats)
    pairs = get_pairs(elem_feats, atoms, list_nbrs, rbf_means)

    if pairs is None:
        return None

    pairs = torch.tensor(pairs, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.float).view(-1, 1)

    return TuplewiseCrystal(pairs, target, idx)


def get_pairs(elem_feats, atoms, list_nbrs, means):
    pairs = list()

    for i in range(0, len(list_nbrs)):
        atom_feats1 = elem_feats[atoms[i] - 1, :]
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            atom_feats2 = elem_feats[atoms[nbrs[j][2]] - 1, :]
            bond_feats = cb.RBF(numpy.full(means.shape[0], nbrs[j][1]), means, beta=0.2)
            pairs.append(numpy.hstack([atom_feats1, atom_feats2, bond_feats]))

    if len(pairs) == 0:
        return None

    return numpy.vstack(pairs)
