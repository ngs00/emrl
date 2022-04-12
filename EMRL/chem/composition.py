import numpy
import pandas
import random
import ast
import chem.base as cb


def load_dataset(id_target_file, comp_idx, target_idx, shuffle=False):
    elem_feats = cb.get_elem_feats()
    id_target = numpy.array(pandas.read_excel(id_target_file))
    mat_feats = list()

    for i in range(0, id_target.shape[0]):
        elems = ast.literal_eval(id_target[i, comp_idx])
        e_sum = numpy.sum([float(elems[key]) for key in elems])
        w_sum_vec = numpy.zeros(elem_feats.shape[1])
        atom_feats = list()

        for e in elems:
            atom_vec = elem_feats[cb.atom_nums[e] - 1, :]
            atom_feats.append(atom_vec)
            w_sum_vec += (float(elems[e]) / e_sum) * atom_vec

        mat_feat = numpy.hstack([w_sum_vec, numpy.std(atom_feats, axis=0), numpy.min(atom_feats, axis=0),
                                 numpy.max(atom_feats, axis=0), id_target[i, target_idx]])
        mat_feats.append(mat_feat)

    if shuffle:
        random.shuffle(mat_feats)

    return numpy.vstack(mat_feats)
