from numba import cuda
import math
import numpy as np


@cuda.jit("void(int64, int64, float64[:,3], float64[:,3], complex128[:])")
def _get_efield_phase(atom_num, q_num, atom_positions, wavevectors, e_field_phase):
    """
    Notice that this function only calculate the phase part of the diffracted field.
    Also, as can be derived from the input argument, this function relies on
    far field approximation.

    :param atom_num:
    :param q_num:
    :param atom_positions:
    :param wavevectors:
    :param e_field_phase:
    :return:
    """
    a_idx = cuda.grid(1)  # Index of the atom
    if a_idx < atom_num:
        for q_idx in range(q_num):
            phase = (atom_positions[a_idx, 0] * wavevectors[q_idx, 0] +
                     atom_positions[a_idx, 1] * wavevectors[q_idx, 1] +
                     atom_positions[a_idx, 2] * wavevectors[q_idx, 2])

            e_field_phase[q_idx] += complex(math.cos(phase), math.sin(phase))


def get_electric_field_phase(q_vectors, sample, t_num=512):
    """
    This is the wrapper function of the gpu calculator
    _get_efield_phase

    :param q_vectors: The wave vectors for each pixel to calculate. The shape is (n,3).
    :param sample: Numpy array of shape (n,3). The last dimension is the x,y,z coordiante of the sample.
    :param t_num: This is a parameter that can be tuned to optimize the GPU calculation.
                    If one does not know how to optimize this, one can just choose one that works.
                    This only influence the calculation efficiency not the result itself.
    :return:
    """
    # Because it seems that the sample data can fit into the GPU altogether
    # Therefore, I decide not to split the data.
    # Instead, just transfer everything into the gpu and do the calculation

    a_num = sample.shape[0]
    q_num = q_vectors.shape[0]
    phase_holder = np.zeros(q_num, dtype=np.complex128)

    # Set some gpu parameters
    t_num = t_num  # threadsperblock
    b_num = (a_num + (t_num - 1)) // t_num  # blockspergrid

    # Get the electric field phase
    _get_efield_phase[b_num, t_num](a_num, q_num, sample, q_vectors, phase_holder)

    return phase_holder
