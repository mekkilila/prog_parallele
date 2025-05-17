# cython_attention.pyx
# cython: language_level=3
# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt

ctypedef fused floating:
    np.float32_t
    np.float64_t

# Mécanisme d'attention scalaire optimisé pour float32/float64

def attention(
    np.ndarray[floating, ndim=2] Q,
    np.ndarray[floating, ndim=2] K,
    np.ndarray[floating, ndim=2] V,
    int n_threads=1,
    int block_size=32
):
    cdef int n_q = Q.shape[0]
    cdef int n_k = K.shape[0]
    cdef int d = Q.shape[1]

    cdef floating[:, :] matQ = Q
    cdef floating[:, :] matK = K
    cdef floating[:, :] matV = V

    # Scores = Q @ K.T / sqrt(d)
    cdef np.ndarray[floating, ndim=2] scores_arr = np.dot(matQ, matK.T) / sqrt(d)
    cdef floating[:, :] scores = scores_arr

    # Allocation des poids softmax
    cdef np.ndarray[floating, ndim=2] weights_arr = np.empty((n_q, n_k), dtype=scores_arr.dtype)
    cdef floating[:, :] weights = weights_arr

    cdef int i, j
    cdef floating row_max, sum_exp, val

    # Softmax ligne par ligne
    for i in range(n_q):
        row_max = scores[i, 0]
        for j in range(1, n_k):
            if scores[i, j] > row_max:
                row_max = scores[i, j]
        sum_exp = 0
        for j in range(n_k):
            val = exp(scores[i, j] - row_max)
            weights[i, j] = val
            sum_exp += val
        for j in range(n_k):
            weights[i, j] /= sum_exp

    # Produit final weights @ V
    return np.dot(weights_arr, matV)
