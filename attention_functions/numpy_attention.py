import numpy as np

def attention(Q, K, V):
    """
    Calcule le mécanisme d'attention scalaire.
    
    Args:
        Q: np.ndarray de forme (batch_size, seq_len_q, d_k)
        K: np.ndarray de forme (batch_size, seq_len_k, d_k)
        V: np.ndarray de forme (batch_size, seq_len_v, d_v)
        
    Returns:
        Attention(Q, K, V): np.ndarray de forme (batch_size, seq_len_q, d_v)
    """
    dk = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(dk)

    # Softmax with numerical stability
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    scores /= np.sum(scores, axis=-1, keepdims=True)

    return np.matmul(scores, V)
