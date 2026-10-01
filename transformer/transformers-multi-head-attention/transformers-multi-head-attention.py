import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def attention(Q, K, d_k, V):
    first_part = softmax((Q@K.transpose(0, 2, 1))/np.sqrt(d_k))
    return first_part @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    d_k = Q.shape[-1] // num_heads
    first = Q @ W_q
    second = K @ W_k 
    third = V @ W_v
    thirds = np.array_split(third, num_heads, -1)
    seconds = np.array_split(second, num_heads, -1)
    firsts = np.array_split(first, num_heads, -1)
    # W_qs = np.array_split(W_q, num_heads, 0)
    # W_ks = np.array_split(W_k, num_heads, 0)
    # W_vs = np.array_split(W_v, num_heads, 0)
    # return Q.shape, W_q.shape
    head = []
    for i in range(num_heads):
        head.append(attention(firsts[i], seconds[i],d_k,  thirds[i]))
    answer = np.concatenate(head, -1 )
    return answer @ W_o