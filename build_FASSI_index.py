## Build FASSI Index
def build_FASSI_index(embeddings):
    """
    Builds a FAISS index using L2 (Euclidean) distance for the provided embeddings.
    Args:
        embeddings (np.ndarray): A 2D NumPy array of shape (n_samples, n_features)
            containing the vector representations to be indexed.
    Returns:
        faiss.IndexFlatL2: A FAISS index object with the embeddings added.
    Raises:
        AttributeError: If `embeddings` does not have a `shape` attribute.
        ValueError: If `embeddings` is not a 2D array.
    Examples:
        >>> import numpy as np
        >>> import faiss
        >>> embeddings = np.random.rand(100, 128).astype('float32')
        >>> index = build_FASSI_index(embeddings)
        >>> index.ntotal
        100
    """
    import faiss
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return(index)