"""
Base index class and distance functions for vector similarity
search. Provides a common interface for index implementations
(e.g. HNSW) and the distance metrics they can use.
"""

from enum import Enum, auto
import pickle as pk
import numpy as np


def cosine_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate cosine distance from vector X to all vectors Y.
    Normalizes both inputs then computes 1 - cosine similarity.
    Range: [0, 2], where 0 means identical direction.
    """
    x = _normalize(x)
    y = _normalize(y)
    return inner_product_distance(x, y)


def inner_product_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate inner product distance from vector X to all vectors Y.
    Returns 1 - dot(X, Y.T). Assumes pre-normalized inputs.
    Equivalent to cosine distance when vectors are unit-normalized.
    """
    return 1.0 - np.dot(x, y.T)


def l2_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate distance from 1 vector X to all vectors Y
    """

    return np.linalg.norm(x - y, axis=1)


def cosine_similarity(x: np.ndarray, y: np.ndarray):
    """
    Calculate cosine similarity from vector X to all vectors Y.
    Normalizes both inputs then computes dot(X, Y.T).
    Range: [-1, 1], where 1 means identical direction.
    """
    x = _normalize(x)
    y = _normalize(y)
    return np.dot(x, y.T)


def _normalize(x: np.ndarray):
    """
    L2-normalize each row of X to unit length.
    Divides each vector by its L2 norm, making all vectors
    lie on the unit hypersphere.
    """
    return x / np.expand_dims(np.linalg.norm(x, axis=1), axis=1)


class DistanceFunction(Enum):
    """
    Supported distance metrics for vector similarity search.
    Passed to Index.__init__ to select the distance function
    used during add() and search().
    """
    COSINE = auto()
    L2 = auto()
    INNER_PRODUCT = auto()


class Index:
    """
    Abstract base class for vector indexes.
    Handles vector storage, distance function selection,
    and serialization. Subclasses implement search().
    """

    n_total: int          # number of vectors currently stored
    vectors: None         # stored vectors as a 2D numpy array (n, d)
    is_trained: bool      # True once at least one vector has been added
    distance: int         # expected dimensionality of input vectors
    distance_function: callable(
        [np.ndarray, np.ndarray], np.ndarray)  # selected distance fn

    distance_function_map = {
        DistanceFunction.COSINE: cosine_distance,
        DistanceFunction.L2: l2_distance,
        DistanceFunction.INNER_PRODUCT: inner_product_distance
    }

    def __init__(self, distance: int, distance_func: DistanceFunction = DistanceFunction.COSINE):
        """
        Initialize an empty index.
        """
        self.n_total = 0
        self.vectors = None
        self.is_trained = False
        self.distance = distance
        self.distance_function = self.distance_function_map[distance_func]

    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.
        On first call, initializes the internal store and marks 
        index as trained. Subsequent calls append to the 
        existing store.
        """
        assert vectors.shape[1] == self.distance

        if self.vectors is None:
            self.vectors = vectors
            self.is_trained = True
        else:
            self.vectors = np.append(self.vectors, vectors, axis=0)

        self.n_total = self.vectors.shape[0]

    def search(
        self, query: np.ndarray, k: int
    ):
        """
        Find the k nearest neighbors to the query vector.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def save(self, file: str):
        """
        Serialize the index to disk using pickle.
        """
        with open(file, "wb") as f:
            pk.dump(self, f)

    @classmethod
    def from_file(cls, file: str):
        """
        Load and return a previously saved index from disk.
        """
        with open(file, "rb") as f:
            return pk.load(f)
