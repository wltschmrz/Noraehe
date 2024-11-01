# <src/pitch_detecting/faiss_index.py>

import faiss
import numpy as np
import pickle
import os


class FAISSIndex:
    def __init__(self, data_path, title_dim=384, timbre_dim=256):
        self.title_dim = title_dim
        self.timbre_dim = timbre_dim
        self.index_title = faiss.IndexFlatL2(self.title_dim)    # L2 거리 기반 index
        self.index_timbre = faiss.IndexFlatL2(self.timbre_dim)  # L2 거리 기반 index
        self.sets = []  # ('id', 'title', ('str', 'str'))

        self.index_title_path = os.path.join(data_path, 'faiss/faiss_index_title.bin')
        self.index_timbre_path = os.path.join(data_path, 'faiss/faiss_index_timbre.bin')
        self.sets_path = os.path.join(data_path, 'faiss/sets_path.pkl')

        if os.path.exists(self.index_title_path) and os.path.exists(self.index_timbre_path) and os.path.exists(self.sets_path):
            self.load(self.index_title_path, self.index_timbre_path, self.sets_path)
        else:
            if not os.path.exists(os.path.join(data_path, 'faiss')):
                os.makedirs(os.path.join(data_path, 'faiss'))
            print(f"No saved file, generating new FAISS index.")

        assert self.index_title.d == self.title_dim and self.index_timbre.d == self.timbre_dim, "Wrong dim"

    def add_vector(self, title_vector, timbre_vector, id_, title_, range_):
        assert isinstance(timbre_vector, np.ndarray) and timbre_vector.shape == (self.timbre_dim,)
        assert title_vector.shape == (self.title_dim,)
        self.index_title.add(np.array([title_vector], dtype='float32'))       # 벡터 추가
        self.index_timbre.add(np.array([timbre_vector], dtype='float32'))  # 벡터 추가
        self.sets.append([id_, title_, range_])

    def timbre_base_search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector], dtype='float32')
        distances, indices = self.index_timbre.search(query_vector, top_k)
        return [self.sets[i] for i in indices[0]]
    
    def title_base_search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector], dtype='float32')
        distances, indices = self.index_title.search(query_vector, top_k)
        return [self.sets[i] for i in indices[0]]

    def save(self):
        faiss.write_index(self.index_title, self.index_title_path)
        faiss.write_index(self.index_timbre, self.index_timbre_path)
        with open(self.sets_path, 'wb') as f:
            pickle.dump(self.sets, f)

    def load(self, index_title_path, index_timbre_path, sets_path):
        self.index_title = faiss.read_index(index_title_path)
        self.index_timbre = faiss.read_index(index_timbre_path)
        with open(sets_path, 'rb') as f:
            self.sets = pickle.load(f)