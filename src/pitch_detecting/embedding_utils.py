# <src/pitch_detecting/embedding_utils.py>

from src.timbre_encoding.timbre_encoder import timbre_enc

def add_or_search_embedding(faiss_index, vocal_wav_path, sbert_model=None, yt_id=None, yt_title=None, range_=None):
    embedding = timbre_enc(vocal_wav_path)
     # yt_title 값 존재 시, faiss adding에 해당. None일 경우, Searching based on user voice.
    if yt_title and (yt_id not in [s[0] for s in faiss_index.sets]):
        title_embed = sbert_model.encode([yt_title], batch_size=1)
        title_vector = title_embed[0]
        faiss_index.add_vector(title_vector, embedding, yt_id, yt_title, range_)
    else:
        return faiss_index.timbre_base_search(embedding)
