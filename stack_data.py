# 웹 크롤링으로 얻어낸 링크 list로 database 바로 저장하는 용도의 파일

import os
import sys
sys.path.append(os.path.abspath('.'))
from src.pitch_detecting.audio_processor import AudioProcessor
from src.pitch_detecting.faiss_index import FAISSIndex
from sentence_transformers import SentenceTransformer
from src.pitch_detecting.vocal_range import VocalRange, KeyShiftCalculator




if __name__ == "__main__":
    urls = [
        'https://www.youtube.com/watch?v=evOsUf9en-Y',  # 너의 모든순간
        'https://www.youtube.com/watch?v=lAq9l8o6UXU',  # 내가 아니라도
        'https://www.youtube.com/watch?v=uCT4YCk6cvs',  # 사랑하지 않아서 그랬어
        'https://www.youtube.com/watch?v=4Sd09Mruhnk',  # 솔직하게 말해서 나
        'https://www.youtube.com/watch?v=FIdFoxVnGgE',  # 우리 왜 헤어져야 해
        'https://www.youtube.com/watch?v=uqQqnWfJyAA',  # 지나오다
    ]

    dir = './data'
    faiss_index = FAISSIndex(data_path=dir, title_dim=512, timbre_dim=256)
    sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    for url in urls:
        processor = AudioProcessor(url, dir)
        processor.process(faiss_index, sbert)