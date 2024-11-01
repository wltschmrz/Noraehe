# <main.py>

import os
import sys
import psutil
import time
sys.path.append(os.path.abspath('.'))
from src.pitch_detecting.audio_processor import AudioProcessor
from src.pitch_detecting.faiss_index import FAISSIndex
from sentence_transformers import SentenceTransformer
from src.pitch_detecting.vocal_range import VocalRange, KeyShiftCalculator

# -----------------------------------------------------------------------------------------------------------------
#                                            << Utility Functions >>
#
#                               - 편리성 위해 임시로 만들어 놓은 함수. 코드에 중요하지 X.
# -----------------------------------------------------------------------------------------------------------------
def print_memory_usage(plag=0):
    process = psutil.Process(os.getpid())  # 현재 프로세스 정보 가져오기
    memory_info = process.memory_info()    # 메모리 사용량 정보 가져오기
    print(f"{plag}. Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")  # RSS는 실제 메모리 사용량을 나타냅니다.
    time.sleep(0.5)  # 1초 대기 (조정 가능)


def printing_result(shift, octave_shift):
    shift_direction = '높임' if shift > 0 else '낮춤'
    octave_direction = '올려' if octave_shift > 0 else '낮춰'
    if shift == 0 and octave_shift == 0:
        print(">> 추천 키 조정: 없음")
    elif octave_shift == 0:
        print(
            f">> 추천 키 조정: {abs(shift)}번 음정 {shift_direction} 버튼을 누르세요.")
    else:
        print(
            f">> 추천 키 조정: {abs(shift)}번 음정 {shift_direction} 버튼을 누르고, "
            f"{abs(octave_shift)} 옥타브 {octave_direction} 부르세요.\n")


# -----------------------------------------------------------------------------------------------------------------
#                                            << Real Acting Functions >>
#
#                                 - 실제 기능 구현된 함수. 자세한 사항은 docstring 참조.
# -----------------------------------------------------------------------------------------------------------------
def initializing():
    """
    모든 기능을 사용하는데 앞서 미리 실행 시켜야 하는 함수. 추후 얻어낸 객체를 활용해 코드 돌아감.
        1. key_calculator: 두 범위를 받아 피치 조정 계산 해주는 객체.
        2. faiss_index: title embedding, timbre embedding 벡터들과 그 메타 데이터 저장 해놓은 데이터 파일을 다루는 객체.
            -> (title emb) (timbre emb) (yt_id, yt_title, range)
        3. sbert: title embedding 만들어줄 sbert 모델 객체

    return:
        - key_calculator, faiss_index, sbert: 객체들
        - current_dir, save_dir: 현재 경로, 저장 경로
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()  # ./
    
    save_dir = os.path.join(current_dir, 'data')  # ./data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    key_calculator = KeyShiftCalculator()
    faiss_index = FAISSIndex(data_path=save_dir, title_dim=512, timbre_dim=256)
    sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')     
    # sbert = SentenceTransformer('jhgan/ko-sbert-sts') --> 이 모델은 output이 약 700 dim 정도로 커서 일단 보류함.
    print_memory_usage(1)  # 메모리 사용량 출력
    return key_calculator, faiss_index, sbert, current_dir, save_dir


def get_user_pitch_range_from_wav(obj, wav_path, save_dir='./data'):
    """
    사용자 녹음 파일을 제공해주었을때 음색 유사도 측정해 노래 추천 set 출력하는 함수.
    웹 페이지 level에서 피치 파악 후 범위를 직접 기입해 주는 방식이 아닐 경우, 피치 범위 값 뽑는데에도 사용 가능.
    
    input:
        - wav_path (str): 사용자 목소리 녹음 wav 파일 저장 경로.
        - save_dir (str): 파일 취급 중 생성되는 중간 파일들 임시 저장할 경로. 프로세스 종료 후 clean_up 함.

    output:
        - user_range_ (Tuple: (str, str)): 사용자 피치 범위. Note 표기 방식으로 튜플 구간의 형태로 제공.
        - similar_sets (List: [List:[str, str, Tuple], ...]):
            사용자 목소리와 음색 유사한 노래 상위 5개 제공. 
            example:
            [['yt_id(1)', 'yt_title(1)', (A3, B3)],
            ['yt_id(2)', 'yt_title(2)', (A4, B4)],
            ['yt_id(3)', 'yt_title(3)', (A3, B4)],
            ['yt_id(4)', 'yt_title(4)', (A2, B3)],
            ['yt_id(5)', 'yt_title(5)', (A2, B2)]]
    """
    key_calculator, faiss_index, sbert = obj

    user_voice_processor = AudioProcessor(youtube_url=None, save_dir=save_dir, original_wav_path=wav_path)
    similar_sets, user_pitch_ = user_voice_processor.vocal_base_searching(faiss_index)
    user_range_ = VocalRange(user_pitch_[0], user_pitch_[1])
    return similar_sets, user_range_


def pitch_adjustment_from_yt_link(obj, youtube_url, user_pitch=('E2', 'A#3'), save_dir='./data'):
    """
    제목 검색 시에도 나오지 않는 곡의 경우에 사용하는 함수.
    링크를 기입하면 그에 해당하는 곡의 피치 조절값을 제안해줌. 동시에, 얻어낸 정보를 데이터베이스에 저장함.

    input:
        - url (str): 해당 곡의 단일 Youtube 링크 str.
        - user_pitch (Tuple:(str,str)): 사용자의 피치 범위. 노트 표기법으로 튜플 구간 형태.

    output:
        - key_shift (int): 음정 조절 값.
        - octave_adjustment (int): 옥타브 조절 값.
    """
    key_calculator, faiss_index, sbert = obj

    user_range = VocalRange(user_pitch[0], user_pitch[1])

    processor = AudioProcessor(youtube_url, save_dir)
    range_notes = processor.process(faiss_index, sbert)

    song_range = VocalRange(range_notes[0], range_notes[1])
    key_shift, octave_adjustment = key_calculator.calculate_key_shift(song_range.get_range(), user_range.get_range())
    printing_result(key_shift, octave_adjustment)
    print_memory_usage(2)  # 메모리 사용량 출력
    return key_shift, octave_adjustment


def search_from_database_with_title(obj, query_title, k=5):
    """
    제목 검색해서 기존 데이터베이스의 값을 불러오는 함수.
    세트 당 yt_id, yt_title, pitch_range 로 구성된 리스트로 묶여있다.
    
    input:
        - wav_path (str): 사용자 목소리 녹음 wav 파일 저장 경로.
        - save_dir (str): 파일 취급 중 생성되는 중간 파일들 임시 저장할 경로. 프로세스 종료 후 clean_up 함.

    output:
        - user_range_ (Tuple: (str, str)): 사용자 피치 범위. Note 표기 방식으로 튜플 구간의 형태로 제공.
        - similar_sets (List: [List:[str, str, Tuple], ...]):
            사용자 목소리와 음색 유사한 노래 상위 5개 제공. 
            example:
            [['yt_id(1)', 'yt_title(1)', (A3, B3)],
            ['yt_id(2)', 'yt_title(2)', (A4, B4)],
            ['yt_id(3)', 'yt_title(3)', (A3, B4)],
            ['yt_id(4)', 'yt_title(4)', (A2, B3)],
            ['yt_id(5)', 'yt_title(5)', (A2, B2)]]
    """
    key_calculator, faiss_index, sbert = obj
    text_emb = sbert.encode([query_title], batch_size=1)
    sets = faiss_index.title_base_search(query_vector=text_emb[0], top_k=k)
    return sets


if __name__ == "__main__":

    ## 초기 설정 ----------------------------------------------------------------------------------------------------------------
    key_calculator, faiss_index, sbert, current_dir, save_dir = initializing()
    obj = (key_calculator, faiss_index, sbert)

    ## 1. 제목 검색 후 기존 데이터베이스로부터 '피치 조절 제안' 제공 -------------------------------------------------------------------

    my_singing_voice = './src/sikyoung.wav'                                         # -> 파일로 내 목소리 받거나,
    _, my_pitch = get_user_pitch_range_from_wav(obj, my_singing_voice, save_dir)
    # (또는)
    my_pitch = ('E2', 'A#3')                                                        # -> 웹에서 실시간 파악 후 직접 범위 기입
    user_range = VocalRange(my_pitch[0], my_pitch[1])

    searching_title = '너의 모둠 순대 간'                                              # -> 검색어 기입

    sets = search_from_database_with_title(obj, searching_title, k=5)
    for set in sets:
        print(f"link: {set[0]} / title: {set[1]} / range: {set[2]}")
        song_range_ = VocalRange(set[2][0], set[2][1])
        key_shift, octave_adjustment = key_calculator.calculate_key_shift(song_range_.get_range(), user_range.get_range())
        printing_result(key_shift, octave_adjustment)

    ## 2. 검색 결과에 없으면 유튜브 링크 기입 후 '피치 조절 제안' 받기 (동시에 데이터베이스 자동 업데이트) -------------------------------

    youtube_url = 'https://www.youtube.com/watch?v=evOsUf9en-Y'  # 너의 모든순간       # -> 유튜브 링크 기입
    key_shift, octave_adjustment = pitch_adjustment_from_yt_link(obj, youtube_url, user_pitch=my_pitch, save_dir=save_dir)

    ## 3. 내 목소리 음색에 어울리는 음악 추천. '음색 유사도' 검색 (피치 제안도 같이 제공) -------------------------------------------------

    my_singing_voice = './src/sikyoung.wav'                                          # -> 내 음색을 보여줄수 있는 내가 부른 음악 파일
    similar_sets, _ = get_user_pitch_range_from_wav(obj, my_singing_voice, save_dir)

    for set in similar_sets:
        print(f"link: {set[0]} / title: {set[1]} / range: {set[2]}")
        song_range_ = VocalRange(set[2][0], set[2][1])
        key_shift, octave_adjustment = key_calculator.calculate_key_shift(song_range_.get_range(), _.get_range())
        printing_result(key_shift, octave_adjustment)

    print_memory_usage(3)  # 메모리 사용량 출력

