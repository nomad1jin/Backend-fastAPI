import pandas as pd
import re
from datetime import datetime
from konlpy.tag import Komoran
from tqdm import tqdm
from dotenv import load_dotenv


def clean_content(text):
    if pd.isna(text):
        return text

    # 이메일 제거 (공백 포함 허용)
    text = re.sub(r'\b[\w.-]+ ?@ ?[\w.-]+\.\w+\b', '', text)

    # @아이디 제거
    text = re.sub(r'@\w+', '', text)

    # URL 제거
    text = re.sub(r'https?://\S+|url\.kr/\S+', '', text)

    # 특수 기호 제거
    text = text.replace('△', '')
    text = text.replace('◇', '')
    text = text.replace('▲', '')
    text = text.replace('■', '')


    # 대괄호 [내용] 제거
    text = re.sub(r'\[.*?\]', '', text)

    # 한자 제거 (기본 + 확장)
    # text = re.sub(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]', '', text)

    # ※ 이후 모두 제거 (뉴스사 공지 등)
    text = re.sub(r'※.*', '', text)

    # "ⓒ"로 시작해서 마침표(또는 줄 끝)까지 삭제
    text = re.sub(r'ⓒ[^.\n]*[.\n]?', '', text)
    text = re.sub(r'▲[^.\n]*[.\n]?', '', text)

    # 따옴표인데 크롤링도중에 \까지 붙음
    text = text.replace("\\'", "")  
    text = re.sub(r'["\'“”‘’`´]', '', text)

    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # 중복 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text


### 전처리 
# 한자 → 한글 매핑 테이블
hanja_to_korean = {
    '中': '중국', '美': '미국', '日': '일본', '亞': '아시아', '人': '사람', '企': '기업', '來': '오다',
    '價': '값', '兆': '조', '免': '면제', '協': '협력', '印': '인도', '反': '반대', '多': '많다',
    '女': '여성', '尹': '윤석열', '弗': '달러', '後': '후', '農心': '농심', '故': '고인',
    '月': '달', '李': '이재명', '株': '주식', '比': '대비', '洪': '홍준표', '獨': '독일', '發': '발',
    '社': '회사', '行': '행', '證': '증권', '車': '자동차', '軍': '군대', '辛': '신', '農': '농업',
    '重': '무거움', '金': '금', '韓': '한국'
}

# 한자 변환 함수
def convert_hanja_to_korean(text, mapping):

    return ''.join([mapping.get(char, char) for char in str(text)])

# publish_date 형식 변환
def convert_korean_datetime(dt_str):
    """
    "2025.07.10. 오후 7:58" 같은 문자열을 datetime 객체로 변환
    """
    try:
        # 공백 제거
        dt_str = dt_str.strip()

        # 오전/오후 분리
        match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})\.\s*(오전|오후)\s*(\d+):(\d+)", dt_str)
        if not match:
            return pd.NaT

        year, month, day, ampm, hour, minute = match.groups()
        hour = int(hour)
        minute = int(minute)

        if ampm == "오전":
            if hour == 12:
                hour = 0
        elif ampm == "오후":
            if hour != 12:
                hour += 12

        dt = datetime(int(year), int(month), int(day), hour, minute)
        return dt
    except:
        return pd.NaT


### konlpy 명사 추출
def get_nouns_with_konlpy(df, column):
    komoran = Komoran()
    print("konlpy 명사 추출 시작\n")
    tqdm.pandas()  # tqdm과 함께 사용할 수 있도록 설정

    df[column] = df[column].fillna("").astype(str)

    # 새 컬럼명: 예를 들어 'title' → 'title_nouns'
    # new_column = f"{column}_nouns"
    new_column = "konlpy_nouns"
    df[new_column] = df[column].progress_apply(komoran.nouns)
    
    return df


# ===== 전처리 =====
def preprocess_df(df):
    df['title'] = df['title'].fillna('').apply(
        lambda x: convert_hanja_to_korean(x, hanja_to_korean)
    )
    df['title'] = df['title'].apply(clean_content)
    df['news_summary'] = df['news_summary'].fillna('').apply(
        lambda x: convert_hanja_to_korean(x, hanja_to_korean)
    )
    df['publish_date'] = df['publish_date'].apply(convert_korean_datetime)
    df = get_nouns_with_konlpy(df, 'news_summary')
    return df

