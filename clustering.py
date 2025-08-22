# clustering_pipeline.py

import pandas as pd
import numpy as np
import ast
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm

tqdm.pandas()

### 1.임베딩
# 모델 로딩
model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def get_embedding(text):
    return model.encode(text if isinstance(text, str) else "", convert_to_numpy=True)


def embed_documents(df, text_column='text'):
    tqdm.write("💡 임베딩 시작")
    df[text_column] = df[text_column].apply(lambda x: x if isinstance(x, str) else "")
    df['embedding'] = df[text_column].progress_apply(get_embedding)  # tqdm 적용
    return df


###2. 1차 군집이자 새로운 데이터 군집
def run_dbscan_on_embeddings(df, eps=0.25, min_samples=3):
    embeddings = np.array(df['embedding'].tolist())
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(embeddings)
    df['embedding_cluster_id'] = labels
    return df, labels


###3. 2차 군집
def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def get_tfidf_matrix(noun_lists, min_df=2, ngram_range=(1, 5),
                     stop_words=["것", "수", "때", "등", "이번", "오늘", "기자", "보도", "사진"]):
    text = [" ".join(nouns) for nouns in noun_lists]
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, stop_words=stop_words)
    return vectorizer.fit_transform(text).toarray()


def run_2nd_tfidf_clustering(
        df, cluster_col='embedding_cluster_id', noun_col='konlpy_nouns',
        result_col='tfidf_cluster_id', flag_col='second',
        eps=0.6, min_samples=3, min_docs=50, min_df=2, ngram_range=(1, 5)
):
    df = df.copy()
    df[noun_col] = df[noun_col].apply(safe_eval)
    df[result_col] = df[cluster_col]
    df[flag_col] = False

    existing_max = df[cluster_col][df[cluster_col] != -1].max()
    current_cluster_id = (existing_max + 1) if pd.notnull(existing_max) else 0

    for cluster_id in tqdm(sorted(df[cluster_col].unique()), desc="2차 군집 클러스터 루프"):
        if cluster_id == -1:
            continue
        sub_df = df[df[cluster_col] == cluster_id].copy()
        if len(sub_df) < min_docs:
            continue
        tfidf_matrix = get_tfidf_matrix(sub_df[noun_col].tolist(), min_df=min_df, ngram_range=ngram_range)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(tfidf_matrix)
        label_to_cluster_id = {}
        for idx, label in zip(sub_df.index, labels):
            if label == -1:
                continue
            if label not in label_to_cluster_id:
                label_to_cluster_id[label] = current_cluster_id
                current_cluster_id += 1
            df.at[idx, result_col] = label_to_cluster_id[label]
            df.at[idx, flag_col] = True

    return df


import numpy as np
from collections import defaultdict
from tqdm import tqdm


### 5. 새로운 데이터 군집화하기 위한 df1000의 centroid 계산
def compute_embedding_centroids_from_tfidf(df, embedding_col='embedding', cluster_col='tfidf_cluster_id'):
    centroids = {}
    cluster_embeddings = defaultdict(list)

    def parse_emb(x):
        if isinstance(x, str):
            return np.fromstring(x.strip("[]"), sep=" ")
        return x

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Centroid 계산"):
        cid = row[cluster_col]
        if cid == -1:
            continue
        emb = parse_emb(row[embedding_col])
        cluster_embeddings[cid].append(emb)

    for cid, vectors in cluster_embeddings.items():
        centroids[cid] = np.mean(np.vstack(vectors), axis=0)

    return centroids


###5. 새로운 데이터 군집화
def assign_new_docs_to_tfidf_cluster(new_df, centroids, threshold=0.85):
    new_df = new_df.copy()
    new_df['news_summary'] = new_df['news_summary'].apply(lambda x: x if isinstance(x, str) else "")
    new_df['embedding'] = new_df['news_summary'].progress_apply(get_embedding)

    cluster_ids = list(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])

    assigned_clusters = []
    for vec in tqdm(new_df['embedding'], desc="신규 문서 군집 배정"):
        sims = cosine_similarity([vec], centroid_matrix)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        assigned_clusters.append(cluster_ids[best_idx] if best_sim >= threshold else -1)

    new_df['assigned_tfidf_cluster'] = assigned_clusters
    return new_df


def run_third_tfidf_on_noise(
        df1000, new_df,
        noun_col='konlpy_nouns',
        eps=0.6, min_samples=2
):
    """
    - df1000: 2차까지 완료된 데이터 (tfidf_cluster_id 존재)
    - new_df: df500을 기존 tf-idf centroid로 매칭한 결과 (assigned_tfidf_cluster 존재)
    - 둘 다에서 클러스터 -1(노이즈)만 추려 TF-IDF DBSCAN으로 재군집
    - 새로 묶인 문서엔 third=True, 클러스터 ID는 기존 max+1부터 부여
    - noun_col은 이미 토큰화된 리스트형 컬럼 사용
    """
    df1000 = df1000.copy()
    new_df = new_df.copy()

    # third 플래그 기본값
    if 'third' not in df1000.columns:
        df1000['third'] = False
    if 'third' not in new_df.columns:
        new_df['third'] = False

    # 1) 노이즈만 추출
    noise_1000 = df1000[df1000['tfidf_cluster_id'] == -1].copy()
    noise_500 = new_df[new_df['assigned_tfidf_cluster'] == -1].copy()

    # 2) 결합
    toks_all = noise_1000[noun_col].apply(safe_eval).tolist() + \
               noise_500[noun_col].apply(safe_eval).tolist()

    noise_all = pd.concat([noise_1000, noise_500], axis=0)

    if len(noise_all) < min_samples:
        print("대상 수가 min_samples 미만이라 스킵합니다.")
        return df1000, new_df

    # 3) TF-IDF 행렬 생성
    tfidf_mat = get_tfidf_matrix(toks_all)

    # 4) DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(tfidf_mat)

    # 5) 새 cluster id 시작값
    existing_ids = []
    existing_ids += df1000.loc[df1000['tfidf_cluster_id'] != -1, 'tfidf_cluster_id'].tolist()
    existing_ids += new_df.loc[new_df['assigned_tfidf_cluster'] != -1, 'assigned_tfidf_cluster'].tolist()
    start_id = max(existing_ids) + 1 if existing_ids else 0

    label_to_newid = {}
    cur = start_id
    for lab in np.unique(labels):
        if lab == -1:
            continue
        label_to_newid[lab] = cur
        cur += 1

    # 6) df1000 반영
    n1000 = len(noise_1000)
    if n1000 > 0:
        for i, idx in enumerate(noise_1000.index):
            lab = labels[i]
            if lab != -1:
                new_id = label_to_newid[lab]
                df1000.at[idx, 'tfidf_cluster_id'] = new_id
                df1000.at[idx, 'third'] = True

    # 7) new_df 반영
    if len(noise_500) > 0:
        for j, idx in enumerate(noise_500.index):
            lab = labels[n1000 + j]
            if lab != -1:
                new_id = label_to_newid[lab]
                new_df.at[idx, 'assigned_tfidf_cluster'] = new_id
                new_df.at[idx, 'third'] = True

    return df1000, new_df


def merge_cluster_third_results(df1000, new_df):
    """
    df1000: 2차/3차 처리까지 반영된 기존 데이터 (tfidf_cluster_id 보유)
    new_df: centroid 매핑 및 3차 반영까지 끝난 신규 데이터
            (assigned_tfidf_cluster 보유, third가 True일 수 있음)

    반환: final_df (병합 결과 DataFrame)
    """
    df1000_export = df1000.copy()
    new_df_export = new_df.copy()

    df1000_export['is_new'] = False
    if 'second' not in df1000_export.columns:
        df1000_export['is_second'] = False
    # ✅ third → is_third 매핑을 "항상" 수행
    df1000_export['is_third'] = df1000_export.get('third', False).astype(bool)

    new_df_export['is_new'] = True
    new_df_export['is_second'] = False
    # ✅ 신규도 동일
    new_df_export['is_third'] = new_df_export.get('third', False).astype(bool)

    # 신규 → 최종 클러스터 컬럼 동기화
    if 'assigned_tfidf_cluster' in new_df_export.columns:
        new_df_export['tfidf_cluster_id'] = new_df_export['assigned_tfidf_cluster']
    elif 'tfidf_cluster_id' not in new_df_export.columns:
        new_df_export['tfidf_cluster_id'] = -1

    columns_to_export = [
        'tfidf_cluster_id', 'title', 'press', 'news_summary',
        'embedding', 'is_second', 'is_third', 'is_new', 'publish_date', 'news_link'
    ]

    for col in columns_to_export:
        if col not in df1000_export.columns:
            df1000_export[col] = None
        if col not in new_df_export.columns:
            new_df_export[col] = None

    final_df = pd.concat(
        [df1000_export[columns_to_export], new_df_export[columns_to_export]],
        ignore_index=True
    )
    final_df.to_csv("final_clustering.csv", index=False, encoding='utf-8-sig')

    return final_df


def assign_cluster_ids_with_df1000_seed(final_df: pd.DataFrame, db_max_id: int | None = None) -> pd.DataFrame:
    df = final_df.copy()
    df["tfidf_cluster_id"] = df["tfidf_cluster_id"].fillna(-1).astype(int)

    seed_candidates = []

    # (a) 기존에 이미 부여돼 있던 cluster_id도 후보에 포함
    if "cluster_id" in df.columns:
        seed_candidates += df.loc[df["cluster_id"] != -1, "cluster_id"].tolist()

    # (b) df1000의 진짜 기존 라벨만 (third 제외)
    mask_seed = (
        (df.get("is_new", False) == False) &
        (df.get("is_third", False) == False) &
        (df["tfidf_cluster_id"] != -1)
    )
    seed_candidates += df.loc[mask_seed, "tfidf_cluster_id"].tolist()

    # (c) DB MAX(id)도 후보에 포함
    if db_max_id is not None:
        seed_candidates.append(int(db_max_id))

    seed_max = max(seed_candidates) if seed_candidates else -1
    start_id = int(seed_max) + 1
    next_id = start_id

    # === 2) A: 신규 & 비노이즈 & 3차 아님 → 군집 단위 발급
    mask_A = (df["is_new"] == True) & (df["is_third"] == False) & (df["tfidf_cluster_id"] != -1)
    uniq_A = sorted(df.loc[mask_A, "tfidf_cluster_id"].unique().tolist())
    map_A = {lbl: (next_id + i) for i, lbl in enumerate(uniq_A)}
    next_id += len(uniq_A)

    # === 3) B: 신규 & 3차(재군집) & 비노이즈 → 군집 단위 발급
    mask_B = (df["is_new"] == True) & (df["is_third"] == True) & (df["tfidf_cluster_id"] != -1)
    uniq_B = sorted(df.loc[mask_B, "tfidf_cluster_id"].unique().tolist())
    map_B = {lbl: (next_id + i) for i, lbl in enumerate(uniq_B)}
    next_id += len(uniq_B)

    # === 4) 적용: 기본은 기존값 보존, 없으면 -1로 채운 뒤 A→B 매핑
    # 기존 df1000의 cluster_id가 있으면 그대로 보존
    if "cluster_id" not in df.columns:
        df["cluster_id"] = -1

    full_map = {**map_A, **map_B}
    need_map = df["tfidf_cluster_id"].isin(full_map.keys())
    df.loc[need_map, "cluster_id"] = df.loc[need_map, "tfidf_cluster_id"].map(full_map).astype(int)

    print(f"df1000 시드 seed_max={seed_max} → 시작={start_id}")
    print(f"A(신규·비노이즈) 군집 수: {len(uniq_A)}")
    print(f"B(신규·3차 재군집) 군집 수: {len(uniq_B)}")
    if (df["cluster_id"] != -1).any():
        print(f"최종 cluster_id 범위: {df['cluster_id'].min()} ~ {df['cluster_id'].max()}")
    else:
        print("최종 cluster_id 범위: 신규 배정 없음 (전부 -1)")
    return df


import os
from datetime import datetime
from preprocessing import preprocess_df
from summarizer import run_summarization
from utils import get_mysql_engine, get_cohere_api_key
from sqlalchemy import text

# ===== 군집 =====
ENGINE = None  # 전역 엔진 재사용


def run_clustering(final_df1000_path: str = "path/to/final_df1000.csv"):
    global ENGINE
    if ENGINE is None:
        try:
            ENGINE = get_mysql_engine()
        except Exception as e:
            print(f"❌ MySQL 엔진 생성 실패: {e}")
            ENGINE = None  # 엔진 없이도 CSV 저장만 하게끔

    if not os.path.exists(final_df1000_path):
        print(f"❌ {final_df1000_path} 파일 없음 → 로컬에서 df1000 처리 먼저 실행 필요")
        return

    # 1) df1000
    df1000 = pd.read_csv(final_df1000_path)

    # 2) 방금 크롤링 데이터
    latest_path = os.path.join(os.path.dirname(__file__), "crawling_latest.csv")
    if not os.path.exists(latest_path):
        print("❌ 최신 크롤링 데이터 없음")
        return

    df500 = pd.read_csv(latest_path)
    if df500.empty:
        print("❌ df500 데이터 없음 → 군집화 생략")
        return

    print(f"📥 df500 로드 완료: {len(df500)}개 기사")

    # 3) 센트로이드/군집 배정
    centroids = compute_embedding_centroids_from_tfidf(df1000)
    new_df = assign_new_docs_to_tfidf_cluster(df500, centroids)

    # 3차 군집화 실행
    df1000, new_df = run_third_tfidf_on_noise(
        df1000,  # 기존 데이터
        new_df,  # 신규 데이터
        noun_col='konlpy_nouns',  # 이미 토큰화된 명사 리스트 컬럼명
        eps=0.6,  # DBSCAN 파라미터
        min_samples=2  # DBSCAN 파라미터
    )

    # 5) 합치기
    final_df = merge_cluster_third_results(df1000, new_df)

    # 6) DB에서 현재 cluster_id 최대값 읽기
    if ENGINE:
        with ENGINE.connect() as conn:
            result = conn.execute(text("SELECT MAX(id) FROM topic"))
            max_cluster_id = result.scalar()
    else:
        max_cluster_id = 0
    print(f"현재 토픽DB 최대 cluster_id: {max_cluster_id}")

    # 7) df1000을 시드로 A→B 군집 단위 부여
    final_df = assign_cluster_ids_with_df1000_seed(final_df, db_max_id=max_cluster_id)

    if (final_df["cluster_id"] != -1).any():
        print(f"변환 후 cluster_id 범위: {final_df['cluster_id'].min()} ~ {final_df['cluster_id'].max()}")
    else:
        print("변환 후 cluster_id 범위: 신규 배정 없음 (전부 -1)")


    # 8) 요약 대상 cluster_id 선정  🔧 (3차로 '새로 생긴' + 기존에 '합류'한)
    # (a) 기존에 df1000 쪽에 존재하던 클러스터(노이즈 제외)
    preexisting_clusters = set(
        final_df.loc[
            (final_df["is_new"] == False) &
            (final_df["is_third"] == False) &   # ★ 추가: 3차에서 새 라벨로 바뀐 것 제외
            (final_df["tfidf_cluster_id"] != -1),
            "tfidf_cluster_id"
        ].unique()
    )

    # (b) 3차 재군집(third=True)으로 새로 생긴 클러스터들 (기존엔 없었음)
    third_created_tfidf = set(
        final_df.loc[
            (final_df["is_third"] == True) & 
            (final_df["tfidf_cluster_id"] != -1), 
            "tfidf_cluster_id"]
            .unique()
    ) - preexisting_clusters

    # (c) 신규(new=True) 문서가 기존(preexisting) 클러스터에 합류한 케이스
    joined_existing_tfidf = set(
        final_df.loc[
            (final_df["is_new"] == True) &
            (final_df["is_third"] == False) &
            (final_df["tfidf_cluster_id"] != -1) &
            (final_df["tfidf_cluster_id"].isin(preexisting_clusters)),
            "tfidf_cluster_id"
        ].unique()
    )

    # 오프셋 적용된 최종 cluster_id로 변환
    new_cluster_ids = set(
        final_df.loc[final_df["tfidf_cluster_id"].isin(third_created_tfidf), "cluster_id"].unique()
    )
    joined_cluster_ids = set(
        final_df.loc[final_df["tfidf_cluster_id"].isin(joined_existing_tfidf), "cluster_id"].unique()
    )

    target_clusters = new_cluster_ids.union(joined_cluster_ids)
    print(f"이번 요약 대상 cluster_id 개수: {len(target_clusters)}개")

    target_df = final_df[final_df["cluster_id"].isin(target_clusters)]

    # 9) 요약 실행
    api_key = get_cohere_api_key()
    today_str = datetime.today().strftime("%m%d")
    os.makedirs("data", exist_ok=True)
    filename1 = f"data/commandr_summary{today_str}.csv"
    filename2 = f"data/commandr_articles{today_str}.csv"

    print(f"📝 요약 시작: {len(target_clusters)}개 클러스터, {len(target_df)}개 기사")

    run_summarization(
        target_df,  # ✅ 이번에 필요한 군집만 전달
        "cluster_id",
        api_key,
        ENGINE,
        filename1,
        filename2
    )

    return {"status": "success", "message": "군집화 완료", "new_docs": len(df500)}