import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

from S3_crawler import attach_cluster_images
from image_not_empty import wait_for_image_urls, _normalize_image_url_series
from utils import get_cohere_api_key, log_failed_cluster
from preprocessing import clean_content

load_dotenv()
api_key = get_cohere_api_key()


def call_commandr_cohere(prompt, cohere_api_key, max_retries=3):
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {cohere_api_key}",
        "Content-Type": "application/json",
        "Cohere-Version": "2024-04-08"
    }
    payload = {
        "model": "command-r-plus",
        "temperature": 0.3,  # 더 안정적인 출력을 위해 낮춤
        "max_tokens": 4000,
        "chat_history": [],
        "message": prompt
    }

    titles = re.findall(r'\d+\.\s+title:\s*(.*)', prompt)
    print(f"\n📝 클러스터 요약 요청 - {len(titles)}개 기사")
    print("-" * 50)

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 429:
            wait_time = 2 ** attempt
            print(f"🚫 429 Too Many Requests. {wait_time}초 후 재시도...")
            time.sleep(wait_time)
            continue
        elif response.status_code != 200:
            print(f"에러 발생: {response.status_code}")
            return f"[요약 실패: {response.status_code}]"
        else:
            print("✅ 응답 수신 완료")
            return response.json()["text"]

    return "[요약 실패: 429 에러 지속 발생]"


def make_commandr_summary_cohere(cluster_df, cluster_id, api_key):
    """간소화된 요약 생성 - title만 반환하도록 개선"""
    texts = []
    for idx, row in cluster_df.iterrows():
        doc = (
            f"{idx + 1}. title: {row['title']}\n"
            f"news_summary: {row['news_summary']}\n"
            f"press: {row['press']}\n"
        )
        texts.append(doc)

    combined = "\n\n".join(texts)

    # 간소화된 프롬프트 - 제목만 반환하도록
    prompt = f"""반드시 주어진 기사 데이터만을 사용하여, 뉴스 기사들을 분석하여 JSON으로 응답해주세요.

기사 데이터:
{combined}

다음 JSON 형식으로만 응답하세요:

{{
  "cluster_id": {cluster_id},
  "topic_name": "이 기사들의 공통 주제를 한 줄로 정리",
  "ai_summary": "통합 요약문을 6문장 이상으로 작성",
  "summary_time": "{datetime.now().strftime('%Y-%m-%d %H:%M')}",
  "article_titles": [
    "기사 제목1",
    "기사 제목2"
  ]
}}

중요:
1. 반드시 위 JSON 형식만 출력
2. article_titles에는 분석한 기사의 정확한 제목만 포함
3. cluster_id는 숫자 {cluster_id}로 설정"""

    return call_commandr_cohere(prompt, api_key)


def match_articles_from_csv(df, article_titles, cluster_id, cluster_col="cluster_id"):
    """CSV에서 제목을 매칭하여 완전한 기사 정보 추출"""
    cluster_df = df[df[cluster_col] == cluster_id].copy() 

    # 🔍 디버그 출력: 요약이 준 제목 vs 실제 DF 제목
    print("[DEBUG] 요약이 준 제목:", article_titles)
    print("[DEBUG] cluster_df csv 제목:", cluster_df['title'].head(10).tolist())

    # 제목 한번더 정규화 (df)
    cluster_df["__norm_title__"] = cluster_df["title"].astype(str).map(clean_content)
    # 제목 한번더 정규화 (요약)
    norm_predicted = [clean_content(t) for t in (article_titles or [])]
    print("[DEBUG] normalized 요약 제목:", norm_predicted)
    print("[DEBUG] normalized cluster_df csv 제목:", cluster_df["__norm_title__"].head(10).tolist())

    cluster_articles = []
    used_idx = set()

    for key in norm_predicted:
        # 정규화된 키로 매칭
        candidates = cluster_df[~cluster_df.index.isin(used_idx)]
        matched = candidates[candidates["__norm_title__"] == key]
        if not matched.empty:
            row = matched.iloc[0]
            used_idx.add(row.name)
            cluster_articles.append({
                "title": row['title'],
                "news_summary": row.get('news_summary'),
                "press": row.get('press'),
                "news_link": row.get('news_link'),
                "publish_date": row.get('publish_date'),
                "image_url": "",
                "is_new": int(row.get('is_new', 0)),
                "is_third": int(row.get('is_third', 0)),
            })
        else:
            print(f"[MATCH_MISS] not found for key='{key}'")

    print(f"✅ 최종 매칭된 기사: {len(cluster_articles)}개")
    return cluster_articles


def safe_json_parse(summary_result):
    """JSON 파싱 시도"""
    if not summary_result or not summary_result.strip():
        print("❌ 응답이 비어있음")
        return None

    # 기본 파싱 시도
    try:
        return json.loads(summary_result.strip())
    except Exception as e:
        print(f"❌ 기본 JSON 파싱 실패: {e}")
        print(f"📄 응답 내용 (처음 100자):\n{summary_result[:100]}")

    # 코드블록에서 추출 시도
    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, summary_result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception as e:
            print(f"❌ 코드블록 JSON 파싱 실패: {e}")
            print(f"📄 코드블록 내용:\n{match.group(1)}")

    print("❌ 모든 JSON 파싱 방법 실패")
    return None


def save_failed_json(cluster_id, json_response, failed_json_path="data/failed_responses.jsonl"):
    """실패한 JSON 응답 저장"""
    os.makedirs(os.path.dirname(failed_json_path), exist_ok=True)

    failed_record = {
        "cluster_id": int(cluster_id),  # int64 -> int 변환
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response": json_response
    }

    with open(failed_json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(failed_record, ensure_ascii=False) + "\n")


def _ensure_articles_schema(path):
    """기사 CSV 스키마 보장 - 디렉토리도 함께 생성"""
    import os
    from pathlib import Path

    # 디렉토리 생성
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cols = ["cluster_id", "title", "news_link", "press", "publish_date",
            "news_summary", "is_new", "is_third"]
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ 기사 CSV 스키마 파일 생성: {path}")


def _ensure_summary_schema(path):
    """요약 CSV 스키마 보장 - 디렉토리도 함께 생성"""
    import os
    from pathlib import Path

    # 디렉토리 생성
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cols = ["cluster_id", "summary_time", "topic_name", "ai_summary", "image_url"]
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ 요약 CSV 스키마 파일 생성: {path}")


def save_commandr_output_to_csv(df, summary_result, cluster_id, summary_csv_path, articles_csv_path,
                                cluster_col="cluster_id"):
    print(f"\n💾 save_commandr_output_to_csv 시작 - cluster_id: {cluster_id}")

    summary_data = safe_json_parse(summary_result)
    if not summary_data:
        print(f"❌ 클러스터 {cluster_id}: JSON 파싱 실패")
        save_failed_json(cluster_id, summary_result)
        _ensure_articles_schema(articles_csv_path)
        _ensure_summary_schema(summary_csv_path)
        return

    _ensure_articles_schema(articles_csv_path)
    _ensure_summary_schema(summary_csv_path)

    summary_row = {
        "cluster_id": int(summary_data.get("cluster_id", cluster_id)),
        "summary_time": summary_data.get("summary_time", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "topic_name": summary_data.get("topic_name", ""),
        "ai_summary": summary_data.get("ai_summary", ""),
        "image_url": ""
    }

    article_titles = summary_data.get("article_titles", [])
    print(f"🔍 기사 매칭 시작 - 대상 제목: {len(article_titles)}개")
    matched_articles = match_articles_from_csv(df, article_titles, cluster_id, cluster_col=cluster_col)  # ← 여기
    print(f"🔍 매칭 완료 - 결과: {len(matched_articles)}개")

    # 기사 데이터 준비
    article_rows = []
    for a in matched_articles:
        article_rows.append({
            "cluster_id": int(summary_data.get("cluster_id", cluster_id)),
            "title": a.get("title"),
            "news_link": a.get("news_link"),
            "press": a.get("press"),
            "publish_date": a.get("publish_date"),
            "news_summary": a.get("news_summary", ""),
            "is_new": int(a.get("is_new", 0)),
            "is_third": int(a.get("is_third", 0)),
        })

    print(f"📝 저장할 데이터 - summary: 1행, articles: {len(article_rows)}행")


    # 요약 저장
    try:
        pd.DataFrame([summary_row]).to_csv(
            summary_csv_path, mode="a", header=not os.path.exists(summary_csv_path),
            index=False, encoding="utf-8-sig"
        )
        print(f"✅ 요약 저장 완료: {cluster_id}")

        # 저장 후 즉시 확인
        test_summ = pd.read_csv(summary_csv_path)
        print(f"📊 요약 저장 후 파일 상태: {len(test_summ)}행")

    except Exception as e:
        print(f"❌ 요약 저장 실패: {cluster_id}, {e}")

    # 기사 저장
    try:
        if article_rows:
            file_exists = os.path.exists(articles_csv_path)

            # ✅ 쓰기 전 정렬 (cluster_id, publish_date 오름차순)
            arts_df = pd.DataFrame(article_rows)

            # publish_date 문자열 → 정규화
            def _parse_dt(s):
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                    try:
                        return pd.to_datetime(s, format=fmt)
                    except Exception:
                        pass
                return pd.to_datetime(s, errors="coerce")
            arts_df["_pub"] = arts_df["publish_date"].apply(_parse_dt)

            arts_df = arts_df.sort_values(["cluster_id", "_pub", "title"], ascending=[True, True, True]).drop(columns=["_pub"])

            arts_df.to_csv(
                articles_csv_path, mode="a", header=not file_exists,
                index=False, encoding="utf-8-sig"
            )
            print(f"✅ 기사 저장 완료: {cluster_id} ({len(arts_df)}건)")


            # 저장 후 즉시 확인
            test_arts = pd.read_csv(articles_csv_path)
            print(f"📊 기사 저장 후 파일 상태: {len(test_arts)}행")

        else:
            print(f"⚠️ 클러스터 {cluster_id}: 매칭된 기사 없음")

    except Exception as e:
        print(f"❌ 기사 저장 실패: {cluster_id}, {e}")
        import traceback
        traceback.print_exc()

'''
삽입 전 내부 중복 제거
news_df.drop_duplicates(subset=["title","news_link"])로 메모리 상에서 한 번 정리.

DB에 이미 있는 것 제외
해당 topic_id의 기존 (title, news_link)를 조회해서 merge 후 새로운 것만 to_sql("append")
'''


def save_commandr_output_to_db(df, summary_result, cluster_id, engine, cluster_col="cluster_id"):
    if engine is None:
        print("⚠️ engine=None: DB 저장 스킵")
        return

    # 1) JSON 파싱
    summary_data = safe_json_parse(summary_result)
    if not summary_data:
        print(f"❌ 클러스터 {cluster_id}: JSON 파싱 실패")
        save_failed_json(cluster_id, summary_result)
        return

    cid = int(summary_data.get("cluster_id", cluster_id))

    # 2) topic 존재 여부 확인 (테이블 미존재 대비)
    try:
        existing_topics = pd.read_sql("SELECT id FROM topic", engine)
        existing_ids = pd.to_numeric(existing_topics["id"], errors="coerce").dropna().astype(int).values
        topic_exists = cid in existing_ids
    except Exception as e:
        print(f"ℹ️ topic 조회 실패(최초 실행 가능성): {e}")
        topic_exists = False

    # 3) topic: 없을 때만 삽입  ->  존재하면 교체(UPDATE)
    summary_row = {
        "id": cid,
        "summary_time": summary_data.get("summary_time", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "topic_name": summary_data.get("topic_name", ""),
        "ai_summary": summary_data.get("ai_summary", "")
    }

    if not topic_exists:
        pd.DataFrame([summary_row]).to_sql("topic", engine, index=False, if_exists="append")
        print(f"🟢 topic 삽입: {cid}")
    else:
        # ✅ 기존 topic 교체(UPDATE)
        with engine.begin() as conn:
            res = conn.execute(
                text("""
                    UPDATE topic
                    SET topic_name  = :topic_name,
                        ai_summary  = :ai_summary,
                        summary_time= :summary_time
                    WHERE id = :id
                """),
                summary_row
            )
        print(f"🟡 topic 업데이트: {cid} (rows={res.rowcount})")


    # 4) 기사 매칭
    article_titles = summary_data.get("article_titles", []) or []
    matched_articles = match_articles_from_csv(df, article_titles, cid, cluster_col=cluster_col)
    print(f"[DB_PREVIEW] topic_id={cid} matched={len(matched_articles)} "
          f"sample={[{'title': a.get('title'), 'link': a.get('news_link')} for a in matched_articles[:3]]}")

    # 4-1) 요약 CSV에서 이 클러스터의 대표 image_url 로드 (없으면 빈 문자열)
    cluster_img = ""
    try:
        today_str = datetime.now().strftime("%m%d")  # 여기서 하나는 날짜가 들어가고 하나는 안 들어가여?
        cands = [f"data/commandr_summary{today_str}.csv", "data/commandr_summary.csv"]
        img_map = {}
        for p in cands:
            if os.path.exists(p):
                _summ = pd.read_csv(p)
                if {"cluster_id", "image_url"}.issubset(_summ.columns):
                    img_map = dict(zip(_summ["cluster_id"].astype(int), _summ["image_url"]))
                    print(f"[IMG_MAP_LOAD_OK] file={p} rows={len(img_map)}")
                    break
        cluster_img = (img_map.get(cid) or "").strip() if img_map else ""
        print(f"[CLUSTER_IMG_PICK] cid={cid} image_url='{cluster_img}'")
    except Exception as e:
        print(f"[IMG_MAP_ERR] cid={cid} err={e}")

    # 4-2) DB에 들어갈 행 생성 (대표 이미지 주입)
    article_rows = [{
        "topic_id": cid,
        "title": a.get("title"),
        "news_link": a.get("news_link"),
        "press": a.get("press"),
        "publish_date": a.get("publish_date"),
        "image_url": cluster_img,  # ★ 여기서 빈값 대신 대표 이미지 넣음
        "news_summary": a.get("news_summary", ""),  # ← ★ 추가!!!!!!!!!!
        "is_new": int(a.get("is_new", 0)),  # --- ADD
        "is_third": int(a.get("is_third", 0)),  # --- ADD
    } for a in matched_articles]

    if not article_rows:
        print(f"ℹ️ 클러스터 {cid}: matched_articles 비어 있음 → news 삽입 없음")
        return

    news_df = pd.DataFrame(article_rows)
    news_df["topic_id"] = news_df["topic_id"].astype(int)
    news_df = news_df.drop_duplicates(subset=["title", "news_link"])

    print(f"[DB_PRE_DUPS] topic_id={cid} rows={len(news_df)} "
          f"NaN_image={news_df['image_url'].isna().sum()} "
          f"empty_image={(news_df['image_url'].astype(str).str.strip() == '').sum()} "
          f"sample={news_df[['title', 'image_url']].head(3).to_dict(orient='records')}")

    # 5) 기존 news와 중복 제거 (테이블 미존재 대비)
    try:
        existing_news = pd.read_sql(
            "SELECT title, news_link FROM news WHERE topic_id = %s",
            engine, params=[cid]
        )
        news_df = news_df.merge(existing_news, on=["title", "news_link"], how="left", indicator=True)
        news_df = news_df[news_df["_merge"] == "left_only"].drop(columns=["_merge"])
    except Exception as e:
        print(f"ℹ️ news 조회 실패(최초 실행 가능성): {e}")
        # 첫 실행이면 그대로 진행

    if len(news_df) > 0:
        print(f"[DB_INSERT_READY] topic_id={cid} insert_rows={len(news_df)} "
              f"sample={news_df[['title', 'image_url']].head(5).to_dict(orient='records')}")
        news_df.to_sql("news", engine, index=False, if_exists="append")
        print(f"✅ news 삽입: {cid} / {len(news_df)}건")
    else:
        print(f"ℹ️ news 중복으로 신규 없음: {cid}")

#### 추가) publish_date를 기준으로 24시간전 데이터면 is_new를 false로 갱신!!!!     
def normalize_is_new(engine):
    with engine.begin() as conn:
        before_new = conn.execute(
            text("SELECT COUNT(*) FROM news WHERE is_new = 1")
        ).scalar()

        result = conn.execute(text("""
            UPDATE news
            SET is_new = CASE
                WHEN is_new = 1
                    AND COALESCE(
                            STR_TO_DATE(publish_date, '%Y-%m-%d %H:%i:%s'),
                            STR_TO_DATE(publish_date, '%Y-%m-%d %H:%i')
                        ) < CONVERT_TZ(NOW(), @@session.time_zone, '+09:00') - INTERVAL 24 HOUR
                THEN 0
                ELSE is_new
            END;
        """))
        updated_rows = result.rowcount

        after_new = conn.execute(
            text("SELECT COUNT(*) FROM news WHERE is_new = 1")
        ).scalar()

    print(f"is_new 정규화 완료: 변경된 행 {updated_rows}개, "
          f"NEW(1) {before_new} → {after_new}")

def run_summarization(
        df,
        cluster_col: str,
        api_key: str,
        engine=None,
        summary_csv_path: str = "data/commandr_summary.csv",
        articles_csv_path: str = "data/commandr_articles.csv"):
    _ensure_articles_schema(articles_csv_path)
    _ensure_summary_schema(summary_csv_path)

    df = df.copy()
    if cluster_col not in df.columns:
        raise ValueError(f"❌ 클러스터 컬럼 '{cluster_col}' 없음")

    df = df.dropna(subset=[cluster_col])
    df[cluster_col] = df[cluster_col].astype(int)

    valid_clusters = df[df[cluster_col] != -1][cluster_col].unique()
    large_clusters = []

    for cluster_id in valid_clusters:
        cluster_df = df[df[cluster_col] == cluster_id]
        if len(cluster_df) >= 2:
            large_clusters.append(cluster_id)

    print(f"📊 요약 대상 클러스터: {len(large_clusters)}개")

    if len(large_clusters) == 0:
        print("⚠️ 요약할 클러스터가 없습니다 (모든 클러스터가 2개 미만)")
        return

    # 클러스터별 요약 처리
    processed_count = 0
    for cluster_id in tqdm(sorted(large_clusters), desc="요약 중"):
        cluster_df = df[df[cluster_col] == cluster_id]

        try:
            print(f"\n🔄 클러스터 {cluster_id} 처리 중... ({len(cluster_df)}개 기사)")
            summary_result = make_commandr_summary_cohere(cluster_df, cluster_id, api_key)

            if summary_result and not summary_result.startswith("[요약 실패"):
                save_commandr_output_to_csv(
                    df, summary_result, cluster_id,
                    summary_csv_path, articles_csv_path,
                    cluster_col=cluster_col   # ← 추가
                )
                processed_count += 1
            else:
                print(f"❌ 클러스터 {cluster_id}: API 요청 실패")
                # log_failed_cluster(cluster_id)

        except Exception as e:
            print(f"❌ 클러스터 {cluster_id} 요약 실패: {e}")
            # log_failed_cluster(cluster_id)
            continue

    print(f"\n✅ 요약 완료: {processed_count}/{len(large_clusters)}개 클러스터 처리")

    # 즉시 파일 상태 확인
    print(f"\n📄 CSV 파일 상태 확인:")
    for path in [summary_csv_path, articles_csv_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            try:
                test_df = pd.read_csv(path)
                print(f"   {path}: {size} bytes, {len(test_df)} 데이터 행")
                if len(test_df) > 0 and 'cluster_id' in test_df.columns:
                    print(f"      cluster_id: {test_df['cluster_id'].tolist()}")
            except Exception as e:
                print(f"      ❌ 읽기 실패: {e}")

    # 이미지 처리
    try:
        if os.path.exists(articles_csv_path) and os.path.exists(summary_csv_path):
            arts_test = pd.read_csv(articles_csv_path)
            summ_test = pd.read_csv(summary_csv_path)

            if len(arts_test) > 0 and len(summ_test) > 0:
                print(f"🖼️ 이미지 처리 시작...")
                attach_cluster_images(
                    articles_csv=articles_csv_path,
                    summary_csv=summary_csv_path,
                    out_summary_csv=summary_csv_path,
                )
                wait_for_image_urls(summary_csv_path, timeout=10)
                print(f"✅ 이미지 처리 완료")

            else:
                print(f"⚠️ CSV 파일에 데이터가 없어 이미지 처리 스킵")
                print(f"   articles: {len(arts_test)}행, summary: {len(summ_test)}행")
    except Exception as e:
        print(f"❌ 이미지 처리 실패: {e}")

    # DB 처리
    if engine is not None:
        try:
            print(f"\n💾 DB 처리 시작...")

            if os.path.exists(articles_csv_path) and os.path.exists(summary_csv_path):
                arts_final = pd.read_csv(articles_csv_path)
                summ_final = pd.read_csv(summary_csv_path)
                summ_final["image_url"] = _normalize_image_url_series(summ_final["image_url"]) # 추가
                print(f"💾 최종 CSV 상태 - articles: {len(arts_final)}행, summary: {len(summ_final)}행")

                if len(summ_final) > 0:
                    print(f"📝 topic 테이블 업데이트 중...")
                    upsert_topic_images_from_summary(engine, summary_csv_path)
                    print(f"✅ topic 테이블 업데이트 완료")

                if len(arts_final) > 0:
                    print(f"📰 news 테이블 삽입 시작...")
                    insert_news_from_csv(engine, articles_csv_path, summary_csv_path)
                    print(f"✅ news 테이블 삽입 완료")
                else:
                    print(f"⚠️ articles CSV가 비어있어 news 삽입 스킵")

                # ★ 여기서 바로 정규화 실행
                normalize_is_new(engine)
                
        except Exception as e:
            print(f"❌ DB 처리 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ DB 엔진이 None이라 DB 저장 스킵")

from sqlalchemy import text
import pandas as pd

def upsert_topic_images_from_summary(engine, summary_csv_path: str):
    summ = pd.read_csv(summary_csv_path)
    if "cluster_id" not in summ.columns:
        print("[TOPIC_IMG] summary CSV에 cluster_id가 없습니다.")
        return
    if "image_url" not in summ.columns:
        print("[TOPIC_IMG] summary CSV에 image_url이 없습니다.")
        return

    # 기존 동작 유지: 정규화는 그대로
    summ["image_url"] = _normalize_image_url_series(summ["image_url"])

    # 같은 cluster_id가 여러 줄이면 마지막만 (원 코드 스타일 유지)
    topics_df = (summ[["cluster_id", "image_url"]]
                 .drop_duplicates(subset=["cluster_id"], keep="last"))

    print("어디가 안돼?2222")
    # UPDATE만 수행: 기존 row의 image_url이 비었을 때만 채움
    rows = []
    for _, r in topics_df.iterrows():
        if pd.isna(r["cluster_id"]):
            continue
        rows.append({
            "id": int(r["cluster_id"]),
            "img": (r.get("image_url") or "").strip()
        })

    if not rows:
        print("[TOPIC_IMG] 업데이트 대상 없음")
        return
    print("어디가 안돼333?")
    stmt = text("""
        UPDATE topic
           SET image_url = CASE
                             WHEN (image_url IS NULL OR image_url = '' OR image_url = 'nan')
                                  AND :img <> ''
                             THEN :img
                             ELSE image_url
                           END
         WHERE id = :id
    """)

    print("어디가 안돼?4444")
    updated = 0
    with engine.begin() as conn:
        for row in rows:
            res = conn.execute(stmt, row)
            updated += (res.rowcount or 0)

    print(f"🖼️ image_url 채움 완료: {updated}건 "
          f"(기존 값 있는 항목은 유지, 새 INSERT 없음)")

# def upsert_topic_images_from_summary(engine, summary_csv_path: str):
#     summ = pd.read_csv(summary_csv_path)
#     if "cluster_id" not in summ.columns:
#         print("[TOPIC_IMG] summary CSV에 cluster_id가 없습니다.")
#         return
#     if "image_url" not in summ.columns:
#         print("[TOPIC_IMG] summary CSV에 image_url이 없습니다.")
#         return
#     print("어디가 안돼?")

#     summ["image_url"] = _normalize_image_url_series(summ["image_url"]) # 추가

#     topics_df = (summ[["cluster_id", "topic_name", "ai_summary", "summary_time", "image_url"]]
#                  .rename(columns={"cluster_id": "id"})
#                  .drop_duplicates(subset=["id"]))

#     print("어디가 안돼?2222")
#     # 기존 topic id 로드
#     try:
#         print("어디가 안돼-------333?")
#         exist = pd.read_sql("SELECT id FROM topic", engine)
#         exist_ids = set(pd.to_numeric(exist["id"], errors="coerce").dropna().astype(int))
#     except Exception as e:
#         print(f"[TOPIC_IMG] 기존 topic 로드 실패: {e} → 최초 삽입일 수 있음")
#         exist_ids = set()

#     # 1) 신규 topic은 image_url 포함해서 INSERT
#     print("어디가 안돼?3333333")
#     new_topics = topics_df[~topics_df["id"].astype(int).isin(exist_ids)].copy()
#     if len(new_topics) > 0:
#         new_topics.to_sql("topic", engine, index=False, if_exists="append")
#         print(f"[TOPIC_IMG][INSERT] rows={len(new_topics)} "
#               f"sample={new_topics[['id', 'image_url']].head(3).to_dict(orient='records')}")
#     else:
#         print("[TOPIC_IMG][INSERT] 신규 없음")

#     # 2) 기존 topic은 image_url이 비어있을 때만 UPDATE
#     print("어디가 안돼?4444444")
#     upd = topics_df[topics_df["id"].astype(int).isin(exist_ids)].copy()
#     upd["image_url"] = upd["image_url"].fillna("").astype(str).str.strip()
#     upd = upd[upd["image_url"] != ""]

#     updated = 0
#     if len(upd) > 0:
#         with engine.begin() as conn:
#             for _, r in upd.iterrows():
#                 res = conn.execute(
#                     text("""
#                         UPDATE topic
#                            SET image_url = :img
#                          WHERE id = :id
#                            AND (image_url IS NULL OR image_url = '' OR image_url = 'nan')
#                     """),
#                     {"img": r["image_url"], "id": int(r["id"])}
#                 )
#                 updated += res.rowcount or 0
#         print(f"[TOPIC_IMG][UPDATE] tried={len(upd)} updated={updated}")
#     else:
#         print("[TOPIC_IMG][UPDATE] 업데이트할 이미지 없음")



def insert_news_from_csv(engine, articles_csv_path, summary_csv_path):
    print(f"\n🔄 insert_news_from_csv 시작...")
    print(f"   articles_csv: {articles_csv_path}")
    print(f"   summary_csv: {summary_csv_path}")

    # 파일 존재 및 크기 확인
    for path, name in [(articles_csv_path, "articles"), (summary_csv_path, "summary")]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"📁 {name} 파일: {size} bytes")

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"   총 {len(lines)}줄")
                    if len(lines) > 1:
                        print(f"   둘째 줄: {lines[1].strip()}")
            except Exception as e:
                print(f"   파일 읽기 실패: {e}")
        else:
            print(f"📁 {name} 파일: 존재하지 않음")
            return

    # CSV 로드
    try:
        arts = pd.read_csv(articles_csv_path)
        summ = pd.read_csv(summary_csv_path)
        print(f"📊 로드 완료 - arts: {len(arts)}행, summ: {len(summ)}행")

        if len(arts) == 0:
            print("⚠️ articles CSV가 비어있음 - 헤더만 있는 상태")
            return

        print(f"📋 arts 데이터 샘플:")
        for i, row in arts.head(2).iterrows():
            print(f"   [{i}] cluster_id: {row.get('cluster_id')}, title: {str(row.get('title', ''))[:50]}...")

    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    # 스키마 보강
    for col, default in {"news_summary": "", "is_new": 0, "is_third": 0}.items():
        if col not in arts.columns:
            arts[col] = default

    # # topic UPSERT
    # topics_df = summ.rename(columns={"cluster_id": "id"})[
    #     ["id", "topic_name", "ai_summary", "summary_time"]].drop_duplicates(subset=["id"])
    # try:
    #     existing = pd.read_sql("SELECT id FROM topic", engine)
    #     exist_ids = set(pd.to_numeric(existing["id"], errors="coerce").dropna().astype(int).tolist())
    #     print(f"📋 기존 topic 개수: {len(exist_ids)}")
    # except Exception as e:
    #     print(f"ℹ️ topic 테이블 조회 실패: {e}")
    #     exist_ids = set()

    # new_topics = topics_df[~topics_df["id"].astype(int).isin(exist_ids)].copy()
    # if len(new_topics) > 0:
    #     new_topics.to_sql("topic", engine, index=False, if_exists="append")
    #     print(f"✅ topic 삽입 완료: {len(new_topics)}개")
    
    # topic UPSERT
    topics_df = summ.rename(columns={"cluster_id": "id"})[
        ["id", "topic_name", "ai_summary", "summary_time"]
    ].drop_duplicates(subset=["id"])
    try:
        existing = pd.read_sql("SELECT id FROM topic", engine)
        exist_ids = set(pd.to_numeric(existing["id"], errors="coerce").dropna().astype(int).tolist())
        print(f"📋 기존 topic 개수: {len(exist_ids)}")
    except Exception as e:
        print(f"ℹ️ topic 테이블 조회 실패: {e}")
        exist_ids = set()

    # 1) 신규 INSERT
    new_topics = topics_df[~topics_df["id"].astype(int).isin(exist_ids)].copy()
    if len(new_topics) > 0:
        new_topics.to_sql("topic", engine, index=False, if_exists="append")
        print(f"✅ topic 삽입 완료: {len(new_topics)}개")

    # 2) ✅ 기존 UPDATE (교체)
    upd_topics = topics_df[topics_df["id"].astype(int).isin(exist_ids)].copy()
    if len(upd_topics) > 0:
        from sqlalchemy import text
        with engine.begin() as conn:
            total_upd = 0
            for _, r in upd_topics.iterrows():
                res = conn.execute(
                    text("""
                        UPDATE topic
                           SET topic_name   = :topic_name,
                               ai_summary   = :ai_summary,
                               summary_time = :summary_time
                         WHERE id = :id
                    """),
                    {
                        "id": int(r["id"]),
                        "topic_name": (r.get("topic_name") or ""),
                        "ai_summary": (r.get("ai_summary") or ""),
                        "summary_time": (r.get("summary_time") or datetime.now().strftime("%Y-%m-%d %H:%M")),
                    }
                )
                total_upd += (res.rowcount or 0)
        print(f"🟡 topic 업데이트 완료: {total_upd}건")


    # news 데이터 정리
    merged = arts.drop_duplicates(subset=["cluster_id", "title", "news_link"]).copy()
    merged["topic_id"] = merged["cluster_id"].astype(int)
    print(f"🔗 topic_id 목록: {merged['topic_id'].unique().tolist()}")

    # 유효한 topic_id만 유지
    try:
        existing = pd.read_sql("SELECT id FROM topic", engine)
        valid_topic_ids = set(pd.to_numeric(existing["id"], errors="coerce").dropna().astype(int).tolist())
        print(f"🎯 유효한 topic_id: {len(valid_topic_ids)}개")

        before_rows = len(merged)
        merged = merged[merged["topic_id"].isin(valid_topic_ids)].copy()
        print(f"🔍 FK 검증 후: {before_rows} → {len(merged)}행")

        if len(merged) == 0:
            print(f"❌ 모든 news가 FK 검증에서 제외됨!")
            print(f"   원본 topic_id: {arts['cluster_id'].unique().tolist()}")
            print(f"   DB의 topic_id: {list(valid_topic_ids)[:10]}")
            return

    except Exception as e:
        print(f"❌ topic_id 검증 실패: {e}")

    # 5) 중복 제거
    try:
        existing_news = pd.read_sql("SELECT topic_id,title,news_link FROM news", engine)
        print(f"📰 기존 news 개수: {len(existing_news)}")

        if len(existing_news) > 0:
            merged = merged.merge(existing_news, on=["topic_id", "title", "news_link"], how="left", indicator=True)
            merged = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        print(f"🆕 중복 제거 후 신규 news: {len(merged)}개")

    except Exception as e:
        print(f"ℹ️ news 중복 검사 실패: {e}")

    # 6) 최종 삽입 (✅ image_url 제외, ✅ news_summary 포함)
    if len(merged) == 0:
        print(f"ℹ️ 삽입할 news가 없음")
        return

    to_ins = merged[
        ["topic_id", "title", "news_link", "press", "publish_date", "news_summary", "is_new", "is_third"]].copy()
    print(f"💾 삽입 준비 완료: {len(to_ins)}행")

    try:
        to_ins.to_sql("news", engine, index=False, if_exists="append")
        print(f"✅ news 삽입 완료: {len(to_ins)}개")

        # 삽입 후 검증
        final_count = pd.read_sql("SELECT COUNT(*) as cnt FROM news", engine).iloc[0]['cnt']
        print(f"📊 최종 news 테이블 총 행수: {final_count}")

    except Exception as e:
        print(f"❌ news 삽입 실패: {e}")
        import traceback
        traceback.print_exc()


def retry_failed_clusters_from_json(df, failed_json_path="data/failed_responses.jsonl",
                                    summary_csv_path="data/commandr_summary.csv",
                                    articles_csv_path="data/commandr_articles.csv",
                                    cluster_col="tfidf_cluster_id"):
    """실패한 클러스터 재처리"""
    if not os.path.exists(failed_json_path):
        print("실패한 JSON 파일이 없습니다.")
        return

    api_key = get_cohere_api_key()
    failed_clusters = []

    # 실패한 클러스터 ID 읽기
    with open(failed_json_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            failed_clusters.append(record["cluster_id"])

    unique_clusters = list(set(failed_clusters))
    print(f"재처리 대상: {len(unique_clusters)}개 클러스터")

    for cluster_id in tqdm(unique_clusters, desc="재처리 중"):
        cluster_df = df[df[cluster_col] == cluster_id]
        if len(cluster_df) < 2:
            continue

        try:
            summary_result = make_commandr_summary_cohere(cluster_df, cluster_id, api_key)  # 수정
            save_commandr_output_to_csv(df, summary_result, cluster_id, summary_csv_path, articles_csv_path)
        except Exception as e:
            print(f"❌ 클러스터 {cluster_id} 재처리 실패: {e}")