# attach_cluster_images_naver.py (debug-heavy)
import os
import pathlib
import re
import traceback
import uuid
from urllib.parse import urlsplit, unquote

import boto3
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# # ---------- 로깅 ----------
# LOG_DIR = pathlib.Path(__file__).parent / "logs"
# LOG_DIR.mkdir(exist_ok=True)
# LOG_FILE = LOG_DIR / "attach_image.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
# )
log = logging.getLogger("attach_image")

# ---------- 환경 ----------
BASE = pathlib.Path(__file__).parent
load_dotenv(dotenv_path=BASE / ".env")  # 파일 기준으로 .env 로드 (PyCharm WD 무관)
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET  = os.getenv("S3_BUCKET")
S3_PREFIX  = os.getenv("S3_UPLOAD_PREFIX", "news_images")
assert S3_BUCKET, "S3_BUCKET 값을 .env에 넣어주세요."

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

def safe_filename_from_url(img_url: str) -> str:
    base = os.path.basename(urlsplit(img_url).path)
    base = unquote(base)
    base = re.sub(r"[^0-9A-Za-z\-_.]", "_", base)
    return base or str(uuid.uuid4())

def _pick_from_srcset(val: str) -> str | None:
    if not val: return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    if not parts: return None
    return parts[-1].split()[0]  # "url 640w" -> "url"

def get_naver_image_url(news_url: str, timeout: int = 10) -> str | None:
    try:
        log.info(f"  [REQ] {news_url}")
        resp = requests.get(news_url, headers=HEADERS, timeout=timeout)
        log.info(f"  [RESP] status={resp.status_code}, len={len(resp.text)}")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        tag = soup.select_one("img#img1")
        cand = None
        if tag:
            log.info("  [ATTRS] img#img1 attrs=%s", dict(tag.attrs))
            # lazy-load 속성 우선 검사
            for attr in ("src", "currentSrc", "data-src", "data-lazysrc", "data-origin-src"):
                v = tag.get(attr)
                if v:
                    cand = v
                    break
            if not cand:
                cand = _pick_from_srcset(tag.get("srcset")) or _pick_from_srcset(tag.get("data-srcset"))

        # 메타 폴백
        if not cand:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                cand = og["content"]
        if not cand:
            tw = soup.find("meta", attrs={"name": "twitter:image"})
            if tw and tw.get("content"):
                cand = tw["content"]

        if not cand:
            log.warning("  [MISS] no usable image url")
            return None

        if cand.startswith("//"):
            cand = "https:" + cand
        if not cand.startswith("http"):
            log.warning("  [BAD_SRC] not http: %s", cand)
            return None
        log.info(f"  [FOUND] image={cand}")
        return cand
    except Exception as e:
        log.error("  [ERR] get_naver_image_url: %s\n%s", e, traceback.format_exc())
        return None

def upload_image_to_s3(img_url: str, cluster_id) -> str | None:
    try:
        log.debug(f"[S3_UPLOAD_START] cid={cluster_id} url={img_url}")
        r = requests.get(img_url, headers=HEADERS, stream=True, timeout=15)
        r.raise_for_status()
        key = f"{S3_PREFIX}/cluster_{cluster_id}/{safe_filename_from_url(img_url)}"
        content_type = r.headers.get("Content-Type") or "image/jpeg"
        s3.upload_fileobj(r.raw, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        log.info(f"[S3_UPLOAD_OK] cid={cluster_id} -> {s3_url}")
        return s3_url
    except Exception as e:
        log.exception(f"[S3_UPLOAD_FAIL] cid={cluster_id} url={img_url} err={e}")
        return None

def attach_cluster_images(
    articles_csv="commandr_articles.csv",
    summary_csv="commandr_summary.csv",
    out_summary_csv="commandr_summary.csv",
):
    # 파일 경로를 스크립트 기준으로 고정
    articles_csv = (BASE / articles_csv)
    summary_csv  = (BASE / summary_csv)
    out_summary_csv = (BASE / out_summary_csv)

    log.info("=== START attach_cluster_images ===")
    log.info("Base Dir: %s", BASE)
    log.info("Using CSV: articles=%s, summary(in)=%s, summary(out)=%s", articles_csv, summary_csv, out_summary_csv)
    log.info("AWS: bucket=%s, region=%s, prefix=%s", S3_BUCKET, AWS_REGION, S3_PREFIX)

    # 기사 CSV 로드
    if not articles_csv.exists():
        raise FileNotFoundError(f"기사 CSV 미존재: {articles_csv}")
    art = pd.read_csv(articles_csv)
    log.info("[ART] rows=%d, cols=%s", len(art), list(art.columns))
    for col in ["cluster_id", "news_link"]:
        if col not in art.columns:
            raise ValueError(f"{articles_csv.name}에 '{col}' 컬럼이 필요합니다.")

    # 요약 CSV 준비
    if summary_csv.exists():
        summ = pd.read_csv(summary_csv)
        log.info("[SUMM] rows=%d, cols=%s", len(summ), list(summ.columns))
        if "cluster_id" not in summ.columns:
            base = pd.DataFrame({"cluster_id": sorted(art["cluster_id"].unique())})
            summ = base.merge(summ, how="left", on="cluster_id")
        if "image_url" not in summ.columns:
            summ["image_url"] = None
    else:
        log.warning("[SUMM] 요약 CSV가 없어 기본 프레임 생성")
        summ = pd.DataFrame({"cluster_id": sorted(art["cluster_id"].unique())})
        summ["image_url"] = None







    # 1) 스킵할 클러스터 계산 부분 교체
    def is_empty(v):
        if v is None:
            return True
        s = str(v).strip()
        return s == "" or s.lower() in {"nan", "none", "null", "na", "nat"}

    filled_mask = ~summ["image_url"].apply(lambda v: is_empty(v))
    processed_or_has_image = set(summ.loc[filled_mask, "cluster_id"].tolist())
    log.info("[SKIP_SET] %d clusters already have image_url", len(processed_or_has_image))

    # 클러스터별 처리
    total_clusters = art["cluster_id"].nunique()
    done, skipped, failed = 0, 0, 0
    cid_to_s3 = {}

    for idx, (cid, grp) in enumerate(art.groupby("cluster_id"), start=1):
        log.info("--- CLUSTER %s (%d/%d) ---", cid, idx, total_clusters)
        if cid in processed_or_has_image:
            log.info("[SKIP] cluster %s: image_url already exists", cid)
            skipped += 1
            cid_to_s3[cid] = None
            continue

        s3_url = None
        for ridx, row in grp.reset_index(drop=True).iterrows():
            link = str(row["news_link"])
            log.debug(f"[TRY_IMG] cid={cid} idx={ridx + 1}/{len(grp)} link={link}")
            img_url = get_naver_image_url(link)
            log.debug(f"[FOUND_IMG_CAND] cid={cid} img={img_url}")
            if not img_url:
                log.info("  [NO_IMG] img#img1 not found: %s", link)
                continue
            s3_url = upload_image_to_s3(img_url, cluster_id=cid)
            if s3_url:
                log.info(f"[CLUSTER_IMG_CHOSEN] cid={cid} s3={s3_url}")
            else:
                log.warning(f"[CLUSTER_IMG_MISS] cid={cid}: 모든 기사에서 이미지 확보 실패")

        if s3_url:
            done += 1
        else:
            failed += 1
            log.warning("  [CLUSTER_FAIL] cid=%s: 모든 기사에서 이미지 확보 실패", cid)
        cid_to_s3[cid] = s3_url

    # image_url 업데이트(비어 있을 때만)
    # 2) 업데이트 시에도 같은 기준 사용
    def choose_new(cid, old):
        if is_empty(old):
            return cid_to_s3.get(cid) or old
        return old

    before_filled = (summ["image_url"].astype(str).str.strip() != "").sum()
    summ["image_url"] = [choose_new(cid, old) for cid, old in zip(summ["cluster_id"], summ["image_url"])]
    after_filled = (summ["image_url"].astype(str).str.strip() != "").sum()

    # 저장
    summ.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    log.info("[DONE] saved -> %s", out_summary_csv)
    log.info("[STATS] picked=%d, skipped=%d, failed=%d, filled(before=%d -> after=%d)",done, skipped, failed, before_filled, after_filled)

if __name__ == "__main__":
    attach_cluster_images()