import time
import pandas as pd
import os

def _normalize_image_url_series(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
             .replace({"nan": "", "NaN": "", "None": ""}))

def wait_for_image_urls(summary_csv_path: str, wait_ids=None,
                        timeout: float = 5.0, interval: float = 0.2) -> pd.DataFrame:
    """
    attach_cluster_images 실행 직후, summary CSV에서 image_url이 비어있지 않도록 최대 timeout까지 재시도.
    - wait_ids: 특정 cluster_id들에 대해 image_url이 비어있지 않은지 확인(1개라도 비면 재시도)
    - 없으면 전체 non-empty 비율이 이전보다 늘어나는 방향으로 안정화될 때까지 대기
    """
    start = time.time()
    last_nonempty = -1

    while True:
        try:
            df = pd.read_csv(summary_csv_path)
        except Exception:
            if time.time() - start > timeout:
                print(f"[WAIT_IMG] CSV 로드 실패(타임아웃) path={summary_csv_path}")
                raise
            time.sleep(interval)
            continue

        if "cluster_id" not in df.columns or "image_url" not in df.columns:
            if time.time() - start > timeout:
                print(f"[WAIT_IMG] 필요한 컬럼 없음(타임아웃) path={summary_csv_path} cols={df.columns}")
                return df
            time.sleep(interval)
            continue

        df["image_url"] = _normalize_image_url_series(df["image_url"])

        if wait_ids:
            sub = df[df["cluster_id"].astype(int).isin([int(x) for x in wait_ids])]
            missing = sub["image_url"].eq("").sum()
            if missing == 0 and len(sub) > 0:
                return df
        else:
            nonempty = (df["image_url"] != "").sum()
            if nonempty >= last_nonempty and nonempty > 0:
                # 한 번 더 읽어서 안정화(쓰기 플러시/파일 시스템 지연 대비)
                time.sleep(interval)
                df2 = pd.read_csv(summary_csv_path)
                if "image_url" in df2.columns:
                    df2["image_url"] = _normalize_image_url_series(df2["image_url"])
                    if (df2["image_url"] != "").sum() >= nonempty:
                        return df2
                return df

            last_nonempty = nonempty

        if time.time() - start > timeout:
            print(f"[WAIT_IMG] 타임아웃 → 일부 image_url 비어있을 수 있음 path={summary_csv_path}")
            return df

        time.sleep(interval)