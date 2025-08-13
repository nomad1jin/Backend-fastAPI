import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 설정
BASE_URL = os.getenv("SPRING_API_BASE", "https://api.newsintelligent.site")
PING_PATH = "/api/notification/ping"
API_URL   = BASE_URL.rstrip("/") + PING_PATH

TIMEOUT   = (3.0, 10.0)  # (connect, read) seconds
VERIFY_SSL = True

# 세션
    s = requests.Session()
    retries = Retry(
        total=5,                # 총 재시도 횟수
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

session = _build_session()
log = logging.getLogger("ping")


# 파이프라인 마지막에 호출
def send_ping(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        resp = session.post(
            API_URL,
            headers=headers,
            timeout=TIMEOUT,
            verify=VERIFY_SSL,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"[PING] 요청 실패(네트워크/SSL): {e}") from e

    # 2xx -> 성공
    if not (200 <= resp.status_code < 300):
        snippet = resp.text[:300]
        raise RuntimeError(f"[PING] HTTP {resp.status_code} 응답: {snippet}")
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}

# 테스트
if __name__ == "__main__":
    result = send_ping({
        "source": "python-pipeline",
        "batchId": os.getenv("BATCH_ID", "manual"),
        "finishedAt": int(time.time() * 1000)
    })
    print("PING OK:", result)