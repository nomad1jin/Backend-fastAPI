# app.py
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Thread, Event

from fastapi import FastAPI

from clustering import run_clustering
from crawler import run_crawling
from notification_ping import send_ping

BASE_DIR = Path(__file__).resolve().parent
FINAL_DF1000_PATH = BASE_DIR / "df1000_result0813.csv"  # df1000
INTERVAL_MIN = int(os.getenv("CRAWL_INTERVAL_MINUTES", "5"))
INTERVAL_SEC = INTERVAL_MIN * 60
CLUSTER_DELAY_SEC = int(os.getenv("CLUSTER_DELAY_SECONDS", "30"))

_stop = Event()
_started = False
_thread: Thread | None = None

def run_cycle():
    # 1) 크롤링
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== [CRAWL START] {started_at} → run_crawling() =====")
    try:
        run_crawling()
        print(f"===== [CRAWL END]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ✓ =====")
    except SystemExit:
        print("⚠️ run_crawling()에서 SystemExit → 이번 사이클 군집 생략")
        return
    except Exception as e:
        print("❌ 크롤링 예외:", e)
        traceback.print_exc()
        return

    # 2) 파일 저장 여유
    if CLUSTER_DELAY_SEC > 0 and _stop.wait(CLUSTER_DELAY_SEC):
        return

    # 3) 군집
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"----- [CLUSTER START] {started_at} → run_clustering() -----")
    try:
        result = run_clustering(FINAL_DF1000_PATH)

        # api 호출
        try:
            ping = send_ping({
                "source": "python-pipeline",
                "batchId": os.getenv("BATCH_ID", "manual"),
                "finishedAt": int(time.time() * 1000)
            })
            print("PING OK:", ping)
        except Exception as e:
            print("⚠️ ping 전송 실패:", e)

        print(f"----- [CLUSTER END]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ✓ {result} -----")
    except SystemExit:
        print("⚠️ run_clustering()에서 SystemExit → 다음 주기까지 대기")
    except Exception as e:
        print("❌ 군집 예외:", e)
        traceback.print_exc()

def worker():
    # 시작 즉시 1회 실행 (원치 않으면 주석)
    run_cycle()
    while not _stop.is_set():
        if _stop.wait(INTERVAL_SEC):  # 종료 신호면 탈출
            break
        run_cycle()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _started, _thread
    if not _started:
        _started = True
        _thread = Thread(target=worker, daemon=True)
        _thread.start()
        print(f"⏱ 스케줄러 시작: {INTERVAL_MIN}분 간격, 군집 지연 {CLUSTER_DELAY_SEC}초")
    # 앱 실행 구간
    yield
    # 종료 처리
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=5)

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}