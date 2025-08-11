# import os
# import pandas as pd
# import time
# import traceback
# from threading import Thread, Event
# from datetime import datetime
# from fastapi import FastAPI

# from crawler import run_crawling 
# from clustering import run_clustering

# FINAL_DF1000_PATH = "path/to/final_df1000.csv"



# app = FastAPI()

# # === 환경변수로 간격/지연 설정 ===
# INTERVAL_MIN = int(os.getenv("CRAWL_INTERVAL_MINUTES", "20"))   # 기본 20분마다 한 사이클
# INTERVAL_SEC = INTERVAL_MIN * 60
# CLUSTER_DELAY_SEC = int(os.getenv("CLUSTER_DELAY_SECONDS", "30"))  # 크롤링 후 군집 시작까지 잠깐 대기(파일 flush 대비)

# _stop = Event()
# _started = False  # 중복 시작 방지

# def worker():
#     # 시작 즉시 1회 사이클 (원치 않으면 이 두 줄 주석)
#     run_cycle()

#     # 이후 주기 반복
#     while not _stop.is_set():
#         if _stop.wait(INTERVAL_SEC):  # 종료 신호면 즉시 탈출
#             break
#         run_cycle()

# def run_cycle():
#     # 1) 크롤링
#     started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"\n===== [CRAWL START] {started_at} → run_crawling() =====")
#     try:
#         run_crawling()
#         print(f"===== [CRAWL END]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ✓ =====")
#     except SystemExit:
#         print("⚠️ run_crawling()에서 SystemExit → 다음 단계로 진행은 스킵")
#         return  # 크롤링이 정상 완료되지 않았으면 군집 생략
#     except Exception as e:
#         print("❌ 크롤링 예외:", e)
#         traceback.print_exc()
#         return  # 실패 시 군집 생략

#     # 2) (선택) 파일 저장 여유
#     if CLUSTER_DELAY_SEC > 0:
#         # 종료 신호가 오면 즉시 탈출
#         if _stop.wait(CLUSTER_DELAY_SEC):
#             return

#     # 3) 군집
#     started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"----- [CLUSTER START] {started_at} → run_clustering() -----")
#     try:
#         result = run_clustering()
#         print(f"----- [CLUSTER END]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ✓ {result} -----")
#     except SystemExit:
#         print("⚠️ run_clustering()에서 SystemExit → 다음 주기까지 대기")
#     except Exception as e:
#         print("❌ 군집 예외:", e)
#         traceback.print_exc()

# @app.on_event("startup")
# def start_worker():
#     global _started
#     if _started:
#         return
#     _started = True
#     Thread(target=worker, daemon=True).start()
#     print(f"⏱ 스케줄러 시작: {INTERVAL_MIN}분 간격, 군집 지연 {CLUSTER_DELAY_SEC}초")

# @app.on_event("shutdown")
# def stop_worker():
#     _stop.set()

# @app.get("/health")
# def health():
#     return {"ok": True}


# app.py
import os
import time
import traceback
from threading import Thread, Event
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from crawler import run_crawling
from clustering import run_clustering

BASE_DIR = Path(__file__).resolve().parent
FINAL_DF1000_PATH = BASE_DIR / "df1000_with_second.csv"  # df1000
INTERVAL_MIN = int(os.getenv("CRAWL_INTERVAL_MINUTES", "10"))
INTERVAL_SEC = INTERVAL_MIN * 20
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
