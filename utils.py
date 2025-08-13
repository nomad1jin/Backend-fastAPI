import os

from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_cohere_api_key():
    return os.getenv("COHERE_API_KEY")

def log_failed_cluster(cluster_id: int):
    with open("logs/failed_clusters.txt", "a", encoding="utf-8") as f:
        f.write(f"{cluster_id}\n")


def get_mysql_engine():
    load_dotenv()  # .env 파일 로드

    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    host = os.getenv("MYSQL_HOST")
    port = os.getenv("MYSQL_PORT", 3306)
    db = os.getenv("MYSQL_DB")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url)
    return engine