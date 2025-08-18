import os
import random
import re
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, WebDriverException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# 크롤링 파일 상단에 추가
from preprocessing import (
    clean_content,
    convert_hanja_to_korean, hanja_to_korean,
    convert_korean_datetime,
    get_nouns_with_konlpy,
)

# 여기서 크롤링 수집 개수 지정
MIN_LINK_COUNT = 3

def remove_brackets(text):
    return re.sub(r'\[.*?\]', '', text).strip()


def delete_10_data():
    file_path = "crawling_total.csv"
    df = pd.read_csv(file_path)

    # 맨 위 10개 삭제
    df = df.iloc[5:].reset_index(drop=True)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"🗑 통합 데이터 10개 삭제 완료 → 남은 행 수: {len(df)}")

    file_path = "crawling_latest.csv"
    df = pd.read_csv(file_path)

    # 맨 위 10개 삭제
    df = df.iloc[5:].reset_index(drop=True)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"🗑 최신 데이터 10개 삭제 완료 → 남은 행 수: {len(df)}")

def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,1696")
    options.add_argument("--lang=ko-KR")
    options.add_experimental_option("prefs", {"intl.accept_languages": "ko,ko_KR"})

    # 환경변수 있을 때만 지정 (없으면 Selenium이 자동 탐색)
    chrome_bin = os.getenv("CHROME_BIN")
    if chrome_bin:
        options.binary_location = chrome_bin

    chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
    if chromedriver_path and os.path.exists(chromedriver_path):
        service = Service(chromedriver_path)
        return webdriver.Chrome(service=service, options=options)
    else:
        # 경로 지정 안 하면 Selenium Manager가 드라이버를 자동으로 받음/사용
        return webdriver.Chrome(options=options)

# def create_driver():
#     options = Options()
#     options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
#     options.add_argument("--headless=new")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--window-size=1280,1696")
#     options.add_argument("--lang=ko-KR")
#     options.add_experimental_option("prefs", {"intl.accept_languages": "ko,ko_KR"})

#     service = Service(os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
#     return webdriver.Chrome(service=service, options=options)

# 전처리
def preprocess_before_save(df: pd.DataFrame) -> pd.DataFrame:
    """한자→한글, 텍스트 클리닝, 날짜변환, 명사리스트 추출까지 한 번에."""
    df = df.copy()

    # 크롤러 컬럼명 → 전처리 표준 컬럼명으로 매핑
    # - summary 필요하면 news_summary로부터 생성
    # - datetime 필요하면 publish_date로부터 생성
    if "summary" not in df.columns and "news_summary" in df.columns:
        df["summary"] = df["news_summary"]
    if "datetime" not in df.columns and "publish_date" in df.columns:
        df["datetime"] = df["publish_date"]

    # 한자 → 한글 + 클리닝
    for col in ["title", "summary"]:
        if col in df.columns:
            df[col] = (
                df[col].fillna("").astype(str)
                .apply(lambda x: convert_hanja_to_korean(x, hanja_to_korean))
                .apply(clean_content)
            )

    # 날짜 문자열 → datetime 변환 (publish_date) 수정!!!!!!!!!!!!!!!
    if "datetime" not in df.columns and "publish_date" in df.columns:
        df["datetime"] = df["publish_date"]  # 기존 로직 유지 (과거 호환)

    parsed = None
    if "datetime" in df.columns:
        parsed = df["datetime"].apply(convert_korean_datetime)

    # ✅ 백업 경로: datetime_sort(ISO) 있으면 거기로 채우기
    if "datetime_sort" in df.columns:
        iso = pd.to_datetime(df["datetime_sort"], errors="coerce")
        # 우선순위: parsed(성공) > iso
        df["publish_date"] = pd.to_datetime(parsed, errors="coerce")
        df.loc[df["publish_date"].isna(), "publish_date"] = iso
    else:
        df["publish_date"] = pd.to_datetime(parsed, errors="coerce")

    # 문자열로 정규화 (DB 저장용)
    df["publish_date"] = df["publish_date"].dt.strftime("%Y-%m-%d %H:%M")


    # 명사 리스트 추출용 텍스트 구성(title + summary)
    df["__text_for_nouns"] = (
        df.get("title", "").astype(str) + " " + df.get("summary", "").astype(str)
    ).str.strip()

    # konlpy 명사 추출 (konlpy_nouns 생성)
    df = get_nouns_with_konlpy(df, "__text_for_nouns")

    # 임시 컬럼 제거
    df.drop(columns=["__text_for_nouns"], inplace=True)

    return df

def load_previous_data(collected_path, total_path):
    # delete_10_data()
    try:
        if os.path.exists(collected_path):
            df = pd.read_csv(collected_path)
            if df.empty or 'news_link' not in df.columns:
                print("collected_links.csv에 유효한 데이터가 없음")
                return pd.DataFrame(), None
            last_seen_news_link = df.iloc[0]['news_link']
            print(f"collected_links.csv 기준 → 마지막 수집 기사: {last_seen_news_link}")
            return df, last_seen_news_link

        elif os.path.exists(total_path):
            print("collected_links.csv 없음 → crawling_total.csv 기준으로 최신 기사 추정")
            df = pd.read_csv(total_path)
            if df.empty or 'news_link' not in df.columns:
                print("crawling_total.csv에 유효한 데이터가 없음")
                return pd.DataFrame(), None
            last_seen_news_link = df.iloc[0]['news_link']
            print(f"crawling_total.csv 기준 → 마지막 수집 기사: {last_seen_news_link}")
            return pd.DataFrame(), last_seen_news_link

        else:
            print("기존 수집 데이터 없음 → 완전 초기 실행")
            return pd.DataFrame(), None

    except Exception as e:
        print(f"기존 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame(), None


def collect_article_links(driver, last_seen_news_link):
    article_links = []
    found_last_seen = False

    while not found_last_seen:
        time.sleep(1)
        try:
            containers = driver.find_elements(By.CSS_SELECTOR, 'div.section_article._TEMPLATE[data-template-id="SECTION_ARTICLE_LIST"]')
        except WebDriverException as e:
            print(f"기사 리스트 불러오기 실패: {e}")
            break

        for container in containers:
            try:
                articles = container.find_elements(By.CSS_SELECTOR, 'a.sa_text_title')
            except Exception:
                continue

            for tag in articles:
                try:
                    href = tag.get_attribute('href')
                    title = tag.text.strip()
                except Exception:
                    continue

                if not href or href in article_links:
                    continue
                if href == last_seen_news_link:
                    found_last_seen = True
                    print(f"\n마지막 본 기사 도달 → {href}")
                    break
                article_links.append(href)
                if len(article_links) % 100 == 0:
                    print(f"[{len(article_links)}] {title} → {href}")

                MAX_TEST_LINKS = 200  # 테스트 시 수집할 최대 링크 수

                # ✅ 테스트 모드: 일정 개수 넘으면 수집 중단
                if MAX_TEST_LINKS and len(article_links) >= MAX_TEST_LINKS:
                    print(f"✅ 테스트를 위해 {MAX_TEST_LINKS}개까지만 수집 후 종료")
                    found_last_seen = True
                    break
            if found_last_seen:
                break

        try:
            more_btn = driver.find_element(By.CSS_SELECTOR, 'a.section_more_inner._CONTENT_LIST_LOAD_MORE_BUTTON')
            more_btn.click()
            time.sleep(random.uniform(2.0, 3.0))
        except NoSuchElementException:
            print("더보기 버튼 없음 → 종료")
            break

    print(f"\n총 {len(article_links)}개 새 기사 링크 확보 완료")
    return article_links


# 이 함수 안에서
def update_and_check_links(driver, last_seen_news_link, base_dir):
    collected_path = os.path.join(base_dir, "collected_links.csv")

    # 기존 링크 읽기
    prev_links = []
    if os.path.exists(collected_path):
        prev_links = pd.read_csv(collected_path)["news_link"].tolist()

    # 새로운 링크 수집 (항상 최신 → 과거 순으로 수집됨)
    new_links = collect_article_links(driver, last_seen_news_link)

    # 병합 순서 보존: 새 링크 먼저, 그 뒤 기존 링크
    combined_links = new_links + [link for link in prev_links if link not in new_links]

    print(f"현재까지 누적 기사 수: {len(combined_links)}")

    pd.DataFrame({"news_link": combined_links}).to_csv(collected_path, index=False)

    if len(combined_links) < MIN_LINK_COUNT: # 여기서 개수 조정하기!
        print(f"아직 {MIN_LINK_COUNT}개 미만 → 크롤링 보류")
        driver.quit()
        sys.exit()

    return combined_links


def crawl_articles(driver, article_links):
    articles = []
    for idx, news_link in enumerate(article_links, 1):
        try:
            driver.get(news_link)
            time.sleep(1.5)

            try:
                summary_btn = driver.find_element(By.CSS_SELECTOR, 'a.media_end_head_autosummary_button')
                summary_btn.click()
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div._SUMMARY_CONTENT_BODY'))
                )
                time.sleep(0.5)
            except (NoSuchElementException, TimeoutException):
                pass

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            raw_title = soup.select_one('h2.media_end_head_headline')
            datetime = soup.select_one('span.media_end_head_info_datestamp_time')
            title = remove_brackets(raw_title.get_text(strip=True))
            newsroom = soup.select_one('a.media_end_head_top_logo img')
            newsroom_text = newsroom.get("alt") if newsroom else ""
            datetime_text = datetime.get_text(strip=True)
            datetime_raw = datetime.get("data-date-time")


            #print(f"[{idx}] DEBUG title={raw_title}, newsroom={newsroom}, datetime={datetime}")

            if not raw_title or not newsroom or not datetime:
                print(f"[{idx}] 필수 정보 누락 → {news_link}")
                continue

            summary = ""
            summary_tag = soup.select_one('div._SUMMARY_CONTENT_BODY')
            if summary_tag:
                strong = summary_tag.find('strong')
                # if strong:
                #     strong.extract()
                summary = remove_brackets(summary_tag.get_text(separator=' ', strip=True))

            articles.append({
                'title': title,
                'press': newsroom_text,
                'news_summary': summary,
                'publish_date': datetime_text,
                'datetime_sort': datetime_raw,
                'news_link': news_link,
            })
        except Exception as e:
            print(f"❌ [{idx}] 기사 크롤링 실패: {e} → {news_link}")
            continue

    #print(f"✅ 총 수집된 기사 수: {len(articles)}")

    df = pd.DataFrame(articles)

    # 요약 비어있는 데이터 제거
    before_drop = len(df)
    df = df[df["news_summary"].notnull() & df["news_summary"].str.strip().ne("")]
    after_drop = len(df)
    print(f"🧹 news_summary 기준 필터링 → {before_drop} → {after_drop}개")

    # publish_date는 그대로 놔두고, 정렬용 datetime_sort만 파싱에 사용
    df["datetime_sort"] = pd.to_datetime(df["datetime_sort"], errors="coerce")

    parsed_valid = df["datetime_sort"].notnull().sum()
    if parsed_valid < len(df):
        print("⚠️ datetime 파싱 실패한 예시들:")
        print(df[df["datetime_sort"].isnull()]["publish_date"].tolist())

    print(f"🕒 datetime 파싱 성공: {parsed_valid} / {len(df)}개")

    # 정렬 + 파싱 실패 제거
    df = df.sort_values("datetime_sort", ascending=False)
    df = df.dropna(subset=["datetime_sort"])

    # 결과 로그
    print(f"✅ 최종 정렬 및 필터링 완료 → {len(df)}개")

    # 필요한 컬럼만 반환
    sorted_articles = df.drop(columns=["datetime_sort"]).to_dict(orient="records")
    return sorted_articles


def save_and_merge(articles, base_dir):
    if not articles:
        print("새로운 기사 없음 → 저장 생략")
        return

    latest_path = os.path.join(base_dir, "crawling_latest.csv")
    total_path = os.path.join(base_dir, "crawling_total.csv")

    try:
        new_df = pd.DataFrame(articles)
        new_df = preprocess_before_save(new_df)
        new_df.to_csv(latest_path, index=False, encoding='utf-8-sig')
        print(f"crawling_latest.csv 저장 완료 ({len(new_df)}개)")
    except Exception as e:
        print(f"crawling_latest 저장 실패: {e}")
        return

    try:
        if os.path.exists(total_path):
            total_df = pd.read_csv(total_path)
        else:
            total_df = pd.DataFrame()
    except Exception as e:
        print(f"crawling_total 읽기 실패: {e}")
        total_df = pd.DataFrame()

    try:
        updated_df = pd.concat([new_df, total_df], ignore_index=True)
        updated_df.drop_duplicates(subset='news_link', inplace=True)
        updated_df.to_csv(total_path, index=False, encoding='utf-8-sig')
        print(f"crawling_total.csv 누적 저장 완료 (총 {len(updated_df)}개)")
    except Exception as e:
        print(f"누적 저장 실패: {e}")


def run_crawling():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    collected_path = os.path.join(BASE_DIR, "collected_links.csv")
    total_path = os.path.join(BASE_DIR, "crawling_total.csv")

    _, last_seen_news_link = load_previous_data(collected_path, total_path)

    try:
        driver = create_driver()
        driver.get("https://news.naver.com/section/101")
    except Exception as e:
        print(f"크롬 드라이버 실행 실패: {e}")
        sys.exit()

    try:
        article_links = update_and_check_links(driver, last_seen_news_link, BASE_DIR)
        articles = crawl_articles(driver, article_links)
        save_and_merge(articles, BASE_DIR)
    finally:
        driver.quit()

    # 링크 초기화
    try:
        if os.path.exists(collected_path):
            os.remove(collected_path)
            print("누적 링크 초기화 완료")
    except Exception as e:
        print(f"링크 초기화 실패: {e}")


# 실행
if __name__ == "__main__":
    run_crawling()