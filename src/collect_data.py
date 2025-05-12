# src/collect_data.py
import os
import requests
import pandas as pd

def collect_roadkill_data(years, service_key, save_path=None):
    """
    지정한 연도별로 공공데이터 포털 API를 통해 로드킬 데이터를 수집합니다.

    Args:
        years (list): 수집할 연도 리스트
        service_key (str): 공공데이터포털 API 인증키
        save_path (str, optional): 저장할 CSV 경로. 지정하지 않으면 저장하지 않음

    Returns:
        pd.DataFrame: 통합된 로드킬 데이터프레임
    """
    base_urls = {
        '2019': 'https://api.odcloud.kr/api/15045544/v1/uddi:64ae17e6-d3b2-46d1-be8d-9e8397aa70df',
        '2020': 'https://api.odcloud.kr/api/15045544/v1/uddi:7f1ff5b3-01a4-4d16-9111-c17fa3f34c14',
        '2021': 'https://api.odcloud.kr/api/15045544/v1/uddi:574ad30d-5e70-4ffb-a3e7-debdeb47925b',
        '2022': 'https://api.odcloud.kr/api/15045544/v1/uddi:362b14d7-6360-4070-8f7b-a2261a8a3e0d',
        '2023': 'https://api.odcloud.kr/api/15045544/v1/uddi:36c7c494-4896-4a76-bc4c-f9133762c595'
    }

    all_dataframes = []

    for year in years:
        print(f"\n📥 {year}년도 데이터 요청 중...")
        api_url = base_urls[str(year)]
        params = {
            'serviceKey': service_key,
            'page': 1,
            'perPage': 1000
        }
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            items = response.json().get('data', [])
            df = pd.DataFrame(items)
            df['년도'] = year
            all_dataframes.append(df)
            print(f"✅ {year}년 데이터 {len(df)}건 수집 완료")
        else:
            print(f"❌ {year}년 데이터 요청 실패: {response.status_code}")
            print(response.text)

    df_all = pd.concat(all_dataframes, ignore_index=True)

    if save_path:
        df_all.to_csv(save_path, index=False)
        print(f"📦 데이터 저장 완료 → {save_path}")

    return df_all


if __name__ == "__main__":
    SERVICE_KEY = os.getenv("ROADKILL_API_KEY", "your_api_key")
    df_all = collect_roadkill_data(
        years=[2019, 2020, 2021, 2022, 2023],
        service_key=SERVICE_KEY,
        save_path="../data/raw/roadkill_2019_2023.csv"
    )
