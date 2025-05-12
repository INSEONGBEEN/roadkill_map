import pandas as pd

def clean_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    불필요한 컬럼 제거 및 숫자형 컬럼 변환
    """
    drop_cols = ['반기', '사고율', '노선코드', '방 향', '방향']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # 발생건수, 위도, 경도 수치형 변환
    df['발생건수'] = pd.to_numeric(df['발생건수'], errors='coerce')
    df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
    df['경도'] = pd.to_numeric(df['경도'], errors='coerce')

    return df

def classify_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    발생건수 기준으로 위험 등급 분류 컬럼 추가
    """
    def classify(val):
        if val <= 3:
            return '저위험'
        elif val <= 7:
            return '중위험'
        else:
            return '고위험'

    df['위험등급'] = df['발생건수'].apply(classify)
    df['고위험_이진'] = df['위험등급'].apply(lambda x: 1 if x == '고위험' else 0)
    return df

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    위도, 경도, 발생건수 모두 결측인 행 제거
    """
    return df.dropna(subset=['위도', '경도', '발생건수'])

def save_processed_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def preprocess_main(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    df = clean_and_convert_types(df)
    df = drop_empty_rows(df)
    df = classify_risk_level(df)
    if save_path:
        save_processed_csv(df, save_path)
    return df
