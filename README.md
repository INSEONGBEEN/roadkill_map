# roadkill_map
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tensorflow as tf
import torch
import folium

print("✅ 라이브러리 설치 완료")

# 공공데이터포털 api 연동
import requests
import pandas as pd
# 데이터 수집
# ▶️ 연도별 API URL 리스트
apis = {
    '2019': 'https://api.odcloud.kr/api/15045544/v1/uddi:64ae17e6-d3b2-46d1-be8d-9e8397aa70df',
    '2020': 'https://api.odcloud.kr/api/15045544/v1/uddi:7f1ff5b3-01a4-4d16-9111-c17fa3f34c14',
    '2021': 'https://api.odcloud.kr/api/15045544/v1/uddi:574ad30d-5e70-4ffb-a3e7-debdeb47925b',
    '2022': 'https://api.odcloud.kr/api/15045544/v1/uddi:362b14d7-6360-4070-8f7b-a2261a8a3e0d',
    '2023': 'https://api.odcloud.kr/api/15045544/v1/uddi:36c7c494-4896-4a76-bc4c-f9133762c595'  # 2023용 URL 넣기
}

# ▶️ 인증키 (디코딩된 상태로!)
SERVICE_KEY = 'your_api_key'

# ▶️ 전체 데이터 저장용 리스트
all_dataframes = []

# ▶️ 연도별로 요청 반복
for year, api_url in apis.items():
    print(f'📥 {year}년도 데이터 요청 중...')
    params = {
        'serviceKey': SERVICE_KEY,
        'page': 1,
        'perPage': 1000  # 필요시 페이지네이션 추가 가능
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        items = response.json()['data']
        df = pd.DataFrame(items)
        df['년도'] = year  # ✅ 연도 컬럼 추가
        all_dataframes.append(df)
        print(f'✅ {year}년도 데이터 {len(df)}건 완료')
    else:
        print(f'❌ {year}년도 요청 실패: {response.status_code}')
        print(response.text)

# ▶️ 하나로 합치기
df_all = pd.concat(all_dataframes, ignore_index=True)

# ▶️ 결과 확인
print(df_all.shape)
print(df_all.head())

# ▶️ CSV로 저장 (선택)
df_all.to_csv('roadkill_2019_2023.csv', index=False)

# 데이터 전처리
# 년도별 데이터 병합시 불필요한 컬럼 삭제
df_all = df_all.drop(['반기', '사고율', '노선코드', '방 향','방향'], axis=1)
#년도 664개 데이터 삽입 이건 년도별 데이터 합치는 과정에서 일부 빈 데이터나 결측치만 있는 행이 들어갔을 가능성
#결측치만 있는 행 찾기
empty_rows = df[df.isnull().sum(axis=1) == (len(df.columns)-1)]  # 년도만 있고 나머지 전부 결측
print(empty_rows)
empty_rows = df[df['발생건수'].isnull() & df['경도'].isnull() & df['위도'].isnull()]
print(empty_rows)

# 머신러닝/시각화를 위해 object 타입 -> 수치형 변환
df_all['발생건수'] = pd.to_numeric(df_all['발생건수'], errors='coerce')
df_all['위도'] = pd.to_numeric(df_all['위도'], errors='coerce')
df_all['경도'] = pd.to_numeric(df_all['경도'], errors='coerce')

# EDA 부분
# 로드킬 발생건수 분포(히스토그램)
plt.figure(figsize=(8, 5))
df_all['발생건수'].hist(bins=15)
plt.title('로드킬 발생건수 분포')
plt.xlabel('발생건수')
plt.ylabel('빈도')
plt.show()

# 연도별 로드킬 발생건수 합계
df_all.groupby('년도')['발생건수'].sum().plot(kind='bar', figsize=(8, 5))
plt.title('연도별 로드킬 발생건수 합계')
plt.xlabel('년도')
plt.ylabel('총 발생건수')
plt.show()

# 본부별 로드킬 발생건수 합계
df_all.groupby('본부명')['발생건수'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))
plt.title('본부별 로드킬 발생건수')
plt.ylabel('총 발생건수')
plt.xlabel('본부명')
plt.show()

# 노선별 상위 10개 다발구간
df_all.groupby('노선명')['발생건수'].sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10, 5))
plt.title('노선별 상위 10개 로드킬 다발구간')
plt.ylabel('총 발생건수')
plt.xlabel('노선명')
plt.show()

# 위경도 산점도(로드킬 발생 위치)
plt.figure(figsize=(8, 8))
plt.scatter(df_all['경도'], df_all['위도'], s=df_all['발생건수']*5, alpha=0.5)
plt.title('로드킬 발생 위치 (경도 vs 위도)')
plt.xlabel('경도')
plt.ylabel('위도')
plt.show()

# 머신러닝 랜덤포레스트 채택 >> 설명력이 6% 밖에 안됨.. > 보완 필요

# 독립,종속변수 설정
feature_cols = ['위도', '경도']
X = df_all[feature_cols]
y = df_all['발생건수']
# 학습 테스트 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 회귀
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.3f}')
print(f'R2 Score: {r2:.3f}')

# 일단 랜덤포레스트 분류 적용 > 모델이 정확하지않음... > 이진분류로 선택함
# 발생건수를 기준으로 저위험 / 중위험 / 고위험 3개 클래스로 분류

# 타겟 변수 만들기
def classify_risk(val):
    if val <= 3:
        return '저위험'
    elif val <= 7:
        return '중위험'
    else:
        return '고위험'

df_all['위험등급'] = df_all['발생건수'].apply(classify_risk)

X = df_all[['위도', '경도']]
y = df_all['위험등급']

#학습 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 랜덤포레스트 분류기
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)

# 평가
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 다중분류 -> 이진분류 -> 이제 좀 정확해짐 ROC AUC : 0.942
# 이진 타겟 만들기 (고위험=1, 나머지=0)
df_all['고위험_이진'] = df_all['위험등급'].apply(lambda x: 1 if x == '고위험' else 0)
X = df_all[['위도', '경도']]
y = df_all['고위험_이진']

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 다시 train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

#랜덤포레스트 이진 분류기
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)

# 평가
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC AUC 스코어
from sklearn.metrics import roc_auc_score

y_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f'ROC AUC: {auc:.3f}')

# 시각화 하면 임의 지점에 대한 로드킬 발생 위험 확률에 대해 나타냄
# 신규 구간 예측

import folium
import pandas as pd

# 신규 포인트 리스트 (위도/경도)
new_points = [
    {'name': '중부선', 'lat': 37.5465, 'lon': 127.2003},
    {'name': '경부선', 'lat': 36.3504, 'lon': 127.3845},
    {'name': '중앙선', 'lat': 36.8725, 'lon': 128.5507},
    {'name': '서해안선', 'lat': 36.7763, 'lon': 126.4504}
]

# 지도 초기화
m = folium.Map(location=[36.5, 127.8], zoom_start=7)

# 각 신규 구간에 대해 예측 + 마커 표시
for point in new_points:
    X_new = pd.DataFrame([[point['lat'], point['lon']]], columns=['위도', '경도'])
    pred = clf.predict(X_new)[0]
    proba = clf.predict_proba(X_new)[0][1]
    risk = '고위험' if pred == 1 else '저위험'
    
    # 색상 지정
    color = 'red' if pred == 1 else 'green'
    
    # 마커 추가
    folium.Marker(
        location=[point['lat'], point['lon']],
        popup=(
            f"{point['name']}<br>"
            f"예측 위험등급: <b>{risk}</b><br>"
            f"고위험 확률: {proba:.2f}"
        ),
        icon=folium.Icon(color=color)
    ).add_to(m)

# 지도 출력
m

# 지금 독립변수 경위도만 사용하니까 랜덤포레스트 회귀가 설명력이 6%밖에 안됨...
# osm을 통해서 환경 데이터를 수집해보려고 함

# 예시로 중부선 구간에서 임의 위치 가져와서 실행
import osmnx as ox

# 좌표 & 반경 설정
lat, lon = 36.74561896, 127.4691292
radius = 500  # 500m

# 가져올 태그 정의
tags = {
    'landuse': True,       # 토지 이용 (forest, farmland 등)
    'natural': True,       # 자연 환경 (water, wood 등)
    'highway': True,       # 도로 정보
    'building': True,      # 건물 정보
    'barrier': True        # 울타리 등 (로드킬 영향 클 수 있음)
}

# 환경 데이터 가져오기
gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)

# 확인할 컬럼 리스트
columns_to_check = ['geometry', 'landuse', 'natural', 'highway', 'building', 'barrier']

# 실제 존재하는 컬럼만 추출
existing_cols = [col for col in columns_to_check if col in gdf.columns]

# 안전하게 출력
if existing_cols:
    print(gdf[existing_cols].head())
else:
    print("해당 위치에서 가져온 컬럼이 없습니다.")

# 숲 유무
if 'landuse' in gdf.columns:
    forest = int('forest' in gdf['landuse'].dropna().values)
else:
    forest = 0

# 건물 개수
if 'building' in gdf.columns:
    num_buildings = gdf['building'].notnull().sum()
else:
    num_buildings = 0

# motorway 유무
if 'highway' in gdf.columns:
    has_motorway = int('motorway' in gdf['highway'].dropna().values)
else:
    has_motorway = 0

print(f"숲 유무: {forest}")
print(f"건물 개수: {num_buildings}")
print(f"고속도로(motorway) 유무: {has_motorway}")

# 데이터 예시: df_all에 '위도'와 '경도' 컬럼이 있다고 가정
# df_all = pd.DataFrame(...)

radius = 500  # 반경 500m
tags = {
    'landuse': True,
    'natural': True,
    'highway': True,
    'building': True,
    'barrier': True
}

env_features = []

for idx, row in df_all.iterrows():
    lat, lon = row['위도'], row['경도']
    print(f"[{idx}] 위치: 위도={lat}, 경도={lon}")
    
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)

        # 숲 유무
        forest_tags = ['forest', 'wood']
        has_forest = 0
        if 'landuse' in gdf.columns:
            if any(tag in gdf['landuse'].dropna().values for tag in forest_tags):
                has_forest = 1
        if 'natural' in gdf.columns:
            if any(tag in gdf['natural'].dropna().values for tag in forest_tags):
                has_forest = 1

        # 농지 유무
        has_farmland = 0
        if 'landuse' in gdf.columns:
            has_farmland = int('farmland' in gdf['landuse'].dropna().values)

        # 도로 타입 개수
        highway_count = 0
        if 'highway' in gdf.columns:
            highway_count = gdf['highway'].notnull().sum()

        # 건물 개수
        num_buildings = 0
        if 'building' in gdf.columns:
            num_buildings = gdf['building'].notnull().sum()

        # barrier 유무
        has_barrier = 0
        if 'barrier' in gdf.columns:
            has_barrier = int(gdf['barrier'].notnull().any())

        env_features.append({
            '숲_유무': has_forest,
            '농지_유무': has_farmland,
            '도로_개수': highway_count,
            '건물_개수': num_buildings,
            'barrier_유무': has_barrier
        })

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        env_features.append({
            '숲_유무': 0,
            '농지_유무': 0,
            '도로_개수': 0,
            '건물_개수': 0,
            'barrier_유무': 0
        })

# 합치기
env_df = pd.DataFrame(env_features)
df_all = pd.concat([df_all.reset_index(drop=True), env_df], axis=1)

# 추가된 환경피처들로 다시 랜덤포레스트 회귀분석 >> 설명력 0.05
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1️⃣ 피처/타겟 준비
feature_cols = ['위도', '경도', '숲_유무', '농지_유무', '도로_개수', '건물_개수', 'barrier_유무']
X = df_all[feature_cols]
y = df_all['발생건수'].astype(float)  # 발생건수 숫자형으로 변환 (중요!)

# 2️⃣ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ 예측
y_pred = model.predict(X_test)

# 5️⃣ 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'✅ RMSE: {rmse:.3f}')
print(f'✅ R2 Score: {r2:.3f}')

# XGBoost도 시도! > 설명력 0.109
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1️⃣ 피처/타겟 준비
feature_cols = ['위도', '경도', '숲_유무', '농지_유무', '도로_개수', '건물_개수', 'barrier_유무']
X = df_all[feature_cols]
y = df_all['발생건수'].astype(float)

# 2️⃣ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ XGBoost 회귀 모델 정의
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# 4️⃣ 모델 학습
model.fit(X_train, y_train)

# 5️⃣ 예측
y_pred = model.predict(X_test)

# 6️⃣ 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'✅ XGBoost RMSE: {rmse:.3f}')
print(f'✅ XGBoost R2 Score: {r2:.3f}')

# 하이퍼파라미터 튜닝 후 모델 학습시키니 설명력 1.43
# 최적 파라미터로 새 모델 생성
best_model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0.5,
    random_state=42
)

# train/test 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 학습
best_model.fit(X_train, y_train)

# 예측
y_pred = best_model.predict(X_test)

# 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'✅ 최종 모델 RMSE: {rmse:.3f}')
print(f'✅ 최종 모델 R2 Score: {r2:.3f}')

# 현재 하이퍼파라미터 튜닝한 랜덤포레스트 회귀 설명력이 14% osm에서 제한속도, 차선수, 도로폭 데이터 수집하려고 함.
import osmnx as ox
import pandas as pd

# 예시: df_all에 '위도', '경도' 컬럼이 있다고 가정
# df_all = pd.DataFrame(...)

radius = 50  # 반경 50m (도로 폭까지 잡기 때문에 좁게 설정)
tags = {
    'highway': True,
    'maxspeed': True,
    'lanes': True,
    'width': True  # 도로폭 가져오기
}

# 결과 저장 리스트
speed_lane_width_data = []

for idx, row in df_all.iterrows():
    lat, lon = row['위도'], row['경도']
    print(f"[{idx}] 위치: 위도={lat}, 경도={lon}")

    try:
        # 도로 피처 가져오기 (라인 타입만 추출)
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        roads = gdf[gdf.geom_type == 'LineString']
        
        # 제한속도 추출
        maxspeed = None
        if 'maxspeed' in roads.columns:
            maxspeed_values = roads['maxspeed'].dropna().tolist()
            if maxspeed_values:
                val = maxspeed_values[0]
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, str):
                    val = ''.join([c for c in val if c.isdigit()])
                    maxspeed = int(val) if val.isdigit() else None
                elif isinstance(val, (int, float)):
                    maxspeed = val

        # 차선수 추출
        lanes = None
        if 'lanes' in roads.columns:
            lanes_values = roads['lanes'].dropna().tolist()
            if lanes_values:
                val = lanes_values[0]
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, str):
                    lanes = int(val) if val.isdigit() else None
                elif isinstance(val, (int, float)):
                    lanes = val

        # 도로폭 추출
        width = None
        if 'width' in roads.columns:
            width_values = roads['width'].dropna().tolist()
            if width_values:
                val = width_values[0]
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, str):
                    width_digits = ''.join([c for c in val if c.isdigit() or c == '.'])
                    width = float(width_digits) if width_digits.replace('.', '', 1).isdigit() else None
                elif isinstance(val, (int, float)):
                    width = val

        speed_lane_width_data.append({
            '제한속도': maxspeed if maxspeed is not None else 0,
            '차선수': lanes if lanes is not None else 0,
            '도로폭': width if width is not None else 0
        })

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        speed_lane_width_data.append({
            '제한속도': 0,
            '차선수': 0,
            '도로폭': 0
        })

# 결과 병합
speed_lane_width_df = pd.DataFrame(speed_lane_width_data)
df_all = pd.concat([df_all.reset_index(drop=True), speed_lane_width_df], axis=1)

# 전체 상관계수 확인 후 필요 없는 컬럼 삭제
corr = df_all[['위도', '경도', '숲_유무', '농지_유무', '도로_개수', '건물_개수', 'barrier_유무', '제한속도', '차선수', '발생건수']].corr()

# 발생건수와 상관관계만 출력
print(corr['발생건수'].sort_values(ascending=False))

# 최종 모델
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import numpy as np

# 1️⃣ 피처 & 타겟
feature_cols = ['위도', '경도', '숲_유무', 'barrier_유무', '제한속도', '차선수']
X = df_all[feature_cols]
y = df_all['발생건수'].astype(float)

# 2️⃣ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ XGBoost 모델 (기본 튜닝으로 먼저 정의)
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# 4️⃣ 모델 학습
model.fit(X_train, y_train)

# 5️⃣ 예측
y_pred = model.predict(X_test)

# 6️⃣ 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'✅ XGBoost 기본 튜닝 RMSE: {rmse:.3f}')
print(f'✅ XGBoost 기본 튜닝 R2 Score: {r2:.3f}')

# 🚀 추가: 하이퍼파라미터 튜닝 (RandomizedSearchCV)
param_dist = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.5, 1, 2]
}

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    scoring=rmse_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 튜닝 학습
random_search.fit(X, y)

print("✅ Best Parameters:", random_search.best_params_)
print("✅ Best RMSE (CV):", np.sqrt(-random_search.best_score_))

# ✅ 최적 파라미터로 새 모델 생성
best_params = random_search.best_params_

best_model = xgb.XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    random_state=42
)

# 학습
best_model.fit(X_train, y_train)

# 예측
y_pred_best = best_model.predict(X_test)

# 평가
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print(f'✅ 최종 튜닝 XGBoost RMSE: {rmse_best:.3f}')
print(f'✅ 최종 튜닝 XGBoost R2 Score: {r2_best:.3f}')

# 회귀모델로 임의 지점에 대한 발생건수 예측 시각화
# 🚀 신규 구간 예측용 데이터
new_points = [
    {'name': '중부선', 'lat': 37.5465, 'lon': 127.2003},
    {'name': '경부선', 'lat': 36.3504, 'lon': 127.3845},
    {'name': '중앙선', 'lat': 36.8725, 'lon': 128.5507},
    {'name': '서해안선', 'lat': 36.7763, 'lon': 126.4504}
]

# 🧮 최적 모델로 신규 포인트 예측
print("\n🚦 신규 구간 예측 결과 (최적 튜닝 모델):")
for point in new_points:
    X_new = pd.DataFrame([[
        point['lat'],     # 위도
        point['lon'],     # 경도
        0,                # 숲_유무 (기본값)
        0,                # barrier_유무 (기본값)
        80,               # 제한속도 (예: 80km/h)
        2                 # 차선수 (예: 2차선)
    ]], columns=feature_cols)
    
    predicted = best_model.predict(X_new)[0]
    print(f"{point['name']} (위도: {point['lat']}, 경도: {point['lon']}) → 예상 로드킬 발생건수: {predicted:.2f}")

# 🗺️ 지도 시각화
m = folium.Map(location=[36.5, 127.5], zoom_start=7)

for point in new_points:
    X_new = pd.DataFrame([[
        point['lat'],
        point['lon'],
        0,
        0,
        80,
        2
    ]], columns=feature_cols)
    
    predicted = best_model.predict(X_new)[0]
    
    folium.CircleMarker(
        location=[point['lat'], point['lon']],
        radius=10,
        popup=f"{point['name']} - 예측 발생건수: {predicted:.2f}",
        color='blue',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# 지도 출력
m

# 로드킬 예측값을 통한 히트맵 제작 (최적 모델 기반)

# 그리드 데이터 생성
import numpy as np
import pandas as pd

# 대한민국 범위 설정
lat_range = np.arange(34.5, 38.5, 0.05)
lon_range = np.arange(126.0, 129.5, 0.05)

grid_points = []

for lat in lat_range:
    for lon in lon_range:
        grid_points.append([lat, lon, 0, 0, 80, 2])  # 숲_유무=0, barrier_유무=0, 제한속도=80, 차선수=2

grid_df = pd.DataFrame(grid_points, columns=['위도', '경도', '숲_유무', 'barrier_유무', '제한속도', '차선수'])

# 예측 수행 (최적 파라미터 모델 사용)
predictions = best_model.predict(grid_df)
grid_df['예측발생건수'] = predictions

# 히트맵 시각화
import folium
from folium.plugins import HeatMap

# 지도 초기화
m = folium.Map(location=[36.5, 127.8], zoom_start=7)

# HeatMap 데이터: [[위도, 경도, 가중치]]
heat_data = [[row['위도'], row['경도'], row['예측발생건수']] for index, row in grid_df.iterrows()]

# HeatMap 추가
HeatMap(
    heat_data,
    radius=12,
    blur=20,
    min_opacity=0.2,
    max_zoom=13
).add_to(m)

# 지도 출력
m

# 교통량 데이터 수집을 위해 한국도로공사 api 연동
# 공간정보노선현황 이것도 api로 가져와야해? 짜증나네 ㅋㅋ
import requests
import pandas as pd

# ✅ 발급받은 인증키
service_key = 'your_api_key'

# ✅ API 호출 URL
url = 'https://data.ex.co.kr/openapi/roadEtcInfo/spinRouteList'

# ✅ 요청 파라미터 설정
params = {
    'key': service_key,
    'type': 'json'
}

# ✅ API 호출
response = requests.get(url, params=params)

# ✅ 결과 확인
if response.status_code == 200:
    data = response.json()
    items = data.get('list', [])
    print(f'✅ 노선 수집 성공: {len(items)}건')
    
    # ✅ 데이터프레임 변환
    df_routes = pd.DataFrame(items)
    print(df_routes.head())
else:
    print(f'❌ 호출 실패: {response.status_code} / {response.text}')

# df_routes: 새로 가져온 노선현황
# df_all: 기존 데이터프레임

df_all = df_all.merge(
    df_routes[['routeNm', 'routeCd', 'routeNo']],
    left_on='노선명',
    right_on='routeNm',
    how='left'
)

# 필요 없는 routeNm 컬럼 삭제
df_all = df_all.drop(columns=['routeNm'])

# 결과 확인
print(df_all.head())

# 아래는 노선별 전체 영업소 교통량 api 연동 과정인데 실시간 데이터만 가져오는 문제 발견해서 실패함..
import requests
import pandas as pd
from datetime import datetime, timedelta

# ✅ 발급받은 인증키
service_key = '3305368534'

# ✅ df_all에 있는 routeNo만 고유 추출
route_nos = df_all['routeNo'].dropna().unique()

# ✅ 결과 저장용 리스트
traffic_data = []

# ✅ 조회할 기간 설정 (예: 2023-06-01 ~ 2023-06-30)
start_date = datetime(2023, 6, 1)
end_date = datetime(2023, 6, 30)

# ✅ 날짜 루프
current_date = start_date
while current_date <= end_date:
    sum_date = current_date.strftime('%Y%m%d')
    print(f"📅 {sum_date} 데이터 수집 중...")

    # ✅ 노선 루프
    for route_no in route_nos:
        params = {
            'key': service_key,
            'type': 'json',
            'routeNo': route_no,
            'sumDate': sum_date
        }
        url = 'https://data.ex.co.kr/openapi/trafficapi/trafficRoute'
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            items = data.get('list', [])
            if items:
                for item in items:
                    item['routeNo'] = route_no
                    item['sumDate'] = sum_date  # 날짜도 붙이기
                    traffic_data.append(item)
            print(f'✅ {route_no} 호출 성공, {len(items)}건 수집됨')
        else:
            print(f'❌ {route_no} 호출 실패: {response.status_code}')
    
    current_date += timedelta(days=1)

# ✅ 데이터프레임으로 변환
df_traffic = pd.DataFrame(traffic_data)
print(df_traffic.head())
