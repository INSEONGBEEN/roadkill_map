import pandas as pd
from src import collect_data, preprocess, model, visualize

# 1️⃣ 데이터 수집 (공공데이터 API 또는 로컬 파일)
# df = collect_data.fetch_from_api(service_key='your_api_key')
df = pd.read_csv('data/processed/roadkill_2019_2023.csv')  # 이미 처리된 데이터 사용

# 2️⃣ 전처리 및 환경 변수 생성
df_cleaned = preprocess.clean_data(df)
df_features = preprocess.add_osm_features(df_cleaned)
df_features = preprocess.add_speed_lane_width(df_features)

# 3️⃣ 탐색적 데이터 분석 및 시각화
visualize.plot_basic_eda(df_features)
visualize.plot_correlation_heatmap(df_features)

# 4️⃣ 회귀 모델 학습 및 평가 (XGBoost)
xgb_model, xgb_metrics = model.train_xgboost(df_features)
print("✅ XGBoost 모델 성능:", xgb_metrics)

# 5️⃣ 분류 모델 학습 및 평가 (RandomForest 이진분류)
rf_clf, clf_metrics = model.train_random_forest_classifier(df_features)
print("✅ RandomForest 이진 분류 성능:", clf_metrics)

# 6️⃣ 시각화 (Feature Importance, SHAP)
visualize.plot_feature_importance(xgb_model, df_features.columns.tolist())
visualize.plot_shap_summary(xgb_model, df_features)

# 7️⃣ 지도 시각화 (예측 마커 및 히트맵)
model.visualize_predictions_on_map(xgb_model)
model.visualize_heatmap_on_map(xgb_model)

print("🎉 모든 실행 완료!")
