import pandas as pd
import numpy as np
import xgboost as xgb
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, make_scorer,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE


def train_xgboost_regressor(df):
    feature_cols = ['위도', '경도', '숲_유무', 'barrier_유무', '제한속도', '차선수']
    X = df[feature_cols]
    y = df['발생건수'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X, y)

    best_model = xgb.XGBRegressor(**random_search.best_params_, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    print("✅ XGBoost R2 Score:", r2_score(y_test, y_pred))
    return best_model


def predict_new_points(model):
    feature_cols = ['위도', '경도', '숲_유무', 'barrier_유무', '제한속도', '차선수']
    new_points = [
        {'name': '중부선', 'lat': 37.5465, 'lon': 127.2003},
        {'name': '경부선', 'lat': 36.3504, 'lon': 127.3845},
        {'name': '중앙선', 'lat': 36.8725, 'lon': 128.5507},
        {'name': '서해안선', 'lat': 36.7763, 'lon': 126.4504}
    ]
    m = folium.Map(location=[36.5, 127.5], zoom_start=7)

    for point in new_points:
        X_new = pd.DataFrame([[point['lat'], point['lon'], 0, 0, 80, 2]], columns=feature_cols)
        pred = model.predict(X_new)[0]
        folium.CircleMarker(
            location=[point['lat'], point['lon']],
            radius=10,
            popup=f"{point['name']} - 예측 발생건수: {pred:.2f}",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    return m


def draw_heatmap(model):
    lat_range = np.arange(34.5, 38.5, 0.05)
    lon_range = np.arange(126.0, 129.5, 0.05)
    grid = [[lat, lon, 0, 0, 80, 2] for lat in lat_range for lon in lon_range]
    grid_df = pd.DataFrame(grid, columns=['위도', '경도', '숲_유무', 'barrier_유무', '제한속도', '차선수'])
    grid_df['예측발생건수'] = model.predict(grid_df)

    heat_data = [[row['위도'], row['경도'], row['예측발생건수']] for _, row in grid_df.iterrows()]
    m = folium.Map(location=[36.5, 127.8], zoom_start=7)
    HeatMap(heat_data, radius=12, blur=20, min_opacity=0.2).add_to(m)
    m.save('로드킬_실제_vs_예측_히트맵.html')
    return m


def train_risk_classifier(df):
    def classify_risk(val):
        if val <= 3:
            return '저위험'
        elif val <= 7:
            return '중위험'
        return '고위험'

    df['위험등급'] = df['발생건수'].apply(classify_risk)
    df['고위험_이진'] = df['위험등급'].apply(lambda x: 1 if x == '고위험' else 0)
    X = df[['위도', '경도']]
    y = df['고위험_이진']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("✅ Risk Classification Report\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    return clf
