# 🦌 머신러닝 기반 로드킬 예측 & 히트맵 시각화 프로젝트

*@injjang*

2025.05 ~ 2025.05

---

## Live Demo

- https://inseongbeen.github.io/roadkill_map/saved_resource_heatmap.html  
- https://inseongbeen.github.io/roadkill_map/saved_resource_predict.html

## GitHub Repo

https://github.com/INSEONGBEEN/roadkill_map

## Dev Log

https://lnjjang.tistory.com/

---

## 💬 프로젝트 소개

**로드킬 공공데이터와 환경 데이터를 결합**하여 로드킬 발생건수를 예측하고, 지도 시각화까지 구현한 프로젝트입니다. Python의 XGBoost 및 folium을 활용하여 **로드킬 다발지 예측 및 히트맵 시각화**를 진행했으며, 교통량 및 도로 환경 데이터 확보를 위해 **한국도로공사 API 연동**도 시도했습니다.

---

## 🛠️ 주요 기능

- **공공데이터 API 연동**
  - 로드킬 발생 데이터(2019~2023) 확보 및 가공
  - 한국도로공사 교통량 API 연동 시도

- **EDA 및 전처리**
  - 결측치 및 이상치 처리, OSM을 통한 환경 피처 구성 (숲 유무, barrier 유무 등)

- **머신러닝 예측**
  - 랜덤포레스트 회귀 & 분류 실험
  - XGBoost 회귀 + 하이퍼파라미터 튜닝 (최종 R² 약 0.14)

- **지도 시각화**
  - 주요 고속도로 로드킬 예측값 CircleMarker 표시
  - 전국 단위 히트맵 제작 및 저장 (folium HeatMap)

---

## 🔄 구현 예정 기능

- 과거 교통량 데이터 확보 후 **모델 보강**
- 로드킬 다발지 클러스터링 및 위험구간 분석
- 인프라(가로등 등) 추가 연계 → 종합 안전지도 구현

---

## ⚙️ 기술 스택

| 항목 | 사용 기술 |
|---|---|
| 언어 | Python |
| 데이터 처리 | pandas, geopandas, scikit-learn, XGBoost |
| 시각화 | folium, matplotlib |
| API | 공공데이터포털 API, 한국도로공사 API |
| 개발환경 | Jupyter Notebook |
| 결과물 저장 | HTML 파일로 지도 저장 |

---

## 📊 모델 성능

- **랜덤포레스트 회귀**: R² 0.06
- **XGBoost 기본 모델**: R² 0.109
- **튜닝 XGBoost 모델**: ✅ **최종 R² 0.139**

---

## 🔍 주요 인사이트

- 위도, 경도, 제한속도, 차선수 등 도로 위치/환경적 요인이 로드킬 발생에 영향
- 숲과 barrier는 **로드킬 감소에 긍정적인 요인**
- 시각화 및 지도 기반 예측 결과는 **실제 발생지와 유사한 분포를 보임**

---

## 🗺️ 시각화 결과

- 신규 고속도로 구간 예측 Circle 시각화
- 전국 범위 히트맵 생성 및 저장
- SHAP summary plot을 통한 모델 해석도 진행

---

## ✍️ 느낀 점

- **데이터 수집의 중요성**: 모델보다도 **데이터의 질과 범위 확보**가 훨씬 중요
- **시각화의 전달력**: 지도 기반 시각화가 인사이트 전달에 매우 효과적
- **실전 감각 향상**: API 연동, 예측모델 구축, 시각화까지 전 과정을 직접 수행
