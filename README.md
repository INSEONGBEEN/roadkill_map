# 🦌 머신러닝 기반 로드킬 예측 & 시각화 시스템

도로교통공사 로드킬 공공데이터를 기반으로 발생 위치를 예측하고,
지도에 시각화하는 프로젝트입니다.

- ML 모델: XGBoost Regressor
- 지도 시각화: Folium 기반 예측 지도 / 히트맵 제공
- 외부 데이터 연동: OSM 환경 데이터, 도로공사 API 활용

---

## 📁 디렉토리 구조

```
roadkill_map/
├── README.md
├── requirements.txt
├── .gitignore
│
├── 📁 data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── 📁 src/
│   ├── __init__.py
│   ├── main.py
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── model.py
│   └── visualize.py
│
├── 📁 outputs/
│   ├── maps/
│   └── plots/
│
└── 📁 docs/
    └── architecture.png
```

---

## 🚀 실행 방법

1. 레포지토리 클론
```bash
git clone https://github.com/INSEONGBEEN/roadkill_map.git
cd roadkill_map
```

2. 가상환경 생성 및 패키지 설치
```bash
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

3. 데이터 준비 (선택)
- `data/raw/`에 로드킬 원본 CSV 및 외부 데이터 위치

4. 메인 실행
```bash
python src/main.py
```

5. 결과 확인
- 예측 지도: `outputs/maps/saved_resource_predict.html`
- 히트맵: `outputs/maps/saved_resource_heatmap.html`
- 시각화 이미지: `outputs/plots/`

---

## 🌐 GitHub Pages 결과물

- [예측 지도 보기](https://inseongbeen.github.io/roadkill_map/saved_resource_predict.html)
- [히트맵 보기](https://inseongbeen.github.io/roadkill_map/saved_resource_heatmap.html)
