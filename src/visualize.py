import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_roadkill_hist(df):
    plt.figure(figsize=(8, 5))
    df['발생건수'].hist(bins=15)
    plt.title('로드킬 발생건수 분포')
    plt.xlabel('발생건수')
    plt.ylabel('빈도')
    plt.show()

def plot_annual_roadkill(df):
    df.groupby('년도')['발생건수'].sum().plot(kind='bar', figsize=(8, 5))
    plt.title('연도별 로드킬 발생건수 합계')
    plt.xlabel('년도')
    plt.ylabel('총 발생건수')
    plt.show()

def plot_branch_roadkill(df):
    df.groupby('본부명')['발생건수'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))
    plt.title('본부별 로드킬 발생건수')
    plt.ylabel('총 발생건수')
    plt.xlabel('본부명')
    plt.show()

def plot_top_routes(df):
    df.groupby('노선명')['발생건수'].sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10, 5))
    plt.title('노선별 상위 10개 로드킬 다발구간')
    plt.ylabel('총 발생건수')
    plt.xlabel('노선명')
    plt.show()

def plot_location_scatter(df):
    plt.figure(figsize=(8, 8))
    plt.scatter(df['경도'], df['위도'], s=df['발생건수']*5, alpha=0.5)
    plt.title('로드킬 발생 위치 (경도 vs 위도)')
    plt.xlabel('경도')
    plt.ylabel('위도')
    plt.show()

def plot_correlation_heatmap(df):
    corr = df[['위도', '경도', '숲_유무', '농지_유무', '도로_개수', '건물_개수', 'barrier_유무', '제한속도', '차선수', '발생건수']].corr()
    print("\n🚨 발생건수와의 상관계수:")
    print(corr['발생건수'].sort_values(ascending=False))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('📈 로드킬 데이터 상관관계 히트맵', fontsize=16)
    plt.show()

def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_cols, importance)
    plt.xlabel('중요도')
    plt.title('🚗 로드킬 예측 모델 Feature Importance')
    plt.show()

def plot_shap_summary(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
