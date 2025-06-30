# ✅ 0. 라이브러리 불러오기
# from datasets import load_dataset  # 더 이상 필요하지 않음
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# matplotlib 설정 변경
plt.ion()  # 대화형 모드 활성화
print("라이브러리 불러오기 완료")


# ✅ 1. 데이터 불러오기
# 현재 스크립트의 절대 경로를 기준으로 데이터 파일 경로 설정
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(current_dir, "student-performance", "data", "train.csv")

# 필요한 열만 선택해서 읽기
columns_to_use = ["Gender", "ParentalEducation", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
df = pd.read_csv(data_path, usecols=columns_to_use)

# 처음부터 1만개 샘플만 추출하여 작업
print("\n📊 전체 데이터 크기:", df.shape)
print("1만개 샘플 추출 중...")
df = df.sample(n=10000, random_state=42)  # 재현성을 위해 random_state 설정
print("추출된 데이터 크기:", df.shape)

# ✅ 1-1. 데이터 확인
print("\n📊 데이터 기본 정보:")
print(df.info())

print("\n📊 데이터 통계 정보:")
print(df.describe())

print("\n📊 처음 5개 행:")
print(df.head())

# 수학 점수 분포를 bins 수를 다르게 해서 비교
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(data=df, x="TestScore_Math", bins=10)
plt.title("수학 점수 분포 (bins=10)")

plt.subplot(1, 3, 2)
sns.histplot(data=df, x="TestScore_Math", bins=30)
plt.title("수학 점수 분포 (bins=30)")

plt.subplot(1, 3, 3)
sns.histplot(data=df, x="TestScore_Math", bins=50)
plt.title("수학 점수 분포 (bins=50)")

plt.tight_layout()
plt.show()

# 3과목 분포 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(data=df, x="TestScore_Math", bins=30)
plt.title("수학 점수 분포")

plt.subplot(1, 3, 2)
sns.histplot(data=df, x="TestScore_Reading", bins=30)
plt.title("읽기 점수 분포")

plt.subplot(1, 3, 3)
sns.histplot(data=df, x="TestScore_Science", bins=30)
plt.title("과학 점수 분포")

plt.tight_layout()
plt.show()

# ✅ 2. 열 선택 및 결측치 제거
cols = ["Gender", "ParentalEducation", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
df = df[cols].dropna()

# ✅ 3. 범주형 변수 인코딩
le_gender = LabelEncoder()
le_parent = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])  # ex) Male=1, Female=0
df["ParentalEducation"] = le_parent.fit_transform(df["ParentalEducation"])  # 학력 수준 수치화

# ✅ 4. 성별에 따른 평균 성적 비교 (EDA)
print("\n📊 성별에 따른 평균 성적:")
print(df.groupby("Gender")[["TestScore_Math", "TestScore_Reading", "TestScore_Science"]].mean())

# ✅ 5. 시각화: 성별 수학 점수 분포
sns.boxplot(x="Gender", y="TestScore_Math", data=df)
plt.title("성별에 따른 수학 점수 분포")
plt.xlabel("Gender (0: Female, 1: Male)")
plt.ylabel("TestScore_Math")
plt.tight_layout()
plt.show()

# ✅ 6. 회귀분석: 부모 학력 + 성별로 수학 점수 예측
print("\n📈 회귀 분석 시작...")

# 특성과 타겟 변수 간의 관계 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x="ParentalEducation", y="TestScore_Math", data=df)
plt.title("부모 학력별 수학 점수 분포")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(x="Gender", y="TestScore_Math", data=df)
plt.title("성별 수학 점수 분포")

plt.tight_layout()
plt.draw()
plt.pause(2)

# 상관관계 출력
print("\n📊 특성 간 상관관계:")
correlation = df[["Gender", "ParentalEducation", "TestScore_Math"]].corr()
print(correlation["TestScore_Math"])

X = df[["Gender", "ParentalEducation"]]
y = df["TestScore_Math"]

print("\n데이터 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("모델 학습 중...")
model = LinearRegression()
model.fit(X_train, y_train)

print("\n📊 특성 중요도:")
for feature, coef in zip(["Gender", "ParentalEducation"], model.coef_):
    print(f"{feature}: {coef:.4f}")

print("예측 수행 중...")
y_pred = model.predict(X_test)

# ✅ 7. 평가 지표 출력
print("\n📈 회귀 모델 평가:")
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5  # RMSE 계산 방식 변경
print("RMSE:", rmse)
print("R²:", r2_score(y_test, y_pred))

# ✅ 8. 예측 결과 시각화
plt.figure(figsize=(8, 6))  # 그래프 크기 지정
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("실제 점수")
plt.ylabel("예측 점수")
plt.title("수학 성적 예측 결과 (실제 vs. 예측)")
plt.tight_layout()
plt.show()

# 마지막에 모든 그래프가 닫히지 않도록 대기
plt.ioff()  # 대화형 모드 비활성화
plt.show()