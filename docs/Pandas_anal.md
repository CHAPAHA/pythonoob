# 📊 2025-06-06 데이터 분석 로그
```bash
cd docs
echo "# 📊 2025-06-06 데이터 분석 로그" > 2025-06-06_README.md
```

## Pandas 공부
### pandas: 구조화된 데이터 처리와 분석
1. 역할
표 형태의 데이터(CSV, Excel, SQL 등)를 다루는 대표적 라이브러리
R의 data.frame과 유사하며, **탐색적 분석(EDA)**과 전처리 단계에 필수

2. 주요 객체
Series: 1차원 데이터 구조 (벡터), DataFrame: 2차원 데이터 구조 (행렬/표 형태)

3. 주요 기능

데이터 불러오기	pd.read_csv(), pd.read_excel()

결측치 처리	df.dropna(), df.fillna()

필터링	df[df["Score"] > 90]

그룹별 집계	df.groupby("Gender").mean()

인덱싱	.loc, .iloc

열 추가/변형	df["LogScore"] = np.log(df["Score"])

4. pandas의 강점
SPSS/R에서 하던 탐색, 정제, 요약 통계처리를 거의 모두 수행 가능, apply(), groupby() 같은 함수형 처리로 유연성 ↑

### 괄호정리

소괄호	()	
1. 함수 호출 print("Hello")
2. 튜플(tuple) x = (1, 2)
3. 수식 우선순위 제어	y = (3 + 5) * 2

대괄호	[]	
1. 리스트(list) 정의 scores = [90, 85, 77]
2. 인덱싱 / 슬라이싱 scores[0]
3. 딕셔너리/배열 요소 접근	df["Gender"]

중괄호	{}	
1. 딕셔너리(dict) 정의 d = {"A": 1, "B": 2}
2. 집합(set) 정의 s = {1, 2, 3}
3. 포맷 문자열 내부 표현 f"Score: {x}"	 

## Scikit-learn 공부
### 🤖 scikit-learn (sklearn): 머신러닝 모델링의 표준 도구
1. 역할

머신러닝 파이프라인을 구성하는 라이브러리
회귀, 분류, 클러스터링, 차원축소, 모델 평가 등 포함

2. 모듈구조
### sklearn
#### ├── preprocessing     ← 전처리 (스케일링 등)
#### ├── model_selection   ← 데이터 분할, 교차검증 등
#### ├── linear_model      ← 선형 회귀, 로지스틱 회귀 등
#### ├── ensemble          ← 랜덤포레스트 등 앙상블 기법
#### ├── metrics           ← RMSE, 정확도 등 평가 지표
#### └── pipeline          ← 전체 흐름을 파이프라인으로 구성

3. 사용예시
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["StudyHours", "AttendanceRate"]]
y = df["TestScore_Math"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```
4. 기본 설계 철학

모든 모델은 fit(), predict() 메서드 중심

입력은 Numpy 배열 또는 pandas DataFrame, 출력도 간결

통계적 추론보다는 예측 성능에 초점 (회귀계수의 유의성 검정은 없음)

만약 회귀계수의 통계적 검정이 필요하다면 statsmodels를 병행하여 사용


## Python
```
# ✅ 0. 라이브러리 불러오기
from datasets import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 1. 데이터 불러오기
dataset = load_dataset("neuralsorcerer/student-performance")
df = dataset["train"].to_pandas()

# ✅ 2. 열 선택 및 결측치 제거
cols = ["Gender", "ParentalEducation", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
df = df[cols].dropna()

# ✅ 3. 범주형 변수 인코딩
le_gender = LabelEncoder()
le_parent = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])  # ex) Male=1, Female=0
df["ParentalEducation"] = le_parent.fit_transform(df["ParentalEducation"])  # 학력 수준 수치화

# ✅ 4. 성별에 따른 평균 성적 비교 (EDA)
print("📊 성별에 따른 평균 성적:")
print(df.groupby("Gender")[["TestScore_Math", "TestScore_Reading", "TestScore_Science"]].mean())

# ✅ 5. 시각화: 성별 수학 점수 분포
sns.boxplot(x="Gender", y="TestScore_Math", data=df)
plt.title("성별에 따른 수학 점수 분포")
plt.xlabel("Gender (0: Female, 1: Male)")
plt.ylabel("TestScore_Math")
plt.tight_layout()
plt.show()

# ✅ 6. 회귀분석: 부모 학력 + 성별로 수학 점수 예측
X = df[["Gender", "ParentalEducation"]]
y = df["TestScore_Math"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ✅ 7. 평가 지표 출력
print("\n📈 회귀 모델 평가:")
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))

# ✅ 8. 예측 결과 시각화
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("실제 점수")
plt.ylabel("예측 점수")
plt.title("수학 성적 예측 결과 (실제 vs. 예측)")
plt.tight_layout()
plt.show()
```
위 코드 실행 중 
```
# ✅ 6. 회귀분석: 부모 학력 + 성별로 수학 점수 예측
X = df[["Gender", "ParentalEducation"]]
y = df["TestScore_Math"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```
이후 멈추어버림. 학습과정이 생각보다 시간이 많이 걸리는 것으로 추정

