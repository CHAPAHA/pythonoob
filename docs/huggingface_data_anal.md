# Huggingface Data Analysis
```bash
cd docs
>> echo "# Huggingface Data Analysis" > huggingface_data_anal.md
```


# 🎯 Hugging Face 데이터 분석: Student Performance Dataset

## 1. 데이터셋 소개
- 학생 성취도 데이터셋
- 학업 성취도, 배경 정보(부모 학력, 학습시간 등) 포함
- 데이터 개수 천만개

## 2. 데이터 다운로드 방법

### PowerShell
```bash
curl -X GET "https://datasets-server.huggingface.co/first-rows?dataset=neuralsorcerer%2Fstudent-performance&config=default&split=train" -o sample.json
```
## Pwer Shell 로 다운이 안되는 이유
-o sample.json → 파일로 저장
이 창에서는 샘플만 주어져서 전체 다운로드 불가

## Git 방식
이 데이터셋은 Git 저장소가 없는 일반 Datasets 형식이라 git clone으로는 다운로드 불가
Hugging Face는 모델은 Git으로 제공하지만
데이터셋은 라이브러리 통해 다운로드하는 방식

### Python 으로 해야 함
가상환경으로 하길 권고 이유는
 가상환경에서 설치해야 하는 이유:

1. 내 컴퓨터 전체를 더럽히지 않고 프로젝트별로 관리.
2. 버전 충돌을 막고 안정적인 개발.
3. 필요없으면 통째로 삭제하면 깔끔.
4. 팀 작업(협업)할 때 가상환경 + requirements.txt → 프로젝트 복제 편함.
그래서

``` bash
cd..
dir
.\myenv\Scripts\Activate
pip install datasets
python
from datasets import load_dataset
>>> dataset = load_dataset("neuralsorcerer/student-performance")
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.38k/5.38k [00:00<?, ?B/s]
C:\Users\sksxk\OneDrive\python\noob\myenv\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\sksxk\.cache\huggingface\hub\datasets--neuralsorcerer--student-performance. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
train.csv: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1.33G/1.33G [00:49<00:00, 26.6MB/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
validation.csv: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 166M/166M [00:07<00:00, 22.1MB/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
test.csv: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 166M/166M [00:08<00:00, 18.7MB/s]
Generating train split: 8000774 examples [00:38, 209872.32 examples/s]
Generating validation split: 999229 examples [00:03, 284640.12 examples/s]
Generating test split: 999997 examples [00:03, 276577.61 examples/s]
```

### 다운로드 완료
train split: 800,774개

validation split: 999,229개

test split: 999,997개

## 데이터 확인
```python
# 데이터셋 split 목록 보기
print(dataset)

# train split 미리보기
print(dataset['train'])

# 혹은 1개 row만 보기
print(dataset['train'][0])
```
### 데이터는 어떻게 저장되는가?

## Hugging Face 데이터 저장 위치
Hugging Face의 load_dataset() 함수는
로컬 컴퓨터에 다운로드해서 캐시(cache)해 놓는다

train.arrow, test.arrow → 데이터가 저장된 파일 (Apache Arrow 포맷)

dataset_info.json → 데이터셋 메타데이터

README.md → 설명서 파일

.arrow 포맷은 바로 열 수 있는 CSV 파일이 아니다다

Hugging Face Datasets 라이브러리가 빠르게 읽어들이기 위해 만든 포맷이다다

load_dataset 으로만 쉽게 불러올 수 있다다

CSV로 변환하고 싶으면 Pandas로 넘겨서 저장해야 함함

# Arrow와 Jason
## ARROW?
Apache Arrow는 컬럼 기반(columnar) 메모리 포맷

빠른 데이터 읽기/쓰기를 위해 만들어짐

대용량 데이터를 빠르고 효율적으로 다루기 위해 최적화된 구조

데이터셋이 커질수록 CSV 같은 텍스트 기반은 느림.

Arrow는 RAM에 최적화된 포맷이라

읽기/쓰기 속도가 빠르고 메모리 사용량이 적음음

Huggingface Datasets 라이브러리가 로컬에서 빠르게 데이터셋 읽고 쓰려고 Arrow 포맷 사용

## Json?
JavaScript Object Notation의 줄임말

텍스트 기반 데이터 교환 포맷

사람도 읽을 수 있고 기계도 쉽게 읽고 쓸 수 있음음

### Json 역할
데이터셋 메타데이터 저장에 사용

예:
데이터셋 이름
스플릿 정보 (train, test 등), 컬럼 이름과 데이터 타입,
라이선스, 설명 등

dict
intent

## Hugging face Git Clone
이게 되는 원리는 LFS 호라성화 시키고
```bash
git lfs install
git clone https://huggingface.co/datasets/neuralsorcerer/student-performance
>>
Cloning into 'student-performance'...
remote: Enumerating objects: 39, done.
remote: Counting objects: 100% (35/35), done.
remote: Compressing objects: 100% (35/35), done.
remote: Total 39 (delta 11), reused 0 (delta 0), pack-reused 4 (from 1)
Unpacking objects: 100% (39/39), 15.63 KiB | 79.00 KiB/s, done.
Updating files: 100% (5/5), done.
Filtering content: 100% (3/3), 1.54 GiB | 22.15 MiB/s, done.
```