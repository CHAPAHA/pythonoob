# 왜 Grep이 안써질까

### 
- cut, grep 등은 Git의 명령어가 아님
- git help에서 보여준 명령 중 grep은 Git 내부 서브커맨드인 git grep을 의미합니다.
- 예: git grep "검색어" ← 이것은 작동합니다.

반면 PowerShell에서 그냥 grep, cut 등은 리눅스/유닉스 명령어이며, Windows PowerShell에는 기본 제공되지 않습니다.

| 명령어          | 용도         | PowerShell 대체 명령                             | Git과 관계             |
| ------------ | ---------- | -------------------------------------------- | ------------------- |
| `grep`       | 텍스트 검색     | `Select-String`                              | `git grep`으로는 사용 가능 |
| `cut`        | 텍스트 필드 자르기 | `ForEach-Object`, `-split`, `Format-Table` 등 | Git과 무관             |
| `awk`, `sed` | 텍스트 조작     | 없음 (PowerShell은 파이프라인 기반 구조로 처리)             | Git과 무관             |

```
# grep 대체
Select-String "import" file.py

# cut 대체
Get-Content file.txt | ForEach-Object { ($_ -split '\t')[1] }  # 탭 기준 두 번째 필드
```