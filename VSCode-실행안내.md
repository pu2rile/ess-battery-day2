# VS Code에서 실행하고 GitHub에 올리기

## 1. 프로젝트 열기

`ESS-Battery-DAY2.code-workspace` 파일을 VS Code로 열거나, **파일 → 폴더 열기**에서 이 폴더를 선택합니다.
Python·Python Debugger·Jupyter 확장을 사용합니다. 현재 컴퓨터에는 이 확장들이 설치되어 있습니다.

## 2. 실행 환경

현재 컴퓨터에는 검증된 Python 3.11 라이브러리와 기존 HDF5 패키지를 재사용하는 로컬 `.venv`를 준비했습니다.
이 환경은 현재 작업 폴더용이며 GitHub와 ZIP에서는 제외됩니다. 다른 위치로 옮기거나 다른 컴퓨터에서 실행할 때는 아래 방식으로 새 환경을 만듭니다.

VS Code의 **터미널 → 새 터미널**을 열고 Python 3.11로 가상환경을 만듭니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows에서는 `py -3.11 -m venv .venv`와 `.venv\Scripts\activate`를 사용합니다.
명령 팔레트에서 **Python: Select Interpreter**를 실행하여 `.venv`의 Python을 선택합니다.
새 환경의 가상환경과 패키지 설치는 최초 한 번 진행합니다. 저장소와 ZIP에는 가상환경이 포함되지 않습니다.

## 3. 코드 실행

`main.py`를 열고 **F5 → DAY2: 전체 분석 실행**을 선택합니다. 또는 터미널에서 다음을 실행합니다.

```bash
python main.py
```

포함된 피처 CSV를 읽어 학습·평가·그래프·오류 분석을 순서대로 실행하고 `results/`를 갱신합니다.
분석 로직은 `src/features.py`, `src/train.py`, `src/plots.py`, `src/diagnostics.py`로 나눴습니다.
노트북으로 확인하려면 `30-ESSHealth-DAY2-complete.ipynb`를 열고 오른쪽 위 커널에서 같은 `.venv`를 선택한 뒤 **Run All**을 누릅니다.
`main.py` 실행은 노트북에 저장된 출력을 자동으로 바꾸지 않습니다.

## 4. GitHub 업로드

이 폴더를 로컬 Git 저장소로 초기화했습니다. VS Code 왼쪽 **소스 제어**에서 파일 변경 목록을 확인할 수 있습니다.
파일을 검토한 뒤 커밋 메시지를 입력해 커밋하고, **Publish Branch / GitHub에 게시**에서 공개(Public) 저장소로 게시합니다.
GitHub 로그인이나 Git 작성자 이름·이메일 설정이 필요하면 본인의 정보를 사용합니다.
이미 제출용 저장소가 있다면 그 저장소를 연결해서 올립니다. 새로운 원격 주소나 계정은 이 프로젝트에 미리 지정하지 않았습니다.

코드·노트북·README·가공 피처·결과를 올립니다. `.venv`, 원본 MAT, 캐시, `.env` 파일은 `.gitignore`로 제외됩니다.
원격 업로드·커밋은 아직 하지 않았습니다. 최종 제출은 공개 저장소 링크를 반별 Slack 제출 스레드에 올리는 것입니다.

## 결과 해석

이번 VS Code 구성은 기존 분석을 실행하기 위한 것입니다. 모델 성능을 개선한 버전은 아니며,
기존 Batch 2 MAPE 124.99%와 일반화 실패에 대한 설명은 README에 유지했습니다.
