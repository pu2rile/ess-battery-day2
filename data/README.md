# 데이터 출처와 재현

## 포함된 가공 데이터

`processed/features.csv`는 원본 MAT에서 만든 **129셀 × 메타데이터·피처** 표다.
초기 100사이클 이내 데이터만 입력으로 사용하고, 타깃은 원본 `cycle_life`다.
`processed/cell_audit.csv`는 원본 139셀 전체와 포함/제외 사유를 기록한다.
`processed/source_manifest.json`에는 이번에 사용한 원본 파일 크기와 SHA-256 해시가 있다.
가공 데이터만으로 모델 학습·성능 평가·노트북을 재실행할 수 있다.

## 원본

- 연구: Severson et al. (2019), DOI: https://doi.org/10.1038/s41560-019-0356-8
- 저자 데이터 처리 코드: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- DAY1 작업에서 확보한 MAT 미러: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

원본 약 7.7 GiB는 패키지에 포함하지 않았다. 아래 파일을 확보하여 `data/raw/`에 넣는다.

```text
data/raw/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

프로젝트 루트에서 실행한다.

```bash
python -m src.features --raw-dir data/raw --out-dir data/processed
python -m src.train
python -m src.plots
python -m src.diagnostics
```

생성된 원본 해시와 이 패키지에 기록된 해시가 다르면 파일 버전이 다를 수 있으므로 셀 수·라벨을 확인한다.
공개 저장소에서는 데이터의 출처와 배포처 이용 조건을 유지한다.

## 전처리 규칙

- 원본 순서를 기준으로 `B1C0`, `B1C1` 등의 셀 ID를 생성한다.
- summary의 cycle이 1부터 연속인지 검사하여 Qdlin 배열의 cycle 10/100 위치를 확인한다.
- cycle 2–100 summary와 cycle 10/100 Qdlin만 피처로 사용한다.
- Qdlin과 Vdlin 길이가 다르면 임의 절단하지 않고 오류로 중단한다.
- QD 0.5–1.3 Ah 밖, IR·충전 시간 0 이하는 피처 결측으로 처리한다.
- 라벨 결측, cycle 100 이후 수명 미보유, 초기 데이터 부족, 잘못된 Delta Q를 제외한다.
- 이번 실행의 제외 사유는 라벨 결측뿐이다: B2 8셀, B3 2셀. 결측 라벨을 임의로 EOL로 간주하지 않는다.
- B1/B2/B3 각각 46/39/44셀을 학습 또는 평가에 사용한다.
- 충전 정책 그룹은 숫자형 C1/Q1/C2 조합으로 정의한다. `-newstructure` 등의 접미사는 그룹을 쪼개는 데 사용하지 않는다.
- 제공된 barcode는 HDF5/MATLAB 객체 참조 형태여서 물리적 일련번호로 사용하지 않았다. 배치+원본 인덱스를 셀 키로 사용하며, 동일 초기 피처의 완전 중복 여부를 별도 검사한다. 이 검사가 물리적 동일 셀의 완전한 검증을 뜻하지는 않는다.

논문 저자 코드에 등장하는 `2017-06-30` 배치와 이번 과제의 `2018-02-20` 배치는 다르다.
다른 배치 버전에 해당하는 이어 붙이기·셀 제외 인덱스를 이번 데이터에 무조건 적용하지 않았다.
DAY1 요약 CSV와 원본 MAT의 셀 수가 달라 DAY2에서는 요약 CSV와 병합하지 않았다.
