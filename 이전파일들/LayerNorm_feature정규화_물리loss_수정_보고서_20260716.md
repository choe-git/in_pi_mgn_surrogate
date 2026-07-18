# LayerNorm, feature normalization, physics-loss logging 수정 보고서

작성일: 2026-07-16

## 1. 수정 전 백업

수정 전에 전체 프로젝트를 다음 경로에 백업했다.

`C:\Users\uno42\OneDrive\Desktop\GNN_surrogate 코드_backup_before_layernorm_feature_stats_20260716_143910`

- 파일 수: 1,252개
- 전체 크기: 1,260,059,119 bytes

## 2. 수정 목적

기존 기능과 파일 구조를 유지하면서 다음 재현성 및 안정성 문제를 보완했다.

1. 표준 MeshGraphNet에 사용되는 MLP output LayerNorm 누락
2. input acceleration과 inflow context를 velocity 통계로 대신 정규화하던 문제
3. physics loss 각 항의 크기를 TensorBoard에서 확인할 수 없던 문제
4. inflow mean/min/max에 독립 noise를 직접 더해 순서가 깨질 수 있던 문제
5. boundary 추론 결과를 정량적으로 검사하기 어려웠던 문제
6. 동작하지 않거나 현재 고정 정책과 충돌하는 argument와 실행 스크립트

## 3. 수정 파일

- `gnn_surrogate/model.py`
- `gnn_surrogate/data.py`
- `gnn_surrogate/train_utils.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/inspect_boundaries.py`
- `scripts/run_paper_like.sh`
- `scripts/run_tmpenv_paper_like.sh`
- `scripts/submit_tmpenv_full_pbs.sh`
- `scripts/run_stability_acceleration_rollout.sh`
- `scripts/run_stability_noise_sweep.sh`
- `scripts/run_stability_pi_light.sh`
- `scripts/run_stability_in_mgn_data.sh`
- `scripts/setup_tmp_env_and_smoke.sh`

## 4. LayerNorm

새로 학습하는 모델은 decoder를 제외한 다음 MLP의 최종 출력에 LayerNorm을 적용한다.

- node encoder
- edge encoder
- 15개 processor block의 edge-update MLP
- 15개 processor block의 node-update MLP

decoder는 물리량인 3-component acceleration을 출력하므로 LayerNorm을 적용하지 않는다.

새 checkpoint의 `args.json`과 checkpoint에는 `"use_layer_norm": true`가 기록된다.

기존 checkpoint에는 이 항목이 없으므로 `evaluate.py`는 자동으로 `use_layer_norm=False` 모델을 구성한다. 따라서 기존 `best.pt`도 기존 구조 그대로 평가할 수 있다. 새 모델과 기존 모델의 weight를 서로 섞어 불러오지는 않는다.

## 5. Feature별 normalization

새 학습은 training split만 사용해 다음 통계를 별도로 계산한다.

| Feature | 적용 통계 |
| --- | --- |
| current velocity | velocity mean/std |
| input acceleration `u(t)-u(t-1)` | input-acceleration mean/std |
| inflow mean/min/max | 각 context component의 mean/std |
| decoder output acceleration `u(t+1)-u(t)` | output-acceleration mean/std |

새 파일 `feature_stats.json`에는 input acceleration과 inflow context 통계가 저장된다. 동일 통계는 checkpoint의 `feature_stats`에도 포함되어 evaluation과 rollout에서 그대로 사용된다.

기존 checkpoint에 `feature_stats`가 없으면 기존 방식인 acceleration/velocity-std 및 inflow/velocity-std-norm으로 자동 복원한다.

통계는 기본적으로 95개 training case의 모든 학습 timestep을 사용한다. validation과 test 데이터는 통계 계산에 포함되지 않는다.

## 6. Training noise

기존 temporal Gaussian noise는 유지했다.

- dynamic node의 `u(t)`와 `u(t-1)`에 correlated noise 적용
- noisy velocity 차이에서 input acceleration 재계산
- wall과 inlet velocity boundary 자체는 고정

inflow context는 mean/min/max 숫자에 독립 noise를 직접 더하지 않는다. 대신 t+1 inlet velocity field에 Gaussian perturbation을 적용한 뒤 speed의 mean/min/max를 다시 계산한다. 따라서 항상 다음 관계가 유지된다.

`minimum <= mean <= maximum`

실제 inlet boundary clamp에는 원래 ground-truth prescribed velocity를 사용하므로 boundary condition은 바뀌지 않는다.

## 7. Physics-loss 기록

기존 논문식 physical operator와 기본 가중치는 유지했다.

- data: 0.5
- continuity: 1/6
- convection: 1/6
- viscosity: 1/6

TensorBoard에 다음 epoch 평균 scalar를 추가했다.

- `train_epoch/weighted_total`
- `train_epoch/data`
- `train_epoch/continuity`
- `train_epoch/convection`
- `train_epoch/viscosity`
- `train_epoch/rmse`

각 epoch 종료 시 동일한 항별 평균을 terminal에도 출력하며 `writer.flush()`를 수행한다.

논문은 초기 epoch의 각 loss 크기를 관찰해 scaling factor를 조정했다고 설명하지만 정확한 보정값은 공개하지 않았다. 따라서 근거 없는 자동 가중치 변경은 하지 않았다. 첫 정식 run의 위 scalar들을 비교한 뒤 필요한 경우 가중치만 조정해야 한다.

## 8. Boundary 추론 및 진단

기존 geometric two-cap 규칙은 유지했다.

1. 전체 trajectory에서 zero-velocity wall 추론
2. tetrahedral outer boundary 추출
3. non-wall open-boundary node 수집
4. principal axis와 1-D two-means로 두 cap 분리
5. 평균 velocity의 inward projection으로 inlet/outlet 결정

각 cap은 최소 3개 node를 가져야 한다. 조건을 만족하지 않으면 잘못된 cap을 조용히 사용하지 않고 기존 오류 경로로 전달한다.

`inspect_boundaries.py`에는 다음 진단을 추가했다.

- outer-boundary node 수
- HDF5 raw wall mask와 zero-velocity wall의 Jaccard score
- wall/inlet/outlet node 수와 centroid
- t=1 mean speed
- cap circularity와 planarity

또한 한 case의 trajectory를 wall과 cap별로 반복해서 읽던 부분을 단일 velocity cache로 통합했다. 추론 규칙과 결과 식은 바뀌지 않고 HDF5 I/O만 감소한다.

## 9. Argument 및 실행 스크립트 정리

학습 CLI에서 다음 항목을 제거했다.

- `--boundary-percentile`: 실제 계산에 사용되지 않음
- `--output-scale`: acceleration 출력으로 고정
- `--acceleration-mode`: `u(t)-u(t-1)` acceleration으로 고정
- `--num-workers-note`: 실행 동작이 없는 metadata 문자열
- train의 `--max-test-cases` alias: `--max-val-cases`만 사용

checkpoint metadata에는 호환성 확인을 위해 다음 고정값을 계속 기록한다.

- `output_scale = acceleration`
- `acceleration_mode = acceleration`
- `seed = 2026`

모든 정식 실행 스크립트의 `--stats-samples 128`을 `--stats-samples all`로 수정했다. smoke test만 빠른 실행을 위해 1 sample을 유지한다.

## 10. 재현성

seed 2026 고정에 더해 다음 설정을 적용했다.

- cuDNN deterministic mode
- cuDNN benchmark 비활성화
- CUBLAS workspace configuration 고정
- PyTorch deterministic algorithms `warn_only`

CUDA의 `index_add` 계열 연산은 환경에 따라 완전한 bitwise 결정성을 제공하지 않을 수 있으므로 `warn_only`로 기록한다. 같은 서버와 software stack에서는 기존보다 재현성이 강화되지만 서로 다른 GPU/CUDA 조합의 완전 동일 결과를 보장하지는 않는다.

## 11. Split과 checkpoint 선택

다음 동작은 변경하지 않았다.

- 95 train / 5 validation / 5 unseen test
- seed 2026 기반 split을 `split.json`과 checkpoint에 저장
- `best.pt`는 validation 50-RMSE로만 선택
- test split은 통계, 학습, validation, `best.pt` 선택에 사용하지 않음
- 50-RMSE는 논문 식처럼 rollout 1~50 step RMSE의 시간 평균

## 12. 검증

- 수정 Python 파일 `py_compile` 통과
- 모든 shell script `bash -n` 통과
- 제거된 학습 argument의 활성 참조가 없음을 `rg`로 확인
- 새 checkpoint의 LayerNorm/feature-stats 경로와 기존 checkpoint fallback 경로를 정적 확인
- 16+16 node synthetic two-cap 분리 통과
- synthetic 80-frame zero-velocity wall 추론 통과

로컬 bundled Python에는 PyTorch와 h5py가 없으므로 실제 tensor forward/backward 및 HDF5 end-to-end smoke test는 서버 환경에서 수행해야 한다.

## 13. 권장 학습 명령

```bash
PYTHONPATH=$PWD python scripts/train.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --output-dir runs/in_pi_mgn \
  --model-variant in-pi-mgn \
  --epochs 20 \
  --message-passing-steps 15 \
  --lr 1e-4 \
  --lr-decay-start-epoch 16 \
  --lr-min 1e-7 \
  --noise-std 0.003 \
  --temporal-noise-correlation 0.8 \
  --val-count 5 \
  --test-count 5 \
  --eval-rollout-steps 50 \
  --best-rmse-steps 50 \
  --eval-domain whole \
  --device cuda
```

LayerNorm과 feature normalization이 달라졌으므로 성능 비교에는 새로 학습한 checkpoint를 사용해야 한다.
