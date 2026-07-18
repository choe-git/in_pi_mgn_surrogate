# Acceleration naming / evaluation update report

작성일: 2026-07-09

## 목적

이번 수정은 이전 코드에서 `delta`라고 부르던 velocity update를 논의한 기준대로 `acceleration`으로 정리하는 것이다.

여기서 `acceleration`은 논문/Graph Network Simulator 계열에서 흔히 쓰는 다음 step update,

```text
u(t + 1) - u(t)
```

를 뜻한다.

기존의 `(u(t) - u(t - 1)) / dt` 의미가 필요할 때는 `physical_acceleration`이라고 부르도록 분리했다.

## 백업

수정 전 전체 프로젝트 백업:

```text
C:\Users\uno42\OneDrive\Desktop\GNN_surrogate 코드_backup_before_acceleration_rename_20260709_013033
```

## 바뀐 파일

```text
gnn_surrogate/data.py
gnn_surrogate/model.py
gnn_surrogate/train_utils.py
scripts/train.py
scripts/evaluate.py
scripts/run_stability_acceleration_rollout.sh
scripts/run_stability_delta_rollout.sh
scripts/run_stability_in_mgn_data.sh
scripts/run_stability_pi_light.sh
scripts/run_stability_noise_sweep.sh
```

## 핵심 변경 1: target_delta -> target_acceleration

`GraphSample`의 target update 필드를 다음처럼 바꿨다.

```text
target_delta        -> target_acceleration
target_acceleration = target_u - current_u
```

모델 decoder output dim은 그대로 3이다. 즉 모델은 여전히 각 node의 3D velocity update를 예측한다.

## 핵심 변경 2: argument 이름 변경

새로 권장하는 argument는 다음이다.

```text
--output-scale acceleration
--acceleration-mode acceleration
```

의미는 다음과 같다.

```text
--output-scale velocity
  decoder output을 velocity std 기준으로 복원한다.

--output-scale acceleration
  decoder output을 target_acceleration = target_u - current_u 통계로 복원한다.

--acceleration-mode acceleration
  In-MGN 입력 feature에 u(t) - u(t-1)을 넣는다.

--acceleration-mode physical_acceleration
  In-MGN 입력 feature에 (u(t) - u(t-1)) / dt를 넣는다.
```

호환성을 위해 예전 `--accel-mode` 이름과 예전 값은 내부적으로 새 이름으로 정규화된다. 새로 학습한 run의 `args.json`에는 새 이름이 저장된다.

## 핵심 변경 3: statistics 기본값을 전체 데이터로 변경

기존 기본값:

```text
--stats-samples 128
```

새 기본값:

```text
--stats-samples all
```

또는 argument를 생략해도 전체 train index를 사용한다.

부분 샘플 통계가 필요하면 여전히 다음처럼 제한할 수 있다.

```text
--stats-samples 512
--stats-sampling uniform
```

## 핵심 변경 4: whole geometry RMSE 추가

`scripts/evaluate.py`는 이제 두 metric을 동시에 출력한다.

```text
1-RMSE-whole
1-RMSE-fluid
50-RMSE-whole
50-RMSE-fluid
```

`whole`은 wall, interior, inlet, outlet을 모두 포함한다.

`fluid`는 기존처럼 wall을 제외한 metric이다.

학습 중 validation/best checkpoint 선택에도 `--eval-domain`을 추가했다.

```text
--eval-domain whole
--eval-domain fluid
```

기본값은 `whole`이다. 논문 표의 "whole geometry RMSE"와 맞춰 보려면 `whole`을 사용하면 된다.

## 새 추천 학습 명령

기존 명령에서 바꿔야 할 부분까지 포함한 추천 실행 예시는 다음이다.

```bash
PYTHONPATH=$PWD python scripts/train.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --output-dir runs/stability_acceleration_rollout \
  --model-variant in-pi-mgn \
  --epochs 20 \
  --message-passing-steps 15 \
  --lr 1e-4 \
  --lr-decay-start-epoch 16 \
  --lr-min 1e-7 \
  --noise-std 0.006 \
  --stats-samples all \
  --output-scale acceleration \
  --acceleration-mode acceleration \
  --physics-operator gradient \
  --continuity-target zero \
  --data-loss-weight 0.5 \
  --continuity-weight 0.1666666667 \
  --convection-weight 0.1666666667 \
  --viscosity-weight 0.1666666667 \
  --eval-rollout-steps 50 \
  --selection-metric rollout \
  --eval-domain whole \
  --device cuda
```

## 새 추천 평가 명령

평가 argument는 checkpoint 경로만 정확히 지정하면 된다.

```bash
PYTHONPATH=$PWD python scripts/evaluate.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --checkpoint runs/stability_acceleration_rollout/YYYYMMDD_HHMMSS/best.pt \
  --rollout-steps 50 \
  --device cuda
```

출력에서 논문식 비교는 `MEAN 50-RMSE-whole`을 보면 되고, 이전 실험과의 연속 비교는 `MEAN 50-RMSE-fluid`를 보면 된다.

## 새 실행 스크립트

대표 스크립트를 새 이름으로 추가했다.

```bash
bash scripts/run_stability_acceleration_rollout.sh
```

이 스크립트는 다음 조합을 사용한다.

```text
model_variant: in-pi-mgn
output_scale: acceleration
acceleration_mode: acceleration
stats_samples: all
physics_operator: gradient
selection_metric: rollout
eval_domain: whole
```

## 주의

이번 변경은 기존 checkpoint를 이어서 학습하는 기능을 추가하지 않았다. `scripts/train.py`는 여전히 실행할 때마다 새 모델을 초기화한다.

run마다 결과가 달라질 수 있는 이유는 checkpoint resume 때문이 아니라, CUDA 연산 비결정성, training order shuffle, initialization seed 적용 범위, `index_add_` 계열 연산 등의 영향일 가능성이 더 크다.
