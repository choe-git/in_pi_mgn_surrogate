# 논문식 physics, cap 추론, noise 및 seed 수정 보고서

작성일: 2026-07-16

## 1. 수정 전 백업

전체 프로젝트를 다음 형제 디렉터리에 백업했다.

`C:\Users\uno42\OneDrive\Desktop\GNN_surrogate 코드_backup_before_paper_physics_20260716_131602`

- 파일 수: 1,251개
- 전체 크기: 1.17GB

## 2. 수정 파일

- `gnn_surrogate/physics.py`
- `gnn_surrogate/data.py`
- `gnn_surrogate/model.py`
- `scripts/train.py`
- `scripts/run_stability_acceleration_rollout.sh`
- `scripts/run_stability_pi_light.sh`

평가 코드, 모델 구조, 데이터 split 방식과 checkpoint 형식은 변경하지 않았다.

## 3. Physical loss

`physics.py`를 논문 식 (6)-(11)에 맞춘 단일 구현으로 정리했다.

- 식 (6): 이웃 velocity 차이와 relative position의 outer product로 velocity gradient를 계산한다.
- 식 (7), (9): gradient trace로 divergence를 구하고 `mean(abs(divergence))`를 continuity loss로 사용한다.
- 식 (8): velocity 차이가 아니라 이웃 gradient 차이로 vector Laplacian을 계산한다.
- 식 (10): prediction과 ground truth의 `u dot grad(u)` 차이에 대한 node 평균 L2 제곱을 사용한다.
- 식 (11): prediction과 ground truth의 Laplacian 차이에 대한 node 평균 L2 제곱을 사용한다.
- data loss도 논문의 `mean_v ||error_v||_2^2`와 같이 component 합을 먼저 구한 뒤 node 평균을 계산한다.

공개된 light HDF5에는 density와 viscosity field가 없으므로 `rho`와 `mu`는 normalized constant로 취급한다. 상수 배율은 기존 physical loss weight가 담당한다. 논문이 실제 사용한 개별 scaling factor 값은 공개하지 않았으므로 weight argument는 유지했다.

제거한 선택지는 다음과 같다.

- `--physics-operator`
- `--continuity-target`

이제 legacy operator나 ground-truth divergence matching으로 전환할 수 없으며 논문식 operator와 zero-divergence만 사용한다.

## 4. Inlet/outlet 추론

2026-07-15에 추가한 face-flux 방식은 제거하고 이전의 geometric two-cap 방식으로 복귀했다.

1. 전체 trajectory에서 항상 velocity가 0인 node를 wall로 찾는다.
2. tetrahedral mesh에서 한 번만 나타나는 triangle을 outer boundary face로 찾는다.
3. outer boundary node 중 wall이 아닌 node를 open-boundary node로 모은다.
4. open-boundary node의 principal axis를 계산한다.
5. axis 위 좌표를 1차원 2-means로 나누어 두 cap을 만든다.
6. 각 cap의 trajectory 평균 velocity가 혈관 내부 방향을 향하는 정도를 비교해 inlet과 outlet을 정한다.

face orientation, triangle area, flux persistence 및 mass-residual 계산 약 200줄과 derivative 기반 legacy fallback을 제거했다. 두 cap을 만들 수 없는 비정상 case는 다른 규칙으로 조용히 대체하지 않고 명확한 오류를 발생시킨다.

## 5. Velocity augmentation

augmentation 선택 기능을 제거하고 기존 1번 temporal noise만 항상 사용한다.

- `u(t)`와 `u(t-1)`에 correlation을 갖는 Gaussian noise를 넣는다.
- local acceleration은 두 noisy velocity의 차이로 다시 계산되므로 동일한 noise 맥락을 유지한다.
- inflow context에도 하나의 global 3-component Gaussian noise를 넣는다.
- wall과 inlet velocity는 prescribed boundary이므로 직접 변경하지 않는다.

제거한 argument는 다음과 같다.

- `--velocity-augmentations`
- `--spatial-noise-smoothing-steps`
- `--magnitude-jitter-std`

남은 noise 조절값은 `--noise-std`와 `--temporal-noise-correlation`뿐이다. `args.json`에는 고정 방식이 `"velocity_augmentation": "temporal_noise"`로 기록된다.

## 6. Seed

`--seed` argument를 제거하고 모든 실행의 seed를 2026으로 고정했다. 데이터 split, 모델 초기화, timestep shuffle과 augmentation random stream이 같은 seed에서 시작한다. `args.json`과 `split.json`에는 2026이 기록된다.

CUDA 연산 자체의 bitwise deterministic mode는 별도 적용하지 않았으므로 서로 다른 GPU나 CUDA 버전에서는 마지막 bit까지 같다고 보장하지 않는다.

## 7. 유지된 기능

- 95 train / 5 validation / 5 test split
- validation N-RMSE 기반 `best.pt` 선택
- test split의 checkpoint 선택 미사용
- acceleration 3-component decoder output
- 미래의 prescribed inlet velocity clamp와 inflow context
- velocity 및 acceleration 전체 train statistics
- 기존 checkpoint를 사용하는 `scripts/evaluate.py`

## 8. 실행 argument 변화

다음처럼 제거된 argument 없이 실행한다.

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
  --eval-rollout-steps 50 \
  --best-rmse-steps 50 \
  --eval-domain whole \
  --device cuda
```

## 9. 코드 정리 결과

핵심 네 파일의 총 길이는 1,633줄에서 1,223줄로 410줄 감소했다. 활성 기능과 직접 관련 없는 별도 모듈은 추가하지 않았다.

## 10. 검증

- 수정한 Python 파일 네 개의 `py_compile` 통과
- 제거한 함수 및 argument의 활성 코드 참조가 남지 않았음을 `rg`로 확인
- 두 원형 cap을 principal-axis 2-means로 분리하고 유속 방향으로 inlet/outlet을 고르는 synthetic smoke test 통과 (16/16 nodes)
- 로컬 bundled Python에는 PyTorch와 h5py가 없어 tensor backward 및 실제 HDF5 smoke test는 서버 환경에서 수행해야 한다.
