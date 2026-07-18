# Whole geometry RMSE 단일화 및 성능 저하 분석 보고서

작성일: 2026-07-17  
대상 프로젝트: `GNN_surrogate 코드`  
기준 백업: `GNN_surrogate 코드_backup_before_whole_rmse_only_20260717_120641`

## 1. 수정 목적

평가 결과에서 `whole`과 `fluid` RMSE가 동시에 출력되어 해석이 갈리는 문제를 제거했다. 앞으로 학습 중 validation, 별도 validation 평가, 완전 미사용 test 평가 모두 wall, inlet, outlet, interior를 포함한 전체 geometry RMSE만 계산하고 출력한다.

이번 수정은 평가 도메인과 출력만 정리한다. 모델 구조, 학습 loss, feature normalization, noise, velocity augmentation, inlet inference, split 생성, `best.pt` 선택 방식은 변경하지 않는다.

## 2. 백업

수정 전 전체 프로젝트를 다음 폴더에 복사했다.

`C:\Users\uno42\OneDrive\Desktop\GNN_surrogate 코드_backup_before_whole_rmse_only_20260717_120641`

백업 확인 결과는 6,189개 파일, 총 6,226,269,261 bytes이다.

## 3. 코드 변경

### `scripts/train.py`

- `--eval-domain` argument를 제거했다.
- 기존 실행 기록과 의미가 분명하도록 `args.eval_domain = "whole"`은 고정 metadata로만 저장한다.
- validation one-step RMSE와 rollout RMSE에 전체 노드 mask만 사용한다.
- TensorBoard에는 학습 중 `train/*`, epoch 평균 `train_epoch/*`, validation의 `eval/1_step_rmse`, `eval/{N}_rollout_rmse`, `eval/selection_rmse`를 함께 기록한다.
- `best_rmse_steps=50`이면 `eval_rollout_steps` 값과 관계없이 매 validation epoch의 `eval/50_rollout_rmse`가 생성된다.
- evaluation 값은 계산 직후 `flush()`하여 TensorBoard에 가능한 즉시 반영한다.
- `best.pt`는 기존과 동일하게 validation의 `best_rmse_steps` rollout RMSE로 선택하되, 그 RMSE는 항상 전체 geometry 기준이다.
- test split은 학습 및 `best.pt` 선택 과정에서 읽거나 사용하지 않는다.

### `scripts/evaluate.py`

- `fluid_node_mask` import와 fluid 전용 계산을 제거했다.
- one-step과 rollout 함수는 전체 geometry RMSE 한 값만 반환한다.
- 콘솔 출력은 case별 `1-RMSE-whole`, `{N}-RMSE-whole`과 전체 평균만 남긴다.
- `--split val`은 저장된 validation 파일을, `--split test`는 저장된 unseen test 파일을 사용한다.

### 실행 셸 스크립트

다음 스크립트에서 더 이상 존재하지 않는 `--eval-domain whole`을 제거했다.

- `scripts/run_paper_like.sh`
- `scripts/run_stability_acceleration_rollout.sh`
- `scripts/run_stability_in_mgn_data.sh`
- `scripts/run_stability_noise_sweep.sh`
- `scripts/run_stability_pi_light.sh`
- `scripts/run_tmpenv_paper_like.sh`
- `scripts/submit_tmpenv_full_pbs.sh`

## 4. 출력 예시

평가 시 다음처럼 whole 결과만 나온다.

```text
case_name: 1-RMSE-whole=..., 50-RMSE-whole=...
Mean 1-RMSE-whole: ...
Mean 50-RMSE-whole: ...
```

`RMSE-fluid` 출력은 더 이상 없다.

## 5. 기존 checkpoint 호환성

기존 `best.pt`와 `last.pt`를 그대로 평가할 수 있다. checkpoint의 모델 flag, feature stats, split 정보 처리 방식은 바꾸지 않았다. 단, 같은 checkpoint를 새 `evaluate.py`로 평가하면 예전처럼 fluid-only 수치를 함께 보여주지 않고 whole 수치만 보여준다.

## 6. 이전 augmentation 1 결과보다 나빠진 이유

현재 코드와 최초 velocity augmentation 1 코드 사이에는 augmentation 외에도 여러 변경이 동시에 존재한다. 따라서 결과 저하를 augmentation 1 자체의 문제로 단정할 수 없다.

가능성이 높은 원인은 다음과 같다.

1. **Acceleration 전용 정규화의 스케일 변화**  
   과거에는 입력 acceleration `u(t)-u(t-1)`을 velocity std로 나눴지만, 현재는 acceleration 자체의 std로 나눈다. 일반적으로 acceleration std가 velocity std보다 작으므로 acceleration feature와 그 안의 noise가 이전보다 크게 표현될 수 있다. 이 비율이 지나치게 크면 장기 rollout에서 오차가 누적될 가능성이 가장 높다.

2. **Inflow context noise의 분포 변화**  
   현재는 inlet의 다음 velocity field를 perturb한 뒤 mean/min/max를 다시 계산한다. 특히 min은 작은 노이즈와 inlet mask 오분류에 민감하다. 학습에서는 noisy context, 평가에서는 clean context를 사용하므로 train/eval 분포 차이가 커질 수 있다.

3. **LayerNorm 추가에 따른 최적화 변화**  
   현재 MLP에는 decoder를 제외하고 LayerNorm이 추가되었다. 같은 learning rate, noise std, epoch 수가 이전 구조에서도 현재 구조에서도 동시에 최적이라는 보장은 없다. 20 epoch에서는 새 구조가 충분히 수렴하지 않았을 수 있다.

4. **Physical loss 항의 상대 크기**  
   physical loss 수식을 논문에 가깝게 바꾸었지만 data, continuity, convection, viscosity의 원시 크기는 서로 다를 수 있다. 가중치를 적용한 뒤에도 physical 항이 data loss보다 크면 one-step data fitting과 50-step rollout이 나빠질 수 있다. TensorBoard의 `train_epoch/data`, `continuity`, `convection`, `viscosity`, `weighted_total`을 비교해야 한다.

5. **Split과 평가 표본 차이**  
   현재 split은 seed 2026으로 train 95, validation 5, test 5를 저장한다. test가 5 case뿐이므로 난도가 높은 geometry가 일부 들어가면 평균 50-RMSE가 크게 달라질 수 있다. 서로 다른 `split.json`의 결과는 모델 변경 효과와 직접 비교하면 안 된다.

6. **Heuristic inlet inference 오차**  
   cap 분할 방식은 이전보다 구조적이지만 원 논문의 비공개 inlet labeling과 완전히 같다고 보장할 수 없다. 잘못된 cap 또는 inlet node는 경계조건과 inflow context 양쪽에 영향을 주므로 장기 rollout에 큰 오차를 만들 수 있다.

## 7. 권장 확인 순서

먼저 코드를 다시 크게 바꾸지 말고 다음을 같은 `split.json`, seed, 학습 argument로 확인하는 것이 안전하다.

1. `feature_stats.json`에서 acceleration std와 velocity std의 비율을 확인한다.
2. TensorBoard에서 각 physical loss의 크기와 weighted total 내 비중을 확인한다.
3. validation 5 case와 test 5 case의 case별 50-RMSE를 확인해 특정 geometry가 평균을 지배하는지 본다.
4. inlet diagnostics로 각 case가 정확히 두 cap으로 나뉘고 inlet node 수가 합리적인지 확인한다.
5. 위 확인 후 단일 요인 ablation을 수행한다. 우선순위는 inflow-context noise 제거, acceleration noise scale 제한, LayerNorm 비교, data-only 비교 순서가 적절하다.

## 8. 검증

- `scripts/train.py`, `scripts/evaluate.py`: Python compile 통과
- `scripts/*.sh`: Bash syntax 검사 통과
- active scripts 내 `--eval-domain`, `RMSE-fluid`, `fluid_node_mask`, `eval/fluid` 참조 없음
- TensorBoard scalar tag: train loss와 validation 1-step/선택 rollout RMSE 동시 기록

실제 PyTorch/CUDA 학습 및 50-step evaluation은 이 로컬 환경에 학습용 의존성과 데이터 실행 환경이 없어 수행하지 않았다.
