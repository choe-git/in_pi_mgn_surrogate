# Validation loss TensorBoard 기록 수정 보고서

## 목적

매 epoch validation에서 계산 가능한 1-step loss를 TensorBoard와 터미널에 기록한다.

## 수정 파일

- `scripts/train.py`

## 구현 내용

- 기존 학습 loss 계산을 `prediction_loss()`로 분리해 학습과 validation이 같은 정의를 사용하도록 했다.
- validation에서는 noise와 velocity augmentation을 적용하지 않은 원본 `current_u`로 1-step 예측을 수행한다.
- non-PI 모델의 validation loss는 learned node의 velocity MSE다.
- PI 모델의 validation loss는 기존 학습과 동일한 가중합이다: data, continuity, convection, viscosity loss.
- 각 epoch의 모든 validation case와 timestep에서 구한 loss를 평균한다.
- TensorBoard tag `eval/validation_loss`와 터미널 항목 `val_loss`를 추가했다.

## 유지한 동작

- validation 1-step/50-rollout RMSE는 기존대로 wall을 포함한 whole geometry에서 계산한다.
- validation loss는 학습 목적함수와 동일하게 learned node에서 계산한다.
- `best.pt`는 계속 `--best-rmse-steps`에 해당하는 validation RMSE로만 선정한다.
- test split은 validation loss 계산 및 `best.pt` 선정에 사용하지 않는다.
- argument와 파일 구조는 변경하지 않았다.

## 검증

- `scripts/train.py` Python 구문 검사 통과.
- 원본 대비 diff를 확인해 위 변경 이외의 동작 변경이 없음을 확인했다.
