# TensorBoard eval 즉시 기록 수정 보고서

## 문제

기존 코드는 epoch의 모든 validation 계산을 마친 뒤 scalar 네 개를 한꺼번에 추가하고 `writer.flush()`를 호출했다. 또한 `SummaryWriter`가 출력 디렉터리 생성보다 먼저 초기화되었다.

## 수정

- 출력 디렉터리를 먼저 생성한 다음 `SummaryWriter`를 초기화한다.
- `max_queue=1`을 지정해 train/eval scalar가 writer queue에 오래 머물지 않게 한다.
- 실행 시 실제 log 경로를 `tensorboard_log_dir=...` 형식으로 출력한다.
- 1-step validation 직후 `eval/1_step_rmse`, `eval/validation_loss`를 기록한다.
- 50-step rollout 직후 `eval/50_rollout_rmse`를 기록한다.
- selection metric 결정 직후 `eval/selection_rmse`를 기록한다.
- 명시적 `writer.flush()`를 제거하고 epoch step을 1부터 기록한다.

## 유지 사항

- 학습 loss, validation 계산식, split, 모델, optimizer 및 `best.pt` 선정 기준은 변경하지 않았다.
- 정상 종료 시 `writer.close()`는 유지한다.

## 검증

- `scripts/train.py` Python 구문 검사 통과.
- 원본 대비 diff에서 TensorBoard 기록 시점과 writer 초기화 외 변경이 없음을 확인했다.
