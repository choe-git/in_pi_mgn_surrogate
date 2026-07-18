# Inlet/Outlet 유량 기반 추론 수정 보고서

작성일: 2026-07-15

## 1. 이번 수정의 범위

이번 작업은 별도 inlet 추론 모듈을 새로 추가하는 방식이 아니라, 직전 백업의 코드로 복원한 뒤 기존 `gnn_surrogate/data.py` 내부만 보수적으로 수정했다.

- 실제 수정 코드: `gnn_surrogate/data.py` 한 파일
- 변경하지 않은 코드: 학습, 평가, 모델, loss, augmentation, split, statistics 관련 코드 전체
- 새 dependency: 없음
- 새 train/evaluate argument: 없음
- 유지한 공개 함수: `infer_open_boundary_masks(...)`, `infer_node_types(...)`
- 유지한 최종 fallback: `find_nonzero_derivate(...)`

앞서 만든 별도 `boundary_inference.py`, audit script, test 파일은 현재 프로젝트에서 제거했다. 검증용 파일은 작업공간 밖의 임시 staging 위치에서만 사용했으며 프로젝트에는 배포하지 않았다.

## 2. 백업과 복원 기준

수정 전 기준은 다음 직전 백업이다.

`이전파일들/backup/GNN_surrogate 코드_backup_before_face_flux_inlet_20260715_201634`

작업 시작 전에 현재 `gnn_surrogate/data.py`와 위 백업의 SHA-256이 일치하는 것을 확인했다. 따라서 이번 수정은 직전 코드에서 시작한 변경이다.

## 3. 기존 방식의 핵심 문제

기존 방식은 다음 순서였다.

1. boundary node 중 wall이 아닌 node를 모두 모은다.
2. 전체 open-boundary node에 대해 하나의 principal axis를 계산한다.
3. 그 축 위 좌표를 2개 군집으로 나누어 두 cap을 만든다.
4. 각 군집의 평균 velocity와 전역 축의 내적을 비교해 inlet을 고른다.

이 방식은 직선에 가까운 혈관에서는 작동하지만, 굽은 혈관에서는 각 cap의 실제 법선 방향과 전역 principal axis가 달라질 수 있다. 또한 node 개수만 반영하고 boundary face 면적은 반영하지 않아 물리적인 유량 판정과 정확히 같지 않다.

## 4. 새 기본 방식

### 4.1 외향 boundary face 법선 복원

각 tetrahedron의 4개 삼각형 면을 열거한 뒤, 전체 mesh에서 한 번만 등장하는 면을 boundary face로 선택한다.

boundary face마다 해당 tetrahedron의 반대편 node를 찾는다. 삼각형 법선이 반대편 node, 즉 tetrahedron 내부를 향하면 법선 부호를 뒤집는다. 이 과정을 거치면 tetrahedron 정점 저장 순서와 무관하게 모든 boundary face 법선이 mesh 바깥을 향한다.

### 4.2 열린 cap 분리

boundary face 중 적어도 하나의 non-wall node를 포함한 면을 open-face 후보로 사용한다. 후보 face가 완전한 edge를 공유하면 같은 연결 성분으로 묶는다.

연결 성분은 삼각형 면적의 합이 큰 순서로 정렬하고, 현재 데이터의 단일 inlet/단일 outlet 구조에 맞춰 가장 큰 두 성분을 두 cap으로 사용한다. 작은 고립 성분은 mesh 또는 wall-mask 경계의 미세 조각으로 취급한다.

### 4.3 면적 가중 signed flux 계산

각 cap과 각 timestep에서 다음 값을 계산한다.

`Q(t) = sum_f A_f * mean(u_f(t)) dot n_f`

- `A_f`: boundary triangle 면적
- `mean(u_f(t))`: triangle의 세 node velocity 평균
- `n_f`: mesh 바깥을 향하는 unit normal

외향 법선을 사용하므로 부호 해석은 직접적이다.

- `mean(Q) < 0`: 유체가 mesh 안으로 들어오므로 inlet
- `mean(Q) > 0`: 유체가 mesh 밖으로 나가므로 outlet

즉, 기존처럼 전역 축과 평균 velocity를 간접 비교하지 않고 실제 cap을 통과하는 면적 가중 유량으로 역할을 정한다.

### 4.4 신뢰도 검사와 fallback

물리 판정이 애매한 데이터에서 잘못된 결과를 강제하지 않도록 두 검사를 추가했다.

1. 유량이 충분히 큰 timestep에서 inlet/outlet 부호 지속성이 각각 0.8 이상이어야 한다.
2. 두 cap의 평균 상대 질량수지 오차가 0.2 이하여야 한다.

두 cap이 검출되지 않거나, 한 cap은 음수이고 다른 cap은 양수라는 조건이 성립하지 않거나, 위 신뢰도 검사를 통과하지 못하면 직전 코드의 principal-axis 방식으로 돌아간다. 그 결과마저 비어 있으면 기존 `find_nonzero_derivate(...)` fallback이 그대로 작동한다.

## 5. 코드 구조

모든 새 보조 함수는 기존 `data.py` 안에 있고 한 방향으로만 호출된다.

1. `_oriented_boundary_faces(...)`: boundary face, 외향 normal, area 계산
2. `_connected_face_components(...)`: edge 기반 face 연결 성분 계산
3. `_open_cap_components(...)`: 가장 큰 두 open cap 선택
4. `_cap_flux(...)`: timestep별 signed flux 계산
5. `_flux_sign_persistence(...)`: 유량 부호 신뢰도 계산
6. `infer_open_boundary_masks(...)`: 위 결과를 조합해 inlet/outlet mask 반환

기존 구현은 `_infer_open_boundary_masks_geometric(...)`으로 이름만 바꾸어 같은 파일 안에 fallback으로 유지했다. `infer_node_types(...)`의 호출부와 반환 형식은 바뀌지 않았다.

## 6. 실제 데이터 검증 결과

세 HDF5 사례 모두 새 유량 방식이 성공해 geometric fallback은 호출되지 않았다.

| Case | Wall nodes | Inlet nodes | Outlet nodes | 기존 Inlet IoU | 기존 Outlet IoU | Flux sign persistence | 평균 질량수지 오차 | 최대 질량수지 오차 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MESH_1 | 4444 | 176 | 182 | 0.994 | 1.000 | 1.000 / 1.000 | 0.0032 | 0.0053 |
| MESH_11 | 7385 | 193 | 185 | 0.995 | 0.995 | 1.000 / 1.000 | 0.0036 | 0.0058 |
| MESH_117 | 6095 | 184 | 170 | 1.000 | 1.000 | 1.000 / 1.000 | 0.0035 | 0.0055 |

IoU가 1보다 조금 작은 사례는 기존 PCA 군집의 cap 경계 node 1개 정도가 face-connectivity 기준과 달랐기 때문이다. 두 방식의 전체적인 cap 위치는 거의 동일하지만, 새 방식은 방향 결정에 실제 외향 법선과 면적 가중 flux를 사용한다.

## 7. 합성 검증

직선 원통형 tetrahedral mesh에서 다음 네 검사를 수행했고 모두 통과했다.

1. 정방향 유동에서 왼쪽 cap을 inlet, 오른쪽 cap을 outlet으로 판정
2. velocity 부호를 뒤집으면 inlet/outlet이 서로 바뀜
3. tetrahedron 정점 순서를 바꿔도 결과가 유지됨
4. 계산한 모든 boundary normal이 반대편 tetrahedron node를 향하지 않음

테스트 중 geometric fallback이 호출되면 강제로 실패하도록 구성했으므로, 위 결과는 새 flux 경로 자체의 결과다.

## 8. 실행 방법과 영향

학습 및 평가 명령은 이전과 완전히 동일하다. inlet/outlet mask는 기존처럼 data cache를 처음 만들 때 자동 계산된다. 별도의 option이나 argument를 넣을 필요가 없다.

기존 방식은 두 cap의 평균 velocity를 계산하면서 trajectory를 cap별로 반복해서 읽었다. 새 방식은 trajectory를 한 번 읽어 두 cap flux를 함께 계산하므로, 새 물리 검사를 추가했지만 velocity 읽기 횟수는 오히려 줄었다.

## 9. 남아 있는 가정

현재 데이터와 논문의 재현 조건에 맞춰 open cap 수를 2개, 즉 inlet 1개와 outlet 1개로 가정한다. 실제 다분지 혈관처럼 outlet이 여러 개인 mesh에 일반화하려면 cap 수를 고정하지 않고 모든 유효 연결 성분을 보존하는 별도 확장이 필요하다. 이번 수정에서는 기존 데이터 구조와 동작 범위를 유지하기 위해 그 확장을 포함하지 않았다.
