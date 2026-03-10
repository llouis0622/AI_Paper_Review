# 1. Reading Context(사전 맥락)

- 고차원 관측을 직접 입력으로 사용하는 강화학습이 왜 불안정했는지 이해하기 위해
- 값 함수 근사와 심층 신경망 결합에서 발생하는 발산 문제를 구조적으로 해결하기 위해
- 하나의 알고리즘으로 다수의 서로 다른 게임을 학습할 수 있는지 검증하기 위해

# 2. Problem Re-definition

- 상태가 이미지 픽셀로 주어지는 환경에서 수작업 특징 없이 정책을 학습해야 함
- 비선형 근사기를 사용하는 Q-learning은 학습 중 발산 가능성이 큼
- 실무적으로 필요한 핵심 요구는 다음 세 가지
    - 상관된 시계열 샘플로 인한 불안정성 제거
    - 목표 값과 예측 값의 상호 의존 완화
    - 단일 구조로 다양한 태스크 일반화

# 3. Core Contributions(논문의 핵심 기여)

### 심층 신경망을 이용한 Q 함수 근사

- 합성곱 신경망을 사용해 픽셀 입력에서 행동 가치 직접 예측
- 행동별 출력 헤드를 두어 단일 순전파로 모든 행동 가치 계산

### Experience Replay 도입

- 과거 전이 샘플을 메모리에 저장
- 무작위 미니배치 샘플링으로 시계열 상관 제거

### Target Network 도입

- 목표 값을 계산하는 네트워크를 주기적으로만 갱신
- 목표와 예측의 강한 결합을 완화하여 학습 안정화

# 4. Method Analysis(설계 관점)

- 입력
    - 최근 프레임을 전처리 후 스택한 이미지
- 모델
    - 합성곱 계층과 완전연결 계층
- 학습 루프
    - 탐험을 포함한 행동 선택
    - 전이 저장
    - 무작위 미니배치 학습
- 추론
    - 행동 가치가 최대인 행동 선택

# 5. Mathematical Formulation Log

- 할인 누적 보상 정의
    
    $R_t \;=\; \sum_{k=t}^{T} \gamma^{k-t} r_k$
    
- 최적 행동 가치 함수
    
    $Q(s,a) \;=\; \mathbb{E}\bigl[r + \gamma \max_{a'} Q(s',a')\bigr]$
    
- 타깃 값 계산
    
    $y \;=\; r + \gamma \max_{a'} Q_{\text{target}}(s',a')$
    
- 제곱 오차 기반 손실
    
    $\mathcal{L}(\theta) \;=\; \mathbb{E}\bigl[(y - Q(s,a;\theta))^2\bigr]$
    

# 6. Experiment as Claim Verification

- 동일한 구조와 하이퍼파라미터로 49개 Atari 게임 평가
- 다수 게임에서 기존 강화학습 방법 대비 성능 우수
- 인간 전문가 수준에 근접한 평균 성능 달성

# 7. Limitations & Failure Modes

- 장기 계획이 필요한 희소 보상 게임에서 성능 저하
- 대규모 연산 자원 요구
- 연속 행동 공간에는 직접 적용 어려움

# 8. Extension & Research Ideas

- Double DQN, Dueling Network
- Prioritized Experience Replay
- 분포적 가치 학습
- 정책 기반 방법과의 결합

# 9. Code Strategy

- 합성곱 기반 Q 네트워크 정의
- 재현 메모리와 타깃 네트워크 구현
- 미니배치 Q-learning 업데이트
- 간단한 환경에서 동작 검증

# 10. One-Paragraph Research Summary

이 논문은 심층 신경망과 Q-learning을 결합하여, 픽셀 입력만으로 다양한 게임에서 인간 수준의 제어를 달성한 최초의 일반 강화학습 에이전트를 제시한다. 경험 재현과 타깃 네트워크라는 두 가지 구조적 장치를 통해 비선형 근사에서 발생하던 불안정을 해결하였으며, 이는 이후 심층 강화학습 연구 전반의 표준 설계로 자리 잡았다.

# 11. Connection to Other Papers

- Q-learning 계열 이론
- 합성곱 신경망 기반 시각 인식
- 이후 Rainbow DQN 계열 연구
- 정책 경사 및 Actor–Critic 계열로의 확장

# 12. Personal Insight Log

- 핵심은 신경망이 아니라 학습 안정화를 위한 데이터 흐름 설계
- 강화학습에서 표현 학습이 언제 효과적인지 보여준 사례
- 이후 모든 심층 강화학습 구조의 출발점