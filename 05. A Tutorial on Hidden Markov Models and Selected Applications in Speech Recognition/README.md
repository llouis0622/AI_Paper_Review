# 1. Reading Context(사전 맥락)

- 시퀀스 데이터를 확률적으로 모델링하는 표준 틀을 정의-문제-해법까지 한 번에 잡기 위해
- HMM에서 실제로 해결해야 하는 문제는 정확히 무엇이고, 각각 어떤 알고리즘이 존재하는가
- HMM은 단순 마르코프 체인 확장으로만 생각하기 쉬운데, 실제 핵심은 어떤 것 인가

# 2. Problem Re-definition

- 관측 시퀀스가 있고, 그 원인이 되는 상태 시퀀스는 보이지 않음, 숨은 상태 때문에 직접 최적화/추론이 어려움
- HMM에서 실무적으로 필요한 건 세가지
    - Evaluation : 주어진 모델이 관측을 얼마나 잘 설명하는가
    - Decoding : 관측을 만든 가장 그럴듯한 상태열은 무엇인가
    - Learning : 관측으로부터 모델 파라미터를 어떻게 추정하는가
- HMM은 숨은 상태를 둔 시퀀스 생성 모델이고, 핵심은 확률 계산을 효율적으로 만드는 동적계획법 + EM

# 3. Core Contributions(논문의 핵심 기여)

### HMM의 표준 정의를 정리해 주는 튜토리얼 표준형

- HMM을 $\lambda = (A, B, \pi)$로 정리
    - $A$ : 상태 전이 확률
    - $B$ : 관측 방출 확률
    - $\pi$ : 초기 상태 분포
- Markov chain과 대비되는 “숨은 상태 + 관측 방출” 구조를 명확히 설명

### “세 가지 기본 문제”를 기준으로 알고리즘을 1:1로 매핑

- Evaluation ↔ Forward
- Decoding ↔ Viterbi
- Learning ↔ Baum-Welch

### 음성인식 적용을 위한 실전적 확장

- 음성은 시간 방향으로 진행되는 특성 때문에 Left-to-right 구조가 자주 쓰임
- 관측이 연속 벡터인 경우 $B$를 mixture Gaussian 등으로 확장하는 방법 제시
- 실제 시스템 블록 다이어그램까지 연결해 모형 → 시스템 관점 생성

# 4. Method Analysis(설계 관점)

- Input : 관측 시퀀스 $O = (O_1, \cdots, O_T)$
- Hidden : 상태 시퀀스 $Q = (q_1, \cdots, q_T)$
- Generative story
    - $q_1 \sim \pi$
    - $q_{t = 1} \sim A_{q_{t^+}}$
    - $O_t \sim B_{q_t}(\cdot)$
- 완전연결 vs Left-to-right 같은 제약 구조가 계산과 표현력을 결정
- 음성인식에서는 발화가 시간 진행을 갖는다는 가정 때문에 Left-to-right가 실용적이라는 논의가 등장

# 5. Mathematical Formulation Log

- Evaluation 목표
    - $P(O \mid \lambda)$를 효율적으로 계산
    - 단순 합은 지수폭발 → Forward로 $O(N^2T)$ 정도로 줄임
- Decoding 목표
    - $\argmax_Q P(Q \mid O, \lambda)$ 또는 동치 형태
    - Viterbi로 최적 경로 DP
- Learning 목표
    - $\lambda$를 데이터에 맞게 추정
    - 상태/전이의 기대 카운트를 구해 재추정하는 Baum-Welch
    - $\gamma, \zeta$ 같은 posterior 기대량을 이용해 $A, B, \pi$ 업데이트를 구성
- Continuous density 확장
    - $B$를 mixture Gaussian 등으로 두고, EM의 M-step이 mixture 파라미터에도 적용됨

# 6. Experiment as Claim Verification

- 검증의 핵심은 “세 문제 각각에 대해, 지수폭발을 DP/EM으로 해결한다”는 절차의 일관성
- 후반부에 음성인식 시스템 블록도를 통해 “HMM이 실제 시스템에서 어디에 들어갔는가”를 보여줌

# 7. Limitations & Failure Modes

- 1차 마르코프 가정
- 관측 조건부 독립 가정
- 상태 지속시간 모델링의 한계 → 이후 explicit duration HMM/HSMM 등의 동기 제공
- 음석인식에서의 분리 : 음향모델만으로는 언어모델, 발음사전 등과 결합이 필요

# 8. Extension & Research Ideas

- HMM → HSMM, factorial HMM, hierarchical HMM
- HMM의 $B$를 더 강하게 만들기 : GMM-HMM에서 DNN-HMM로 확장 아이디어
- “평가/디코딩/학습” 3문제 프레임을 그대로 유지한 채, 각 파트를 현대화

# 9. Code Strategy

- Discrete HMM로 Forward 구현해 $P(O \mid \lambda)$ 계산
- Viterbi로 최적 상태열 추정
- Baum-Welch로 $A, B, \pi$ 재추정
- 장난감 시퀀스로 학습 후 로그우도 증가 확인
- ground-truth HMM을 만들어 샘플링하고, 학습이 원모형에 수렴하는지 관찰

# 10. One-Paragraph Research Summary

이 논문은 HMM을 $(A, B, \pi)$로 정식화하고, 관측 시퀀스에 대해 우도 평가, 최적 상태열 디코딩, 파라미터 학습이라는 세 가지 핵심 문제로 분해하여 각각의 표준 해법을 제시한다. 또한 Left-to-right 구조와 연속 관측 등 음성인식에 필요한 실전 확장을 포함해 확률 모델 → 실제 인식 시스템으로 연결되는 튼튼한 기반을 제공한다.

# 11. Connection to Other Papers

- 고전 확률 모델 계열 : Markov chain, EM 계열 학습
- 음성 인식 실전 흐름 : GMM-HMM → DNN-HMM → end-to-end로 넘어가도, 정렬/디코딩/학습 관점은 계속 살아 있음
- 시퀀스 모델 일반화 : HMM의 구조적 아이디어가 CRF 등 판별 모델로도 연결

# 12. Personal Insight Log

- HMM의 진짜 힘은 모델이 아니라 문제를  3개로 쪼개고, 각각을 계산 가능한 형태로 만든 설계력
- 딥러닝 이전 시대에도 학습 + 추론으로 시퀀스를 다뤘고, 현대 모델들은 이 틀을 다른 방식으로 대체/흡수
- 이 논문은 evaluation/decoding/learning 중 어디를 어떻게 바꾸는가 관점으로 해석