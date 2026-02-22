# 1. Reading Context(사전 맥락)

- 언어 모델의 크기 증가가 사용자 의도 충족으로 직결되지 않는 문제를 해결하기 위함
- 언어 모델의 실패 원인을 단순 성능 부족이 아닌 목적 함수 불일치로 재정의
- 인간의 선호를 학습 신호로 직접 사용하는 정렬 기법의 실증적 검증이 핵심

# 2. Problem Re-definition

- 기존 언어 모델의 목표는 다음 토큰 예측
- 실제 사용 환경의 목표는 지시 따르기, 사실성 유지, 유해성 회피
- 두 목표는 구조적으로 불일치
- 문제의 핵심은 모델 성능이 아니라 목적 함수의 정렬 실패
- 인간이 원하는 출력을 직접 학습 신호로 삼는 방법이 필요

# 3. Core Contributions(논문의 핵심 기여)

### Instruction Following을 별도 학습 목표로 정식화

- 언어 모델을 일반 생성기가 아닌 지시 수행 에이전트로 재정의
- 인간이 원하는 행동을 명시적으로 학습 대상에 포함

### RLHF 파이프라인의 대규모 실증

- 지도 학습, 보상 모델, 강화 학습의 단계적 결합
- 단일 기법이 아닌 절차 전체가 성능을 좌우함을 입증

### 모델 크기보다 정렬이 중요함을 실험적으로 증명

- 13억 파라미터 모델이 1750억 GPT-3보다 선호됨
- 스케일링보다 학습 신호 설계가 더 중요함을 시사

# 4. Method Analysis(설계 관점)

- 전체 구조는 세 단계
- Supervised Fine-Tuning
    
    인간이 작성한 이상적 응답으로 지도 학습
    
- Reward Model Training
    
    인간이 선호한 출력 순서를 학습
    
- Reinforcement Learning
    
    보상 모델을 이용한 정책 최적화
    

# 5. Mathematical Formulation Log

- 최적화 목표
    
    $\max_{\pi_\theta} \; \mathbb{E}_{x \sim D, y \sim \pi_\theta} \bigl[ r_\phi(x, y) \bigr]$
    
- 보상 모델 학습 손실
    
    $\mathcal{L}(\phi)
    = - \mathbb{E}_{(x, y_w, y_l)}
    \bigl[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\bigr]$
    
- PPO 목적 함수
    
    $\mathcal{L}_{PPO}
    = \mathbb{E}\bigl[
    \min(
    \rho_t(\theta) A_t,
    \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t
    )
    \bigr]$
    
- KL 정규화 항
    
    $\mathcal{L}_{KL}
    = \beta \, D_{KL}(\pi_\theta \,\|\, \pi_{SFT})$
    

# 6. Experiment as Claim Verification

- 인간 평가에서 InstructGPT가 일관되게 선호됨
- 사실성 지표 TruthfulQA에서 개선 확인
- 독성 생성 빈도 감소
- 공개 NLP 벤치마크 성능 저하는 PPO-ptx로 완화 가능

# 7. Limitations & Failure Modes

- 인간 라벨러의 가치 편향 반영 가능성
- 문화적 다양성 부족
- 악의적 지시에도 순응하는 경향
- 완전한 안전성 확보에는 미달

# 8. Extension & Research Ideas

- 다중 집단 선호를 반영한 보상 모델
- 헌법 기반 학습으로의 확장
- 자동 보상 모델의 자기 개선
- 비언어적 태스크로의 일반화

# 9. Code Strategy

- 전체 구조를 하나의 스크립트로 재현
- SFT, Reward Model, PPO를 최소 구현
- 실제 대규모 학습이 아닌 절차 검증 목적

# 10. One-Paragraph Research Summary

본 논문은 언어 모델의 실패 원인을 성능이 아닌 목적 함수의 불일치로 규정하고, 인간 선호를 직접 학습 신호로 사용하는 강화 학습 기반 정렬 방법을 제안한다. 지도 학습, 보상 모델, PPO를 결합한 RLHF 파이프라인을 통해 모델 크기 증가보다 정렬이 사용자 만족도를 더 크게 좌우함을 실험적으로 입증하며, 현대 LLM 정렬 연구의 출발점을 제공한다.

# 11. Connection to Other Papers

- Learning from Human Preferences
- Constitutional AI
- Few-Shot Learners
- Scaling Laws와의 대비
- Instruction Tuning 계열 연구

# 12. Personal Insight Log

- 언어 모델의 진짜 성능은 목적 함수에서 결정됨
- 인간 피드백은 데이터가 아니라 제약 조건
- RLHF는 모델 학습이 아니라 행동 교정에 가깝다