# 1. Reading Context(사전 맥락)

- NLP에서 라벨 데이터는 희소하지만 비지도 텍스트는 풍부함
- Word embedding 수준을 넘어 문장 및 문서 단위 의미 표현이 필요
- task별 구조 설계 없이 범용 표현 학습이 가능한지에 대한 문제의식

# 2. Problem Re-definition

- 개별 태스크마다 다른 모델을 설계하는 방식은 확장성이 없음
- 비지도 학습으로 얻은 표현을 어떻게 다양한 NLU 태스크로 전이할 것인가가 핵심
- 실무적으로 필요한 세 요소
    - 대규모 비지도 데이터 활용
    - 구조 변경 없는 태스크 적응
    - 장거리 의존성 처리 능력

# 3. Core Contributions(논문의 핵심 기여)

### Generative Pre-training + Discriminative Fine-tuning 프레임 제시

- 언어 모델을 먼저 학습한 뒤 태스크에 맞게 미세조정
- 표현 학습과 태스크 학습을 분리

### Transformer 기반 언어 모델 사용

- RNN 계열 대비 장거리 의존성 처리 우수
- decoder-only Transformer 구조 채택

### Task-aware Input Transformation

- 입력 구조만 변환하고 모델 구조는 유지
- entailment, QA, similarity를 모두 시퀀스로 변환

# 4. Method Analysis(설계 관점)

- Pre-training
    - 비지도 텍스트에서 다음 토큰 예측
- Fine-tuning
    - 동일 모델에 얕은 출력층만 추가
- 입력 표현
    - 여러 문장 구조를 하나의 토큰 시퀀스로 직렬화
- 설계 철학
    - 모델은 범용
    - 태스크 적응은 입력 변환으로 해결

# 5. Mathematical Formulation Log

- 언어 모델 목표 함수
    
    $\mathcal{L}1(U) = \sum_i \log P(u_i \mid u{i-k}, \ldots, u_{i-1})$
    
- Transformer hidden state
    
    $h_0 = U W_e + W_p$
    
    $h_l = \text{TransformerBlock}(h_{l-1})$
    
- 토큰 분포
    
    $P(u) = \text{softmax}(h_n W_e^\top)$
    
- 지도 학습 출력
    
    $P(y \mid x_1, \ldots, x_m) = \text{softmax}(h_m^l W_y)$
    
- Fine-tuning 목적 함수
    
    $\mathcal{L}2(C) = \sum{(x,y)} \log P(y \mid x)$
    
- 보조 목표 포함
    
    $\mathcal{L}_3 = \mathcal{L}_2 + \lambda \mathcal{L}_1$
    

# 6. Experiment as Claim Verification

- GLUE 벤치마크 다수 태스크에서 SOTA 달성
- SNLI, MultiNLI, RACE, Story Cloze 등에서 큰 성능 향상
- pre-training 제거 시 성능 급감 확인

# 7. Limitations & Failure Modes

- 양방향 문맥 미사용
- 생성 기반이라 추론 비용 큼
- 태스크 특화 구조 대비 한계 존재

# 8. Extension & Research Ideas

- GPT-2, GPT-3로의 스케일 확장
- Bidirectional 확장으로 BERT 등장
- In-context learning으로 발전

# 9. Code Strategy

- Decoder-only Transformer 구현
- Causal masking 적용
- Pre-training용 LM loss
- 단일 파일 구조

# 10. One-Paragraph Research Summary

이 논문은 대규모 비지도 텍스트에서 언어 모델을 사전 학습한 뒤, 최소한의 구조 변경만으로 다양한 자연어 이해 태스크에 전이하는 방법을 제시한다. Transformer 기반 생성 모델을 통해 장거리 의존성을 포착하고, 입력 변환을 통해 태스크 특화 문제를 통일된 시퀀스 예측 문제로 환원함으로써 범용 언어 이해 모델의 가능성을 처음으로 실증했다.

# 11. Connection to Other Papers

- ELMo
- BERT
- GPT-2, GPT-3
- Semi-supervised Learning 계열

# 12. Personal Insight Log

- GPT의 본질은 모델이 아니라 학습 절차의 분리
- 언어 모델은 표현 학습기로 기능 가능
- 이후 모든 대형 언어 모델의 설계 기준점