# 1. Reading Context(사전 맥락)

- RNN 기반 Seq2Seq와 Attention 모델이 성능은 높지만 병렬화가 어려움
- 긴 시퀀스에서 학습 시간과 메모리 비용이 급격히 증가
- Attention이 핵심이라면, 다른 구조 요소는 필수인가에 대한 문제 제기
- 순환과 합성을 제거한 순수 Attention 기반 구조 가능성 탐색

# 2. Problem Re-definition

- 기존 시퀀스 모델의 핵심 병목
    - 순차적 계산으로 인한 병렬화 불가
    - 긴 거리 의존성 학습의 비효율
- 문제를 다음과 같이 재정의
    - 시퀀스 간 의존성을 순차 처리 없이 모델링할 수 있는가
    - 모든 위치 간 상호작용을 일정 깊이로 처리할 수 있는가
- 실무적으로 필요한 조건
    - 완전 병렬 학습
    - 긴 거리 의존성에 대한 짧은 경로
    - 기존 성능을 넘는 번역 품질

# 3. Core Contributions(논문의 핵심 기여)

### 순수 Self-Attention 기반 Transformer 제안

- RNN과 CNN을 완전히 제거
- Encoder와 Decoder 모두 Self-Attention과 Feed-Forward 블록으로 구성

### Scaled Dot-Product Attention 정식화

- 내적 기반 Attention의 수치적 불안정성 문제 해결
- 차원에 따른 스케일링 도입

### Multi-Head Attention 도입

- 서로 다른 표현 서브공간에서 병렬 Attention 수행
- 단일 Attention의 평균화 한계 극복

### Positional Encoding 설계

- 순서 정보가 없는 Attention 구조에 위치 정보 주입
- 사인, 코사인 기반 위치 표현 제안

# 4. Method Analysis(설계 관점)

- Encoder
    - Self-Attention
    - Position-wise Feed-Forward Network
- Decoder
    - Masked Self-Attention
    - Encoder-Decoder Attention
    - Feed-Forward Network
- 공통 설계
    - Residual Connection
    - Layer Normalization
- 모든 서브레이어는 동일 차원 표현 유지

# 5. Mathematical Formulation Log

- Scaled Dot-Product Attention
    
    $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
    
- Multi-Head Attention
    
    $\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
    
    $\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W^O$
    
- Position-wise Feed-Forward Network
    
    $\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
    
- Positional Encoding
    
    $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
    
    $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
    

# 6. Experiment as Claim Verification

- WMT 2014 English–German
    - 기존 최고 성능 대비 BLEU 점수 대폭 향상
- WMT 2014 English–French
    - 단일 모델 기준 새로운 최고 성능 달성
- 학습 시간
    - RNN 기반 모델 대비 현저히 단축
- Attention head 시각화를 통해 문법적 역할 분화 확인

# 7. Limitations & Failure Modes

- Self-Attention의 계산 복잡도는 시퀀스 길이 제곱에 비례
- 매우 긴 시퀀스에서는 메모리 부담 발생
- 위치 정보는 외부 주입에 의존
- 구조 단순성으로 인한 inductive bias 부족 가능성

# 8. Extension & Research Ideas

- Sparse Attention
- Long-Sequence Transformer
- Relative Positional Encoding
- Vision Transformer
- Multimodal Transformer

# 9. Code Strategy

- Multi-Head Attention 모듈 단독 구현
- Encoder Block과 Decoder Block 분리
- Positional Encoding 모듈 포함
- 단일 파일 구성

# 10. One-Paragraph Research Summary

이 논문은 시퀀스 모델링에서 순환과 합성을 제거하고, Self-Attention만으로 입력과 출력 간 의존성을 모델링하는 Transformer 구조를 제안한다. 모든 위치 간 상호작용을 일정 깊이로 처리함으로써 병렬화와 긴 거리 의존성 학습을 동시에 달성하였고, 기계번역에서 기존 모델을 성능과 효율성 모두에서 능가했다. 이후 자연어 처리 전반의 표준 아키텍처가 되었다.

# 11. Connection to Other Papers

- Neural Machine Translation by Jointly Learning to Align and Translate
- Sequence to Sequence Learning with Neural Networks
- BERT
- GPT 계열 모델
- Vision Transformer

# 12. Personal Insight Log

- Transformer의 혁신은 Attention 자체보다 구조적 단순화
- 경로 길이를 줄이는 것이 장기 의존성 학습의 핵심
- 이후 모델들은 대부분 이 구조를 확장하거나 제약 완화하는 방향으로 발전