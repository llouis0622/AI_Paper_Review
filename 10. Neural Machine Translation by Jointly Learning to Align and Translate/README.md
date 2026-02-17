# 1. Reading Context(사전 맥락)

- 기존 Seq2Seq 모델이 고정 길이 벡터 병목으로 인해 긴 문장에서 성능이 급격히 저하됨
- 번역 문제의 본질인 정렬 문제를 신경망 내부에서 직접 다루고자 함
- 통계적 기계번역의 핵심 요소였던 alignment를 신경망으로 흡수하려는 시도

# 2. Problem Re-definition

- 기존 Encoder–Decoder는 입력 문장을 하나의 고정 벡터로 압축
- 문장이 길어질수록 정보 손실이 누적
- 번역 과정에서 실제로 필요한 것은
    - 매 시점마다 다른 입력 위치에 집중하는 능력
    - 정렬을 명시적으로 주지 않아도 학습 가능한 구조
- 문제를 다음과 같이 재정의
    - 출력 단어마다 조건부로 입력의 관련 부분을 선택할 수 있는가

# 3. Core Contributions(논문의 핵심 기여)

### Alignment와 Translation의 공동 학습

- 번역과 정렬을 분리하지 않고 하나의 목적 함수로 학습
- alignment를 잠재변수가 아닌 미분 가능한 연산으로 구성

### Attention 메커니즘 도입

- 각 출력 시점마다 입력 전체를 가중합으로 참조
- 고정 길이 컨텍스트 벡터 병목 제거

### Bidirectional Encoder 활용

- 입력 단어 주변의 문맥을 양방향으로 반영
- 각 입력 위치에 대한 풍부한 annotation 생성

# 4. Method Analysis(설계 관점)

- Encoder
    - Bidirectional RNN
    - 입력 시퀀스를 annotation sequence로 변환
- Decoder
    - RNN 기반 생성 모델
    - 이전 상태와 context vector를 이용해 다음 단어 생성
- 핵심 설계
    - context vector를 시간마다 새로 계산
    - Decoder가 Encoder를 조회하는 구조

# 5. Mathematical Formulation Log

- 번역 목표 확률
    
    $p(y \mid x) = \prod_{i=1}^{T_y} p(y_i \mid y_{<i}, x)$
    
- Context vector
    
    $c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$
    
- Attention weight
    
    $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$
    
- Alignment score
    
    $e_{ij} = a(s_{i-1}, h_j)$
    
- Decoder 상태 업데이트
    
    $s_i = f(s_{i-1}, y_{i-1}, c_i)$
    

# 6. Experiment as Claim Verification

- WMT 2014 English–French 번역 실험
- 문장 길이가 길어질수록 기존 Encoder–Decoder 대비 큰 성능 격차
- BLEU 점수에서 통계적 기계번역 시스템과 동급 성능 달성
- Attention weight 시각화를 통해 직관적인 정렬 확인

# 7. Limitations & Failure Modes

- 출력 시점마다 입력 전체를 조회하므로 계산 비용 증가
- 매우 긴 시퀀스에서는 attention 계산이 부담
- 단어 단위 처리로 인한 희귀 단어 문제 존재

# 8. Extension & Research Ideas

- Global attention에서 local attention으로 확장
- Self-attention 구조로의 일반화
- Transformer로의 구조적 발전
- Multimodal alignment 문제로 확장

# 9. Code Strategy

- Bidirectional Encoder RNN 구현
- Additive Attention 메커니즘 구현
- Decoder에서 context vector 결합
- 단일 파일로 Encoder, Attention, Decoder 구성

# 10. One-Paragraph Research Summary

이 논문은 기존 Seq2Seq 모델의 고정 길이 표현 병목을 근본적으로 해결하기 위해, 번역 과정에서 매 시점마다 입력 문장의 관련 부분을 선택하는 attention 기반 구조를 제안한다. Alignment를 미분 가능한 방식으로 모델 내부에 포함시켜 번역과 정렬을 동시에 학습하며, 긴 문장에서도 안정적인 성능을 보였다. 이 구조는 이후 모든 신경 기계번역 모델과 Transformer의 직접적인 출발점이 되었다.

# 11. Connection to Other Papers

- Sequence to Sequence Learning with Neural Networks
- RNN Encoder–Decoder
- Transformer
- Self-Attention Models
- Modern NMT Systems

# 12. Personal Insight Log

- Attention의 본질은 가중합이 아니라 조건부 조회
- 번역 문제의 핵심이 정렬임을 다시 명확히 드러낸 논문
- 이후 시퀀스 모델링의 패러다임 전환점