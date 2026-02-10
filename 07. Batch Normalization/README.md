# 1. Reading Context(사전 맥락)

- 딥네트워크가 깊어질수록 학습이 극도로 불안정해지는 문제가 반복적으로 관찰됨
- 학습률을 크게 잡으면 발산하고, 작게 잡으면 수렴이 지나치게 느림
- ReLU, 초기화 기법 등으로 완화는 되었지만 근본 원인은 명확히 설명되지 않았음
- 이 논문은 “왜 깊은 네트워크 학습이 어려운가”를 분포 관점에서 재정의함

# 2. Problem Re-definition

- 기존 인식
    - 문제는 기울기 소실이나 폭주
- 논문의 재정의
    - 각 층 입력 분포가 학습 중 계속 변함
    - 이로 인해 상위 층이 계속 적응해야 함
- 이를 논문에서는 Internal Covariate Shift라고 정의
- 실무적으로 필요한 조건
    - 입력 분포 변화에 둔감한 학습
    - 초기화와 학습률에 덜 민감한 구조
    - 깊은 네트워크에서도 안정적 수렴

# 3. Core Contributions(논문의 핵심 기여)

### Internal Covariate Shift 개념 정식화

- 네트워크 내부에서도 covariate shift가 발생함을 명확히 정의
- 학습 난이도의 주요 원인을 분포 이동으로 설명

### Mini-batch 기반 정규화 연산 제안

- 각 층 입력을 미니배치 통계로 정규화
- 정규화를 모델 내부 연산으로 포함

### 학습 가능한 scale, shift 도입

- 정규화로 인한 표현력 손실을 방지
- 정규화가 항등 변환도 표현 가능하도록 설계

### 최적화 가속과 정규화 효과 동시 확보

- 더 큰 학습률 사용 가능
- Dropout 없이도 일반화 성능 유지

# 4. Method Analysis(설계 관점)

- 적용 위치
    - 선형 변환 이후, 비선형 함수 이전
- 핵심 구성 요소
    - 평균 제거
    - 분산 정규화
    - 학습 가능한 선형 변환
- Convolution layer에서는
    - 같은 feature map 전체를 하나의 통계로 정규화
- 학습 단계와 추론 단계의 동작 분리

# 5. Mathematical Formulation Log

- 미니배치 평균
    
    $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$
    
- 미니배치 분산
    
    $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$
    
- 정규화된 활성값
    
    $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$
    
- 학습 가능한 변환
    
    $y_i = \gamma \hat{x}_i + \beta$
    
- 추론 단계 정규화
    
    $\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}$
    

# 6. Experiment as Claim Verification

- MNIST
    - 수렴 속도 대폭 향상
    - 활성 분포 안정화 시각적 확인
- ImageNet
    - 동일 정확도를 약 14배 적은 스텝으로 달성
    - 단일 모델 기준 기존 최고 성능 초과
- Sigmoid 사용 네트워크도 학습 가능해짐

# 7. Limitations & Failure Modes

- 미니배치 크기에 민감
- 작은 배치에서는 통계 노이즈 증가
- RNN, Online learning에는 직접 적용 어려움
- 학습과 추론 동작이 다르다는 구조적 복잡성

# 8. Extension & Research Ideas

- Layer Normalization
- Instance Normalization
- Group Normalization
- Batch Renormalization
- Normalization-free network 설계

# 9. Code Strategy

- 선형 계층 뒤에 Batch Normalization 삽입
- 학습 시 batch 통계 사용
- 추론 시 running mean, variance 사용
- 단일 파일로 모델 정의와 학습 루프 포함

# 10. One-Paragraph Research Summary

이 논문은 딥네트워크 학습을 어렵게 만드는 핵심 원인을 내부 활성 분포의 변화로 정의하고, 이를 미니배치 기반 정규화를 통해 구조적으로 해결한다. Batch Normalization은 학습 안정성과 수렴 속도를 동시에 개선하며, 큰 학습률과 단순 초기화를 가능하게 만든다. 이후 대부분의 딥러닝 아키텍처에서 기본 구성 요소로 채택되었다.

# 11. Connection to Other Papers

- Dropout과의 비교 및 대체 관계
- Layer Normalization 계열 연구
- Optimization 안정성 연구
- Modern CNN, Transformer 구조 전반

# 12. Personal Insight Log

- Batch Normalization은 정규화 기법이 아니라 학습 조건 제어 장치
- 최적화 문제를 모델 구조 문제로 전환한 사례
- 이후 딥러닝 설계 패러다임을 바꾼 논문