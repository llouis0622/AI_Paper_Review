# 1. Reading Context(사전 맥락)

- 다층 신경망은 표현력은 높지만 학습이 거의 불가능하다는 인식이 지배적이던 시기
- 역전파는 깊은 구조에서 지역 최소점과 기울기 소실 문제로 실질적 학습이 어려움
- 깊은 생성 모델을 안정적으로 학습할 수 있는 새로운 학습 전략이 필요

# 2. Problem Re-definition

- 깊은 확률 모델에서의 핵심 문제
    - 후방 분포 추론이 매우 어려움
    - 전체 파라미터를 동시에 학습하기 어려움
- 해결해야 할 조건
    - 국소적 학습 규칙
    - 층별로 안정적인 표현 학습
    - 깊이가 늘어나도 학습이 망가지지 않는 구조
- 문제를 다음과 같이 재정의
    - 전체 모델 학습이 아니라 **층별 표현 학습을 누적**하는 문제

# 3. Core Contributions(논문의 핵심 기여)

### Greedy Layer-wise Pretraining 제안

- 각 층을 독립적인 Restricted Boltzmann Machine으로 학습
- 하위 층의 출력을 상위 층의 데이터로 사용
- 전체 모델을 한 번에 학습하지 않음

### Deep Belief Net 구조 정식화

- 최상위 두 층은 무방향 그래프
- 하위 층들은 방향성 생성 모델
- 생성과 추론이 결합된 하이브리드 구조

### 빠르고 안정적인 학습 가능성 입증

- MNIST 등에서 기존 방법 대비 큰 성능 향상
- 깊은 모델이 실제로 학습 가능하다는 실증적 증거 제시

# 4. Method Analysis(설계 관점)

- 학습 절차는 두 단계
    - 비지도 사전학습
    - 선택적 지도 미세조정
- 각 층 학습 시
    - 입력은 바로 아래 층의 은닉 표현
    - 출력은 그 층의 은닉 변수
- RBM 학습은 국소적 가중치 업데이트만 필요
- 깊이가 증가해도 학습 난이도 급증 없음

# 5. Mathematical Formulation Log

- DBN의 결합 분포 정의
    
    $p(v,h^1,\ldots,h^L) = p(v \mid h^1)\prod_{l=1}^{L-1} p(h^l \mid h^{l+1})p(h^{L-1},h^L)$
    
- RBM 에너지 함수
    
    $E(v,h) = -b^\top v - c^\top h - v^\top W h$
    
- 확률 분포
    
    $p(v,h) = \frac{1}{Z}\exp(-E(v,h))$
    
- 조건부 독립성
    
    $p(h_j=1 \mid v) = \sigma(c_j + W_j^\top v)$
    
    $p(v_i=1 \mid h) = \sigma(b_i + W_i h)$
    
- Contrastive Divergence 업데이트
    
    $\Delta W \propto \langle v h^\top \rangle_{\text{data}} - \langle v h^\top \rangle_{\text{model}}$
    

# 6. Experiment as Claim Verification

- MNIST에서 깊은 DBN이 얕은 모델보다 낮은 오류율 달성
- 사전학습이 없는 경우 성능 급락
- 층별 사전학습이 표현의 질을 점진적으로 개선함을 확인

# 7. Limitations & Failure Modes

- RBM 학습 자체가 느림
- Contrastive Divergence는 근사 학습
- 연속 변수 처리에 제약
- 이후 더 단순한 구조로 대체됨

# 8. Extension & Research Ideas

- RBM 대신 Autoencoder 사용
- DBN에서 DBM으로 확장
- 확률적 사전학습에서 결정론적 사전학습으로 전환
- 이후 딥러닝 학습 패러다임의 전환점 제공

# 9. Code Strategy

- Bernoulli-Bernoulli RBM 구현
- CD-1 학습
- 두 층 RBM을 순차적으로 학습
- 단일 파일 구성

# 10. One-Paragraph Research Summary

이 논문은 깊은 신경망이 학습되지 않는다는 기존 인식을 뒤집고, 층별로 Restricted Boltzmann Machine을 학습하는 방식으로 깊은 확률 모델을 안정적으로 훈련할 수 있음을 보였다. 전체 모델을 한 번에 최적화하지 않고, 표현을 점진적으로 쌓아 올리는 전략은 이후 딥러닝 사전학습 패러다임의 핵심이 되었으며, 깊은 모델 학습의 실질적 출발점 역할을 했다.

# 11. Connection to Other Papers

- Restricted Boltzmann Machines
- Autoencoders
- Deep Boltzmann Machines
- Unsupervised Pretraining
- 현대 딥러닝 초기 연구 흐름

# 12. Personal Insight Log

- 이 논문의 핵심은 모델이 아니라 학습 순서
- 깊이를 한 번에 정복하지 않고, 층 단위로 길을 닦는 전략
- 이후 역전파가 가능해진 것도 이 사전학습 덕분