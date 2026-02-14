# 1. Reading Context(사전 맥락)

- 네트워크 깊이가 증가할수록 성능이 좋아질 것이라는 직관이 실제 학습에서는 깨지는 이유를 이해하기 위해
- vanishing gradient 문제가 해결된 이후에도 남아 있는 degradation problem의 정체를 규명하기 위해
- “더 깊게 쌓는 것”이 아니라 “어떻게 쌓아야 하는가”라는 설계 관점을 정리하기 위해

# 2. Problem Re-definition

- Batch Normalization 등으로도 매우 깊은 네트워크는 학습이 어려움
- 깊이가 증가하면 overfitting이 아니라 training error 자체가 증가
- 이 현상은 표현력 문제가 아니라 최적화 구조 문제
- 실질적으로 필요한 것은 다음 세 가지
    - 깊이가 증가해도 학습이 망가지지 않는 구조
    - 기존 얕은 모델보다 최소한 성능이 나빠지지 않는 보장
    - 추가 파라미터 없이 최적화 난이도를 낮추는 방법

# 3. Core Contributions(논문의 핵심 기여)

### Residual Learning Framework 제안

- 직접 함수 $H(x)$를 학습하지 않고 잔차 함수 $F(x)$를 학습
- 네트워크는 다음 형태를 가짐
    
    $H(x)=F(x)+x$
    

### Identity Shortcut Connection 도입

- 입력을 그대로 더하는 경로를 추가
- 파라미터와 계산량 증가 없음
- gradient가 직접 전달되는 경로 확보

### Degradation Problem의 구조적 해결

- 깊이가 증가해도 training error가 감소
- 100층 이상에서도 안정적인 최적화 가능

### 매우 깊은 네트워크의 실증

- ImageNet에서 최대 152층 네트워크 학습
- VGG보다 깊지만 계산량은 더 적음
- ILSVRC 2015 우승

# 4. Method Analysis(설계 관점)

- 입력
    
    이미지 $x$
    
- 기본 블록
    
    잔차 블록
    
- Residual Block 구조
    
    입력 $x$
    
    비선형 변환 $F(x)$
    
    출력 $y=F(x)+x$
    
- 핵심 설계 원칙
    - 각 블록은 전체 변환이 아니라 “수정량”만 담당
    - 깊을수록 각 층의 역할은 점점 작아짐
    - 정보 흐름은 항상 보존됨

# 5. Mathematical Formulation Log

- 기존 네트워크가 학습하는 목표
    
    $H(x)$
    
- ResNet에서의 재정의
    
    $F(x)=H(x)−x$
    
- Residual Block 출력
    
    $y=F(x,{W_i})+x$
    
- 차원이 다른 경우
    
    $y = F(x, \{W_i\}) + W_s x$
    
- 핵심 효과
    - $F(x)=0$이면 항등 함수가 됨
    - 최적해 공간에 “쉬운 해”가 항상 존재

# 6. Experiment as Claim Verification

- CIFAR-10
    - 20층부터 110층, 1200층까지 실험
    - Plain net은 깊어질수록 성능 붕괴
    - ResNet은 깊어질수록 성능 향상
- ImageNet
    - ResNet-34, 50, 101, 152 비교
    - 깊이 증가에 따라 Top-1, Top-5 error 지속 감소
    - 152-layer 단일 모델이 기존 ensemble 성능 초과

# 7. Limitations & Failure Modes

- 매우 깊은 모델은 메모리 사용량 증가
- 구조가 복잡해 해석성 감소
- 과도한 깊이는 작은 데이터셋에서 과적합 가능
- Residual 구조만으로 모든 문제 해결 불가

# 8. Extension & Research Ideas

- Pre-activation ResNet
- Wide ResNet
- DenseNet과의 구조적 비교
- Transformer residual connection으로의 일반화
- Optimization 관점에서의 잔차 공간 분석

# 9. Code Strategy

- Residual Block을 명시적으로 구현
- Identity shortcut과 projection shortcut 분리
- CIFAR-10 기준 간단한 ResNet 구성
- 하나의 파일에서 전체 구조 정의 및 학습 가능하도록 구성

# 10. One-Paragraph Research Summary

이 논문은 매우 깊은 신경망이 학습 과정에서 오히려 성능이 저하되는 degradation problem이 최적화 구조의 문제임을 지적하고, 이를 해결하기 위해 잔차 학습이라는 새로운 네트워크 설계를 제안한다. 각 층이 전체 함수를 학습하는 대신 입력에 대한 수정량만을 학습하도록 함으로써, 정보와 그래디언트의 흐름을 보존하고 매우 깊은 네트워크에서도 안정적인 학습을 가능하게 했다. 이 구조는 이후 거의 모든 딥러닝 모델의 기본 설계 원리가 되었다.

# 11. Connection to Other Papers

- VGG
    
    깊이 증가의 한계를 드러낸 전신 모델
    
- Batch Normalization
    
    ResNet 학습 안정성의 기반 기술
    
- Highway Network
    
    게이트 기반 skip connection과의 대비
    
- DenseNet
    
    residual 개념의 극단적 확장
    

# 12. Personal Insight Log

- ResNet의 핵심은 “깊이”가 아니라 “경로 설계”
- 학습이 안 되는 문제는 대부분 표현력보다 최적화 문제
- Residual connection은 구조적 prior를 설계에 넣은 사례
- 현대 모델 대부분은 ResNet의 직계 후손