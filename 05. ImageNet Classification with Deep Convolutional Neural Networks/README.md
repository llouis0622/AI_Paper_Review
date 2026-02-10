# 1. Reading Context(사전 맥락)

- 대규모 이미지 인식 문제에서 기존 수작업 특징 기반 방법이 한계를 보이던 상황
- CNN은 개념적으로 존재했지만, 대규모 고해상도 데이터에서는 계산량 문제로 실용적이지 않았음
- 이 논문은 데이터 규모, 모델 깊이, 하드웨어 활용을 동시에 밀어붙인 첫 성공 사례

# 2. Problem Re-definition

- 문제 설정
    - 120만 장 이상의 고해상도 이미지
    - 1000개 클래스 분류 문제
- 기존 접근의 한계
    - 얕은 모델의 표현력 부족
    - 수작업 특징 설계의 확장성 문제
- 핵심 질문
    - 대규모 데이터에서 깊은 CNN을 실제로 학습시킬 수 있는가
    - 과적합 없이 일반화 성능을 얻을 수 있는가

# 3. Core Contributions(논문의 핵심 기여)

### 대규모 CNN의 실질적 성공

- 8개 층을 가진 깊은 CNN을 ImageNet에 최초로 성공적으로 적용
- 파라미터 수 약 6000만 규모

### ReLU 비선형 함수의 본격적 채택

- 포화 비선형 함수 대신 ReLU 사용
- 학습 속도 대폭 향상

### GPU 기반 병렬 학습 구조

- 모델을 두 개 GPU로 분산
- 계산량과 메모리 한계를 동시에 해결

### Dropout 정규화 도입

- 완전연결층에서 과적합 억제
- 이후 딥러닝 표준 기법으로 자리잡음

# 4. Method Analysis(설계 관점)

- 입력 데이터
    - 크기 정규화된 RGB 이미지
- 모델 구성
    - 합성곱 계층 5개
    - 완전연결 계층 3개
- 계층별 핵심 설계
    - 초기 계층에서 큰 receptive field
    - 후반부에서 점진적 추상화
- 예측 방식
    - Softmax 기반 다중 클래스 분류

# 5. Mathematical Formulation Log

- 분류 확률 모델
    
    $p(y \mid x) = \text{Softmax}(f_\theta(x))$
    
- 학습 목표
    
    $\min_\theta \; \mathbb{E}_{(x,y)} \bigl[-\log p(y \mid x)\bigr]$
    
- ReLU 비선형성
    
    $f(x) = \max(0, x)$
    
- 드롭아웃 적용
    
    $h^{(l)} = m^{(l)} \odot a^{(l)}, \quad m^{(l)} \sim \text{Bernoulli}(p)$
    
- SGD 기반 파라미터 업데이트
    
    $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$
    

# 6. Experiment as Claim Verification

- ImageNet LSVRC 2010
    - Top-5 error 약 17퍼센트
- ILSVRC 2012
    - Top-5 error 약 15퍼센트
- 기존 SIFT 기반 방법 대비 압도적 성능 향상
- 단일 구조가 아니라 앙상블 시 추가 개선 확인

# 7. Limitations & Failure Modes

- 파라미터 수가 매우 커 메모리 요구량 큼
- 학습 비용이 매우 높음
- 구조가 수작업 설계에 크게 의존
- 공간적 불변성 외의 변형에는 취약

# 8. Extension & Research Ideas

- AlexNet 이후
    - VGG에서 깊이 증가
    - GoogLeNet에서 구조적 효율성 추구
    - ResNet에서 학습 안정성 문제 해결
- Dropout 이후
    - Batch Normalization
    - Layer Normalization

# 9. Code Strategy

- AlexNet 구조를 그대로 재현
- ReLU와 Dropout 명시적 포함
- ImageNet 대신 소규모 데이터셋에서도 동작 가능하도록 구성
- 단일 코드 파일로 모델 정의와 학습 루프 포함

# 10. One-Paragraph Research Summary

이 논문은 대규모 이미지 분류 문제에서 깊은 합성곱 신경망이 실질적으로 작동할 수 있음을 처음으로 입증한 연구이다. ReLU, Dropout, GPU 병렬 학습이라는 설계 선택을 통해 계산 문제와 과적합 문제를 동시에 해결하였으며, 이후 모든 컴퓨터 비전 딥러닝 모델의 기준점을 형성했다.

# 11. Connection to Other Papers

- LeNet
- VGG
- GoogLeNet
- ResNet
- 현대 비전 파운데이션 모델의 출발점

# 12. Personal Insight Log

- 이 논문의 혁신은 새로운 이론이 아니라 공학적 결단의 집합
- 데이터, 모델, 하드웨어를 동시에 확장해야 성능이 열린다는 교훈
- 딥러닝 연구에서 구현 가능성이 이론만큼 중요하다는 전환점