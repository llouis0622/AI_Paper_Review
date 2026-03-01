# 1. Reading Context(사전 맥락)

- GAN 계열은 샘플 품질은 높지만 학습 불안정 문제가 지속
- 우도 기반 모델은 안정적이나 샘플 품질이 제한됨
- 점진적 노이즈 제거를 통해 안정적 학습과 고품질 샘플을 동시에 달성할 수 있는지 검증 필요

# 2. Problem Re-definition

- 데이터 분포 $p_{\text{data}}(x_0)$를 모사하는 생성 모델 학습
- 직접적인 우도 최적화는 어렵거나 샘플링이 비효율적
- 실무적으로 필요한 핵심
    - 안정적인 학습
    - 고품질 샘플링
    - 명시적 확률 모델 기반

# 3. Core Contributions(논문의 핵심 기여)

### 점진적 노이즈 추가·제거 프레임 정식화

- 데이터에 가우시안 노이즈를 단계적으로 추가하는 전방 과정
- 이를 역으로 되돌리는 역과정 학습

### Score Matching과의 연결

- 역과정을 노이즈 예측 문제로 재정식화
- 가중 변분 하한이 다중 스케일 denoising score matching과 동치임을 제시

### 고품질 샘플링 실증

- CIFAR-10, LSUN에서 SOTA 수준의 FID 달성
- 점진적 디코딩을 통한 안정적 생성

# 4. Method Analysis(설계 관점)

- Forward Process
    
    데이터에 점진적으로 가우시안 노이즈 추가
    
- Reverse Process
    
    신경망이 노이즈를 예측하여 단계적으로 제거
    
- 네트워크
    
    시간 인덱스를 입력으로 받는 U-Net 계열
    
- 설계 목표
    
    각 단계의 문제를 **단순한 회귀 문제**로 환원
    

# 5. Mathematical Formulation Log

- 전방 확산 과정
    
    $q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right)$
    
- 누적 표현
    
    $q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I\right),
    \quad
    \bar{\alpha}_t=\prod_{s=1}^{t}(1-\beta_s)$
    
- 역과정 모델
    
    $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\!\left(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I\right)$
    
- 노이즈 예측 파라미터화
    
    $\mu_\theta(x_t,t)
    =
    \frac{1}{\sqrt{\alpha_t}}
    \left(
    x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,
    \varepsilon_\theta(x_t,t)
    \right)$
    
- 단순화된 학습 목적
    
    $\mathcal{L}_{\text{simple}}
    =
    \mathbb{E}_{t,x_0,\varepsilon}
    \left[
    \left\|
    \varepsilon-\varepsilon_\theta\!\left(
    \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\varepsilon,t
    \right)
    \right\|^2
    \right]$
    

# 6. Experiment as Claim Verification

- CIFAR-10에서 FID 3.17 달성
- 단계 수 증가에 따라 전역 구조가 먼저 형성되고 세부 디테일이 후반에 추가됨
- 역과정 분산 고정이 학습 안정성에 기여함을 확인

# 7. Limitations & Failure Modes

- 샘플링 단계 수가 많아 생성 속도 느림
- 장기 의존성 표현은 단계 수에 민감
- 초기 노이즈 스케줄 선택에 성능 의존

# 8. Extension & Research Ideas

- DDIM을 통한 결정론적 가속 샘플링
- Score-based SDE 일반화
- 조건부 생성과 텍스트-이미지 결합
- Latent diffusion으로 계산 비용 절감

# 9. Code Strategy

- 고정 노이즈 스케줄 사용
- $\varepsilon$-prediction 기반 학습
- 단일 파일로 학습·샘플링 구현

# 10. One-Paragraph Research Summary

이 논문은 데이터에 점진적으로 가우시안 노이즈를 추가한 뒤 이를 단계적으로 제거하는 확률적 생성 프레임을 제안한다. 역과정을 노이즈 예측 문제로 재정식화함으로써 학습을 단순화하고, score matching과의 이론적 연결을 통해 안정적 학습과 고품질 샘플링을 동시에 달성했다. 이는 이후 확산 모델 계열의 표준 설계로 자리 잡았다.

# 11. Connection to Other Papers

- Denoising Score Matching
- Langevin Dynamics
- DDIM
- Latent Diffusion Models

# 12. Personal Insight Log

- 확산 모델의 힘은 복잡한 생성 문제를 반복적 회귀로 분해한 점
- 손실 설계가 모델 구조보다 더 중요
- 이후 연구의 핵심은 샘플링 속도 개선