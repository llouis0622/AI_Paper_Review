# 1. Reading Context(사전 맥락)

- 이미지 인식은 합성곱 기반 귀납 편향에 의존해 왔음
- Transformer는 시퀀스 처리에 강점을 보였으나 비전 적용은 제한적
- 대규모 사전학습이 가능할 때, 합성곱 없이도 경쟁력이 있는지 검증 필요

# 2. Problem Re-definition

- 이미지를 픽셀 격자가 아닌 시퀀스로 다룰 수 있는가
- 지역성, 이동 불변성 같은 귀납 편향 없이도 일반화가 가능한가
- 실무적으로 필요한 핵심
    - 단순한 구조
    - 대규모 사전학습 친화성
    - 전이 학습 효율

# 3. Core Contributions(논문의 핵심 기여)

### 이미지 패치의 시퀀스화

- 이미지를 고정 크기 패치로 분할
- 각 패치를 토큰으로 간주하여 Transformer 입력으로 사용

### 순수 Transformer Encoder 기반 비전 모델 제시

- 합성곱 레이어 미사용
- Encoder 블록만으로 이미지 분류 수행

### 대규모 사전학습의 중요성 실증

- 충분한 데이터가 주어질 때 CNN 대비 우수 성능
- 데이터 규모가 성능을 좌우함을 명확히 제시

# 4. Method Analysis(설계 관점)

- 입력 처리
    - 이미지를 패치 단위로 분할 후 선형 임베딩
- 시퀀스 구성
    - 클래스 토큰과 위치 임베딩 추가
- 네트워크
    - 다층 Transformer Encoder 스택
- 출력
    - 클래스 토큰을 이용한 분류

# 5. Mathematical Formulation Log

- 입력 이미지의 패치 분할 수
    
    $N = \frac{HW}{P^2}$
    
- 패치 임베딩과 위치 임베딩을 포함한 초기 시퀀스
    
    $z_0 = [x_{\text{class}}; x_p E] + E_{\text{pos}}$
    
- Transformer Encoder 레이어
    
    $z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$
    
    $z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$
    
- 최종 분류 입력
    
    $y = \text{LN}(z_L^{(0)})$
    

# 6. Experiment as Claim Verification

- ImageNet-21k, JFT-300M 사전학습 후 ImageNet 미세조정
- 대규모 데이터에서 CNN 계열을 안정적으로 상회
- 소규모 데이터에서는 성능 열세 확인

# 7. Limitations & Failure Modes

- 소량 데이터 환경에서 성능 저하
- 계산량 증가에 따른 메모리 부담
- 위치 임베딩 보간에 대한 민감성

# 8. Extension & Research Ideas

- 데이터 효율 개선을 위한 하이브리드 구조
- 사전학습 비용 절감을 위한 distillation
- 멀티모달 Transformer로의 확장

# 9. Code Strategy

- 패치 임베딩을 합성곱으로 구현
- Encoder 블록을 표준 Transformer로 구성
- 클래스 토큰 기반 분류
- 단일 파일 구성

# 10. One-Paragraph Research Summary

이 논문은 이미지를 패치 시퀀스로 변환하여 Transformer Encoder만으로 이미지 분류를 수행하는 Vision Transformer를 제안한다. 합성곱의 귀납 편향을 제거한 대신 대규모 사전학습을 통해 일반화를 달성하며, 비전 모델 설계의 패러다임을 구조 중심에서 데이터 중심으로 전환하는 계기를 제공했다.

# 11. Connection to Other Papers

- Transformer
- Data-efficient Image Transformers
- Swin Transformer
- Multimodal Pretraining Models

# 12. Personal Insight Log

- ViT의 성능은 구조보다 데이터 규모에 좌우됨
- 귀납 편향 제거는 단순화이자 위험
- 이후 연구는 데이터 효율 보완에 집중