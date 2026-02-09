# 1. Reading Context(사전 맥락)

- 문서 집합을 기억 기반 특징이 아닌 확률적 생성 모델로 설명하려는 시도
- pLSI가 가진 과적합과 일반화 불가능 문제를 구조적으로 해결하려는 목적
- 교환가능성 가정을 명시적으로 받아들이고, 이에 맞는 베이지안 계층 모델 제시

# 2. Problem Re-definition

- 문서는 여러 주제가 섞인 결과이며, 각 단어는 잠재 주제에서 생성됨
- 기존 모델의 문제
    - 문서별 파라미터 수가 데이터 크기에 따라 증가
    - 새로운 문서에 대한 확률 정의가 불명확
- 필요한 조건
    - 문서 단위 확률 모델
    - 주제 혼합을 확률 변수로 모델링
    - 데이터 크기와 무관한 파라미터 수

# 3. Core Contributions(논문의 핵심 기여)

### 문서 수준의 확률적 생성 모델 제시

- 문서별 주제 혼합을 Dirichlet 분포로 모델링
- 각 단어는 주제 선택 후 단어 분포에서 생성

### pLSI의 구조적 한계 해결

- 문서 인덱스 의존 파라미터 제거
- 새로운 문서에 대한 확률 계산 가능

### 변분 추론 기반의 실용적 추론 절차 제시

- 정확 추론이 불가능함을 명시
- 계산 가능한 하한을 통한 근사 추론 제공

# 4. Method Analysis(설계 관점)

- 계층 구조
    - 코퍼스 수준 파라미터
    - 문서 수준 잠재 변수
    - 단어 수준 잠재 변수
- 설계 핵심
    - 단어 교환가능성
    - 문서 교환가능성
    - 주제는 반복 샘플링됨
- 혼합 모델이지만 문서가 단일 클러스터에 묶이지 않음

# 5. Mathematical Formulation Log

- 문서 생성 확률
    
    $p(\theta \mid \alpha) = \mathrm{Dir}(\alpha)$
    
- 단어 생성 과정의 결합 분포
    
    $p(\theta, z, w \mid \alpha, \beta) = p(\theta \mid \alpha)\prod_{n=1}^{N} p(z_n \mid \theta)p(w_n \mid z_n, \beta)$
    
- 문서의 주변 분포
    
    $p(w \mid \alpha, \beta) = \int p(\theta \mid \alpha)\prod_{n=1}^{N}\sum_{z_n} p(z_n \mid \theta)p(w_n \mid z_n, \beta), d\theta$
    
- 변분 분포 가정
    
    $q(\theta, z \mid \gamma, \phi) = q(\theta \mid \gamma)\prod_{n=1}^{N} q(z_n \mid \phi_n)$
    
- 변분 파라미터 업데이트
    
    $\phi_{ni} \propto \beta_{i w_n}\exp{\Psi(\gamma_i)-\Psi(\sum_j \gamma_j)}$
    
    $\gamma_i = \alpha_i + \sum_{n=1}^{N}\phi_{ni}$
    

# 6. Experiment as Claim Verification

- perplexity 기준으로 unigram, mixture, pLSI 대비 성능 비교
- 주제 수 증가 시 pLSI와 mixture는 과적합 발생
- LDA는 안정적으로 일반화 성능 유지
- 분류와 협업 필터링에서도 표현력 유지 확인

# 7. Limitations & Failure Modes

- bag-of-words 가정으로 순서 정보 소실
- 변분 추론으로 인한 근사 오차
- 주제 수 선택에 따른 민감성
- 희소 단어 처리의 한계

# 8. Extension & Research Ideas

- Gibbs sampling 기반 추론
- Dynamic topic model
- Hierarchical Dirichlet Process
- 문장 단위 또는 시간 의존 확장

# 9. Code Strategy

- 변분 추론 기반 LDA
- 문서 단위 추론과 코퍼스 단위 파라미터 분리
- 단일 파일로 핵심 알고리즘 구현
- 소형 말뭉치 기준 동작 확인

# 10. One-Paragraph Research Summary

이 논문은 문서를 잠재 주제들의 확률적 혼합으로 설명하는 계층적 베이지안 모델을 제시한다. 주제 혼합을 확률 변수로 도입함으로써 기존 pLSI의 과적합과 일반화 불가능 문제를 해결하고, 문서와 단어의 교환가능성 가정 하에서 일관된 생성 모델을 구축한다. 정확 추론이 불가능한 구조를 변분 추론으로 해결하여 실용성을 확보했으며, 이후 토픽 모델 계열 연구의 표준적 출발점이 되었다.

# 11. Connection to Other Papers

- pLSI
- HDP
- Dynamic Topic Models
- Neural Topic Models
- Variational EM 계열

# 12. Personal Insight Log

- 핵심은 주제가 아니라 문서 수준 확률 변수
- 교환가능성 가정을 정면으로 받아들인 설계
- 이후 신경망 기반 토픽 모델도 이 구조를 변형한 형태