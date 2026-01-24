# 1. Reading Context(사전 맥락)

- 퍼셉트론의 한계를 어떻게 극복했는지 확인하기 위해
- 은닉층은 어떻게 학습될 수 있는가
- 다층 신경망은 표현력은 있지만 학습 방법이 없다는 것이 기존 통념

# 2. Problem Re-definition

- 출력층은 오차를 계산할 수 있으나, 은닉층은 정답 신호가 없음
- 오차를 출력층에서 은닉층으로 전달할 수 있다면 학습 가능
- 표현 학습의 핵심 문제는 구조가 아니라, 신호 전달 방식임

# 3. Core Contributions(논문의 핵심 기여)

### Backpropagation 알고리즘의 일반화

- 체인 룰을 이용해 다층 네트워크 전체의 gradient 계산
- 은닉층도 목적 함수에 기여하도록 만듦

### 은닉 표현의 학습 가능성 증명

- 은닉 유닛이 단순 중간 노드가 아니라 문제 구조를 압축/분해한 표현 공간을 형성

### Perceptron 비판에 대한 실질적 해결

- 단층 한계를 구조 추가 + 학습 규칙으로 극복
- 신경망 연구 부활의 결정적 계기

# 4. Method Analysis(설계 관점)

- 네트워크 구조
    - Input Layer
    - Hidden Layers
    - Output Layer
- Forward pass : 입력 → 출력 계산
- Backward pass : 오차 → gradient → 가중치 업데이트

# 5. Mathematical Formulation Log

- 목적 함수
    
    $$
    E = \sum(y - d)^2
    $$
    
- 출력층 gradient
    
    $$
    \frac{\partial E}{\partial w} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial net}
    $$
    
- 은닉층 핵심 식
    
    $$
    \frac{\partial E}{\partial y_j} = \sum_k \frac{\partial E}{\partial y_k} w_{jk}
    $$
    
- 은닉 유닛은 자신이 출력 오차에 얼마나 기여했는지로 학습
- 정답 없는 학습 문제를 수학적으로 해결

# 6. Experiment as Claim Verification

- XOR 문제 해결
- 대칭성 탐지, 구조적 분류 문제 해결
- 은닉층이 의미 있는 내부 코드를 형성함을 시각적으로 제시
- 단층 perceptron으로 불가능한 문제 해결
- 은닉 표현이 task 구조 반영

# 7. Limitations & Failure Modes

- 지역 최적점
- 학습 속도 느림
- 생물학적 타당성 부족
- 대규모 데이터/깊은 네트워크에는 당시 계산 자원 한계

# 8. Extension & Research Ideas

- Backprop → Stochastic Gradient Descent
- 깊은 은닉층 → Deep Learning
- Representation Learning의 출발점
- 이후 CNN, RNN, Transformer의 학습 원리적 기반

# 9. Code Strategy

- MLP + sigmoid
- 수치 미분 vs backprop gradient 비교
- 은닉층 가중치가 실제로 의미 있게 변화하는지

# 10. One-Paragraph Research Summary

이 논문은 오차를 출력층에서 은닉층으로 전파함으로써 다층 신경망의 학습을 가능하게 했고, 은닉 표현이 문제 구조를 스스로 학습할 수 있음을 보여주었다. 이는 현대 딥러닝의 수학적/개념적 출발점이다.

# 11. Connection to Other Papers

- Rosenblatt: Perceptron
- Minsky & Papert 비판 극복
- Deep Neural Networks, Representation Learning
- 모든 gradient-based deep learning

# 12. Personal Insight Log

- 학습의 본질은 표현을 바꾸는 것
- 은닉층은 계산 장치가 아니라 지식의 형태
- 이 논문 없이 딥러닝은 존재하지 않음