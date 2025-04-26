---
layout: single
title: "[논문리뷰] BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
editor: 박상원
comments: true
mathjax: true
---
# BEVFormer 논문 리뷰

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%;">
  <video controls style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
    <source src="https://private-user-images.githubusercontent.com/27915819/161392594-fc0082f7-5c37-4919-830a-2dd423c1d025.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ0NDgzMTcsIm5iZiI6MTc0NDQ0ODAxNywicGF0aCI6Ii8yNzkxNTgxOS8xNjEzOTI1OTQtZmMwMDgyZjctNWMzNy00OTE5LTgzMGEtMmRkNDIzYzFkMDI1Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEyVDA4NTMzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY3ZmIzNDU5OGM3ZjRkMGI4OWRiZGJkMDE2OGIzMTMxM2YzZmExYjM2YTU2YzgwOWUwNzhkNTNmMmY5M2FjY2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Ol3LWy0cZ24UT3_gtcUXgtV4_ni5lhVgCvOjxiVUpVk" type="video/mp4">
    브라우저가 동영상을 지원하지 않습니다.
  </video>
</div>


Bevformer의 BEV-segmentation/object detection 결과

기존의 BEV context를 만드는 방식들은 multi camera 이미지 정보를 기하학적인 방법을 통해 직접 BEV grid에 matching 해주는 것이었던 것에 반해,  

- BEVFormer는 각 그리드 셀이 담당하는 영역을 나타내는 BEV 쿼리를 활용해, 명시적인 깊이 추정 없이도 이미지 특징을 BEV 공간으로 효과적으로 matching 할 수 있게 했다.
- 공간 교차 어텐션 모듈에서는 Deformable Attention을 확장해, 각 BEV 쿼리가 다중 카메라의 관심 영역에서 필요한 정보만 선택적으로 추출하여 computation-cost를 줄였다.
- temporal self-attention 모듈을 통해 과거 프레임의 BEV 특징을 재활용, 움직이는 객체의 속도 추정 및 가려진 객체의 검출 성능을 크게 향상시켰다.

즉, transformer를 적극적으로 활용하여 BEV representation 성능을 올린 모델이라 볼 수 있다.

이러한 차별화된 구조 덕분에 BEVFormer는 기존 방식들이 가진 깊이 추정의 부정확성 문제를 회피하고, 더 안정적이고 효율적으로 3D 객체 감지와 맵 분할 등의 작업을 수행하여 당시 기준으로 BEV-representation SOTA를 찍었던 모델이다.

![그림1. BEV segmentation Leaderboard on nuScenes Data - [Link](https://paperswithcode.com/sota/bird-s-eye-view-semantic-segmentation-on)](../figs/2025-04-14-BEVFormer/image.png)

그림1. BEV segmentation Leaderboard on nuScenes Data - [Link](https://paperswithcode.com/sota/bird-s-eye-view-semantic-segmentation-on)

논문 리뷰는 아래와 같은 흐름으로 구성했다.

## Abstract 및 I. Introduction

### A. Multi-camera based BEV representation의 필요성

![그림2. Tesla FSD의 BEV representation을 활용한 UI - [Link](https://www.youtube.com/watch?v=Jn2sEWXeJOs&t=550s)](../figs/2025-04-14-BEVFormer/image%201.png)

그림2. Tesla FSD의 BEV representation을 활용한 UI - [Link](https://www.youtube.com/watch?v=Jn2sEWXeJOs&t=550s)

자율주행 자동차는 주변 환경을 정확히 인지하기 위해 **3차원 공간**에서의 객체 위치와 지도를 파악해야 한다. **Bird’s-Eye-View(BEV)**, 즉 하늘에서 내려다본 듯한 시점의 표현은 이러한 주변 상황을 직관적으로 보여주는 활용할 수 있는 방식이다. BEV map 위에서는 물체들의 위치와 크기가 한눈에 보이며, 주행 가능한 도로나 차선 등의 정보를 통합적으로 표현할 수 있어 **주행 경로 계획**이나 복잡한 교통 상황 이해에 유리하다.

이 BEVformer Lidar를 사용하지 않고, 카메라 이미지들만으로만 BEV feature를 생성하는 모델이다. Lidar의 성능이 좋음에도 불구하고 사용하지 않는 연구를 하는 이유는 카메라가 Lidar에 비해 훨씬 값이 싸고, 장거리 detection에 더 장점이 있으며, vision-based 도로 요소 (ex 신호등, 정지선, 표지판 등)을 인식하는데 더 바람직한 이점이 있어서 이다.  

### B. 기존 앞선 방법들의 문제점 (2D to 3D)

![그림2. **깊이 추정 기반**의 BEV 생성 방법의 대표주자인 LSS의 처참한 depth estimation 능력…](../figs/2025-04-14-BEVFormer/image%202.png)

그림2. **깊이 추정 기반**의 BEV 생성 방법의 대표주자인 LSS의 처참한 depth estimation 능력…

문제는 일반적인 카메라 영상은 정면 또는 측면 시점의 **2D 이미지**이기 때문에, 여러 카메라에서 얻은 2D 정보를 합쳐 하나의 3D **BEV 표현**으로 변환하는 일이 쉽지 않다는 점이다. 카메라만으로 BEV 표현을 만들려면 각 픽셀의 깊이(거리)를 추정해 3D 공간으로 투영하거나, 여러 뷰의 정보를 모아야 하는데, 이 과정에서 오차가 누적될 위험이 있다. 예를 들어, **깊이 추정 기반**의 BEV 생성 방법들은 깊이 예측이 조금만 틀려도 결과 BEV 특징 맵이 부정확해지고 최종 객체 검출 성능까지 떨어지는 문제가 보고되었다.  ([BEVDepth](https://arxiv.org/abs/2206.10092))

이러한 한계를 극복하기 위해, **BEVFormer** 논문에서는 아예 **Transformer**의 어텐션 메커니즘을 통해 깊이 추정을 거치지 않고 **학습을 통해 기하학 정보를 추정할 필요 없이 바로 BEV 표현을 생성**하는 새로운 접근을 제시했다. 

![그림3. **RNN의 hidden state** 시각화](../figs/2025-04-14-BEVFormer/image%203.png)

그림3. **RNN의 hidden state** 시각화

또 한 가지 고려할 점은 **시간적 정보**이다. 사람은 영상을 볼 때 이전 순간들의 기억을 통해 현재 가려진 물체를 유추하거나 움직이는 물체의 속도를 가늠합니다. 자율주행 인지에서도 연속된 프레임 간의 **Temporal (시간적) 정보**를 활용하면, 정지된 이미지 한 장으로는 파악하기 어려운 물체의 **속도 추정**이나 **가려진 물체 검출**에 큰 도움이 될 수 있다. 그러나 과거의 BEV 생성 방법들은 주로 **한 시점**의 멀티 카메라 영상만 처리하고, 시간적 정보 활용에는 소극적이었다. 연속된 시점의 피처를 단순 누적하면 **계산 부담**이 크고 노이즈가 섞일 우려가 있기 때문이다. BEVFormer는 여기에 착안하여 **RNN의 hidden state**처럼 이전 시점의 BEV 표현을 현재 시점에 전달하는 구조를 도입한다. 이러한 **시공간적(Spatiotemporal) Transformer**기반 접근을 통해, BEVFormer는 여러 카메라 영상들을 통합하면서도 시간에 따른 변화도 반영한 **강력한 BEV 표현**을 학습하고자 했다.

## II. Related work

### **A. 트랜스포머 기반 2D 인식 (Transformer-based 2D Perception)**

  Transformer 구조를 기반으로 객체 인식을 수행할 수 있도록 만든 모델은 DETR이다. Object Query를 이용해 전역 Multi-Head Attention을 통해서 2D 객체 감지를 수행하지만, 학습 시간이 오래 걸리는 문제가 있었다. 이를 해결하기 위해 제안된 Deformable DETR는 각 쿼리가 전체 key에 대해서 연산하는 것이 아닌 미리 정해진 로컬 영역 내에서 샘플링하여 효율적으로 어텐션을 계산한다. BEVFormer는 이 변형 가능한 어텐션 메커니즘을 확장하여 3D 인식 작업에 적용함으로써, 다중 카메라로부터 얻은 정보의 BEV query상에 효율적인 matching을 가능하게 한다.

  이를 자세히 설명하면 다음과 같다.

**1. Multi-Head Attention (MHA)**

기존 Transformer에서 사용하는 Multi-Head Attention은 모든 입력 토큰(또는 feature 위치)과의 전역적인 상호작용을 수행한다. 간단히 수식으로 나타내면:

**$\mathbb{\text{MultiHeadAttention}(q, X) = \sum_{i=1}^{N_{\text{head}}} W_i \left(\sum_{j=1}^{N_{\text{key}}} A_{ij} \cdot W'_i\, x_j\right)}$**

여기서:

- $q$는 query 벡터,
- $x_j$는 입력 feature의  $j$번째 요소,
- $W_i$ 와 $W'_i$는 각 헤드별 학습 가능한 가중치 행렬,
- $A_{ij}$는 query와 각 key 사이의 어텐션 가중치로, 보통 softmax를 통해 정규화되어 모든 $j$ 에 대해 $\sum_j A_{ij} = 1$ 을 만족한다.

이 방식은 입력 전체와의 상호작용을 계산하므로, 입력 크기가 커지면 계산 비용이 $O(N^2)$ 로 증가하는 단점이 있다.

**2. Deformable Attention**

![그림 4. Deformable attention 모듈 - [Link](https://arxiv.org/abs/2010.04159)](../figs/2025-04-14-BEVFormer/image%204.png)

그림 4. Deformable attention 모듈 - [Link](https://arxiv.org/abs/2010.04159)

Deformable Attention은 위의 전역적인 상호작용 대신, **미리 정의된 지역 영역**(local region) 내에서 선택적으로 정보를 활용한다. 수식으로는 다음과 같이 표현할 수 있다:

$\text{DeformAttention}(q, p, X) = \sum_{i=1}^{N_{\text{head}}} W_i \left(\sum_{j=1}^{N_{\text{key}}} A_{ij} \cdot W'i\, x(p + \Delta p{ij})\right)$

여기서:

- $p$는 해당 query가 기준으로 하는 참조 위치(reference point)이다.
- $\Delta p_{ij}$  는 각 헤드 $i$ 와 샘플 $j$ 에 대해 학습되는 위치 오프셋으로, 이를 통해 query는 전역이 아닌, 국부적으로 중요한 영역(즉, "관심 영역")만을 선택적으로 샘플링한다.
- $x(p + \Delta p_{ij})$는 보간법(예: bilinear interpolation)을 통해 위치 $p + \Delta p_{ij}$에서의 feature 값을 추출하는 함수이다.

이 방식은 각 query가 전체 입력이 아닌, 소수의 (예를 들어, $N_{\text{key}} = 4$ ) 샘플링 포인트만 고려하기 때문에 계산 비용을 크게 줄일 수 있다.

자세한 내용은 Deformable DETR 논문을 통해 알아볼 수 있다.

### **B. 카메라 기반 3D 인식 (Camera-based 3D Perception)**

- **단안 프레임워크와 후처리 방식:**

초기 방법들은 단일 이미지에서 3D 바운딩 박스를 예측하거나, 서로 다른 카메라 뷰에서 개별적으로 2D 감지를 수행한 후 크로스 카메라 후처리로 3D 정보를 복원하는 방식이었다.

- **BEV 기반 접근:**

최근 연구들은 이미지의 깊이 정보를 활용해 2D 특징을 조감도(BEV) 표현으로 변환하는 방법(Lift-Splat, BEVDet 등)을 사용하지만, 이 방식은 깊이 추정 모듈의 부정확성에 취약하다.

- **3D 쿼리 기반 방법:**

DETR3D와 같은 방법은 학습 가능한 3D 쿼리를 2D 이미지에 투영하여 직접 3D 바운딩 박스를 예측한다. 그러나 이들 방식은 BEV 표현의 장점을 충분히 활용하지 못하는 단점이 있다.

## 3. BEVFormer

### 3.1 Overall Architecture

![그림 5. **Overall architecture of BEVFormer.**](../figs/2025-04-14-BEVFormer/image%205.png)

그림 5. **Overall architecture of BEVFormer.**

BEVformer의 Architecture는 Transformer의 Encoder 구조를 따른다. 하지만, 크게 세 가지 핵심 구성요소가 Encoder 구조에 추가되었고, 이들을 통해 BEVformer의 구조를 설명할 수 있다.

- BEV Queries
- Spatial Cross-attention Layer
- Temporal self-attention Layer

### 3.2 BEV Queries

- **정의:**
BEV 쿼리는 $Q \in \mathbb{R}^{H \times W \times C}$  형태로 미리 정의된, 학습 가능한 파라미터 집합이다. 여기서  $H$ 와  $W$ 는 BEV 평면의 높이와 너비,  $C$ 는 feature의 차원이다.
- **역할:**
각 BEV 쿼리  $Q_p$  (여기서  $p=(x,y)$  위치에 해당)는 BEV 평면 상의 하나의 그리드 셀에 대응하며, 이 셀은 실제 세계의 $s$  미터 크기를 나타낸다. 기본적으로 BEV 평면의 중심은 에고 차량(ego car)의 위치로 설정되며, 일반적인 위치 임베딩을 추가하여 쿼리의 공간 정보를 보강한다.
- **의의:**
이 BEV 쿼리들은 이후 단계에서 다중 카메라 이미지와 과거 시간의 BEV 특징으로부터 정보를 조회(lookup)하고 집계하는 “질의(query)” 역할을 수행한다.

### **3.3 Spatial Cross-attention**

3차원 BEV Queries를 정의했으면, CNN으로 이루어진 model backbone을 통해 얻은 image feature를 key와 value로 사용해서 Query와 연결 해야한다. 이 논문에서 연결방법으로 제시한 방법은 Spatial Cross-attention 일명 SCA 모듈이다. 

- **문제 인식:**
BEVformer는 Attention을 활용해서 Query와 image feature를 이어준다. 다중 카메라 이미지 입력은 매우 큰 규모의 데이터를 포함하므로, 모든 이미지 위치와 전역적으로 상호작용하는 전통적인 Multi-Head Attention (MHA)의 계산 비용은 너무 크다.
- **해결책 – Deformable Attention:**
BEVFormer는 Deformable Attention 메커니즘을 변형하여 사용한다.

![그림 6. SCA모듈 설명](../figs/2025-04-14-BEVFormer/image%206.png)

그림 6. SCA모듈 설명

**1. BEV 쿼리의 “리프트” 및 3D 참조점 샘플링**

- BEV 평면의 각 쿼리  $Q_p$  (여기서  $p=(x,y)$)를 기둥(pillar) 형태의 쿼리로 Lift한다.
- 각 기둥에서  $N_{\text{ref}}$ 개의 3D 참조점, 즉, $\{(x', y', z'j) \mid j=1,2,\dots,N{\text{ref}}\}$
를 샘플링한다.
여기서  $(x', y')$ 는  $Q_p$ 의 실제 2D 위치 (에고 차량 기준)이며,  $z'_j$ 는 미리 정의된 앵커 높이(anchor heights)이다.

---

**2. 3D 참조점의 2D 투영**

![그림 7. lift된 점을 image에 사용하여 deformable-attention의 refference point를 찾는 과정 시각화](../figs/2025-04-14-BEVFormer/image%207.png)

그림 7. lift된 점을 image에 사용하여 deformable-attention의 refference point를 찾는 과정 시각화

- 각 3D 참조점은 카메라의 프로젝션 매트릭스  $T_i$ 를 사용하여 해당 카메라 뷰의 2D 좌표로 투영된다.
- 이를 위한 투영 함수는 다음과 같이 정의된다.
$\mathcal{P}(p, i, j) = (x_{ij}, y_{ij}), \quad \text{where} \quad z_{ij} \cdot \begin{bmatrix} x_{ij} \\ y_{ij} \\ 1 \end{bmatrix} = T_i \cdot \begin{bmatrix} x' \\ y' \\ z'_j \\ 1 \end{bmatrix}.$
- 한 BEV 쿼리  $Q_p$ 에 대해, 투영된 2D 포인트들은 모든 카메라 뷰에서 나타나지 않고, 일부 뷰에서만 “적중(hit)”하게 된다. 이 적중된 카메라 뷰의 집합을  $\mathcal{V}_{\text{hit}}$ 라고 한다.

---

**3. 공간 크로스 어텐션 (Spatial Cross-Attention, SCA)**

- 각 BEV 쿼리  Q_p 는, 적중된 각 카메라 뷰  $i \in \mathcal{V}_{\text{hit}}$ **에서, 샘플링한  $*N{\text{ref}}*$ 개의 2D 참조점 주위의 특징을 추출한다.
- 이를 위해, Deformable Attention을 사용하여, 각 참조점  $j$ 에 대해 해당 뷰의 특징  $F^i_t$ 에서 값을 추출한다.
- 이 과정을 수식으로 나타내면:

![image.png](../figs/2025-04-14-BEVFormer/image%208.png)

- 여기서
    - $F^i_t$ 는  $i$ -번째 카메라 뷰의 특징,
    - $\text{DeformAttn}(\cdot)$ 는 Deformable Attention 연산으로, query  $Q_p$ 와 투영된 참조점  $\mathcal{P}(p, i, j)$ 를 기반으로 해당 위치의 특징을 샘플링하고 가중 합을 수행한다.
- 최종적으로, BEV 쿼리  $Q_p$ 는 적중된 카메라 뷰들의 특징을 평균하여 공간 정보를 matching한다.

이러한 공간 크로스 어텐션은, 전역 어텐션보다 계산 비용을 크게 줄이면서도 각 BEV 쿼리가 필요한 정보만 효과적으로 추출할 수 있도록 돕는다.

### **3.4 Temporal Self-attention**

![그림 8. temporal self-attention 고정 시각화 (output이 spatial cross-attention input으로 들어감)](../figs/2025-04-14-BEVFormer/image%209.png)

그림 8. temporal self-attention 고정 시각화 (output이 spatial cross-attention input으로 들어감)

**배경 및 필요성**

- 단일 정적 이미지에서는 움직이는 객체의 속도 추정이나 가려진 객체 검출이 어렵다.
- 시간적 단서가 없다면, 과거 정보 없이 현재 상태만으로는 충분한 3D 정보를 복원하기 힘들다.

**주요 아이디어**

![그림 9. Aligned BEV feature ( $B_{t-1}$ )와 Current BEV Queries ( $Q_t$ )](../figs/2025-04-14-BEVFormer/image%2010.png)

그림 9. Aligned BEV feature ( $B_{t-1}$ )와 Current BEV Queries ( $Q_t$ )

- **히스토리 BEV 특징 활용:**
    - 이전 타임스탬프 $t-1$에서 추출한 BEV 특징 $B_{t-1}$을, 에고 모션(ego-motion)을 고려해 정렬(alignment)한다. ($B_{t-1}$를 t 시각의 위치를 좌표계 기준으로 shift해서 정렬함)
    - 정렬된 특징을 $B'_{t-1}$로 표기하여, 현재 BEV 쿼리 $Q$와 동일한 실제 위치를 나타내도록 만든다.
- **시간적 셀프 어텐션의 작동 원리:**
    - 각 현재 BEV 쿼리  $Q_p$  (위치 $p=(x,y)$)는, 정렬된 이전 BEV 특징과 현재 BEV 쿼리 자체  Q 를 모두 참조하여 시간 정보를 matching한다.
    - 이를 위해, Deformable Attention을 확장한 Temporal Self-Attention (TSA) 모듈을 사용한다.

**수식**

TSA는 다음과 같이 표현된다.

![image.png](../figs/2025-04-14-BEVFormer/image%2011.png)

- 여기서,
    - $V$ 는 현재 BEV 쿼리  $Q$ 와 정렬된 이전 BEV 특징  $B'_{t-1}$ 의 집합이다.
    - DeformAttn 연산은 앞서 설명한 것과 같이, query  $Q_p$ 와 기준 위치  $p$ 를 중심으로,  $V$ 로부터 국부 영역의 정보를 샘플링해 가중 합을 수행한다.

**특이 사항**

- **오프셋 예측:**
    - 시간적 셀프 어텐션에서는, 오프셋  $\Delta p$ 가 단순히 query  $Q_p$ 만으로 예측되는 것이 아니라,  $Q$ 와  $B'_{t-1}$ 의 결합으로 예측된다.
- **첫 번째 샘플 처리:**
    - 시퀀스의 첫 번째 샘플에는 과거 BEV 특징이 없으므로,  $\{Q, B'_{t-1}\}$  대신  $\{Q, Q\}$ 를 사용하여 단순 self-attention으로 동작한다.

**의의**

- 여러 타임스탬프의 BEV 특징을 단순히 스태킹하는 것보다, TSA를 통해 시간적 연속성(temporal dependency)을 효과적으로 모델링할 수 있어 계산 비용 및 방해 정보(interference)를 줄일 수 있다.

### **3.5 Applications of BEV Features**

**다목적 2D 특징 맵으로서의 BEV**

- BEV 특징  $B_t \in \mathbb{R}^{H \times W \times C}$는 2D 형태의 특징 맵으로, 다양한 자율주행 인식 작업에 재사용 가능하다.

**응용 예시**

1. **3D 객체 감지:**
    - BEV 특징을 입력으로 받아 3D 바운딩 박스와 속도 정보를 예측하는 detection head (예: Deformable DETR 기반)를 구성한다.
    - 2D 바운딩 박스 대신 3D 바운딩 박스와 물체의 속도를 직접 예측하며, 종단 간 학습이 가능하도록 NMS(post-processing) 없이 동작한다.
2. **지도 세분화:**
    - BEV 특징은 일반적인 2D 의미론적 세분화 기법(예: Panoptic SegFormer의 마스크 디코더)과 결합해, 3D 공간 내 도로, 차선, 차량 등의 영역을 세분화하는 데 사용된다.
    - 클래스 고정 쿼리를 사용해 각 의미 범주(자동차, 차량, 도로, 차선 등)의 세분화 마스크를 생성한다.

### **3.6 Implementations Details**

**교육 단계 (Training Phase)**

- **샘플링 전략:**
    - 각 타임스탬프  $t$ 에 대해, 지난 2초간의 연속 시퀀스에서 무작위로 3개의 추가 샘플을 선택하여 총 4개 샘플 ($t-3, t-2, t-1, t$)을 사용한다.
    - 이 전략은 다양한 에고 모션(ego-motion)을 보강하는 데이터 증강 효과를 가져온다.
- **BEV 특징 생성:**
    - $t-3, t-2, t-1$의 샘플은 반복적으로 BEV 특징을 생성하며, 이 단계에서는 그래디언트가 전달되지 않는다.
    - $t-3$의 첫 번째 샘플에는 과거 BEV가 없으므로, Temporal Self-Attention이 단순 self-attention으로 동작한다.
    - $t$ 시점에서는 다중 카메라 이미지와 이전 BEV 특징 $B_{t-1}$을 사용해 BEV 특징 $B_t$를 생성한다. 이때 시간 및 공간 단서가 모두 결합된다.
- **손실 함수 계산:**
    - 최종적으로 생성된 BEV 특징 $B_t$는 3D 객체 감지 및 지도 세분화 헤드에 입력되어, 해당 작업의 손실 함수를 통해 모델이 학습된다.

**추론 단계 (Inference Phase)**

- **온라인 추론 전략:**
    - 비디오 시퀀스의 각 프레임을 시간순으로 처리한다.
    - 이전 타임스탬프의 BEV 특징 $B_{t-1}$를 저장하고 다음 프레임의 처리 시 재사용하여 시간 효율적인 온라인 추론을 수행한다.
    - 이 방식은 실제 자율주행 시스템에서 요구하는 실시간 처리와 일관성을 제공한다.

## **4 Experiments**

![그림 10. **Visualization results of BEVFormer on nuScenes val set.**](../figs/2025-04-14-BEVFormer/image%2012.png)

그림 10. **Visualization results of BEVFormer on nuScenes val set.**

BEVFormer의 성능을 검증하기 위해, 논문에서는 두 가지 대표적인 자율주행 데이터셋인 nuScenes와 Waymo Open Dataset을 사용했다. 이 섹션에서는 데이터셋의 구성, 실험 환경 설정, 그리고 주요 실험 결과에 대해 작성되어있다.

### **4.1 Datasets**

**nuScenes 데이터셋**

- **구성:**
    - 약 1,000개의 장면(scene)이 포함되며, 각 장면은 약 20초 동안 지속된다.
    - 주요 샘플은 2Hz로 주석(annotation)되어 있으며, 각 샘플은 6개의 카메라로부터 촬영된 RGB 이미지와 360° 수평 시야를 제공한다.
    - 10개의 범주에 대해 총 1.4M개의 3D 바운딩 박스가 주석으로 제공된다.
- **평가 메트릭:**
    - 평균 정밀도(mAP)는 3D IoU 대신, 지상 평면의 중심 거리로 계산된다.
    - 또한, ATE(Translation Error), ASE(Scale Error), AOE(Orientation Error), AVE(Velocity Error), AAE(Attribute Error) 등 5가지 TP(True Positive) 메트릭이 함께 제공되며, 이를 종합한 nuScenes Detection Score (NDS)가 정의되어 있다.
    
    ![image.png](../figs/2025-04-14-BEVFormer/image%2013.png)
    

**Waymo Open Dataset**

- **구성:**
    - 798개의 훈련 시퀀스와 202개의 검증 시퀀스로 이루어진 대규모 데이터셋이다.
    - 각 프레임마다 5개의 이미지가 제공되나, 수평 시야는 약 252°에 그친다. 하지만, 레이블은 에고 차량 주변 360°를 커버한다.
- **전처리:**
    - 훈련 및 검증 시, 이미지에서 보이지 않는 경계 상자는 제거합니다.
    - 대규모 데이터 특성상, 훈련 시 5번째 프레임마다 샘플링하여 하위 집합으로 사용하며, 차량 범주만을 감지한다.
- **평가 메트릭:**
    - 3D IoU 임계값 0.5 및 0.7을 사용하여 mAP를 계산한다.

### **4.2 Experimental Settings**

논문에서는 이전 연구 [45, 47, 31]를 따르며, 두 가지 백본 네트워크를 사용한다.

1. **ResNet101-DCN**
    - FCOS3D 체크포인트에서 초기화됨.
2. **VoVnet-99**
    - DD3D 체크포인트에서 초기화됨.

**기본 설정**

- **FPN 출력:**
    - FPN의 다중 스케일 feature는  $\frac{1}{16}$ ,  $\frac{1}{32}$ ,  $\frac{1}{64}$  크기로 사용되며, 채널 수  $C = 256$ 이다.
- **BEV 쿼리 설정 (nuScenes 실험):**
    - BEV 쿼리의 기본 크기는 $200 \times 200$.
    - 인식 범위는 $X, Y$ 축 각각 $[-51.2\,m, 51.2\,m]$.
    - BEV 그리드의 해상도 $s = 0.512\,m$.
    - 학습 가능한 위치 임베딩을 추가한다.
- **BEV 인코더:**
    - 총 6개의 인코더 레이어로 구성되어 있으며, 각 레이어에서 BEV 쿼리를 지속적으로 정제(refine)한다.
    - 각 인코더의 입력 BEV feature는 이전 타임스탬프의  $B_{t-1}$ 이며, 이 부분은 학습 시 그래디언트가 전달되지 않는다.
- **공간 크로스 어텐션:**
    - 각 로컬 BEV 쿼리 당, 3D 공간에서  $N_{\text{ref}} = 4$ 개의 타겟 포인트가 선택된다.
    - 미리 정의된 높이 앵커는 $[-5\,m, 3\,m]$ 범위에서 균일하게 샘플링된다.
    - 각 2D 참조점 주위로 각 헤드마다 4개의 샘플링 포인트를 사용한다.
- **훈련:**
    - 기본적으로 24 epoch, 학습률 $2 \times 10^{-4}$로 모델을 훈련한다.

**Waymo 실험 설정**

- 카메라 시스템의 특성으로 인해, BEV 쿼리의 기본 공간 모양을 $300 \times 220$으로 변경한다.
- 인식 범위는 $X$축 $[-35.0\,m, 75.0\,m]$와 $Y$축 $[-75.0\,m, 75.0\,m]$로 설정한다.
- BEV 그리드의 해상도는 $0.5\,m$로 조정되며, 에고 차량의 위치는 BEV 내에서 $(70, 150)$에 해당한다.

**Baselines**

- BEVFormer와 공정한 비교를 위해, VPN과 Lift-Splat 를 사용하여 BEV 생성 방법을 대체한 버전과 비교한다.
- 또한, Temporal 정보를 사용하지 않는 정적 버전인 BEVFormer-S도 함께 평가한다.

### **4.3 3D Object Detection Results**

논문에서는 nuScenes와 Waymo 두 데이터셋에 대해 3D 객체 감지 성능을 평가하였다.

![image.png](../figs/2025-04-14-BEVFormer/image%2014.png)

- **nuScenes 테스트 및 검증 결과:**
    - BEVFormer는 nuScenes 테스트 셋에서 NDS 56.9%를 달성하였으며, 이는 이전 최첨단 모델인 DETR3D보다 약 9.0포인트 높은 성능이다.
    - 또한, BEVFormer는 LiDAR 기반 모델(예: SSN, PointPainting)과도 견줄 수 있는 성능을 보인다.
- **Waymo 실험 결과:**
    - IoU 기준 0.5 및 0.7로 차량 범주를 평가하였으며, nuScenes 메트릭도 함께 사용해 성능을 비교했다.
    - BEVFormer는 DETR3D 및 단안 방식(CaDNN)과 비교하여, 평균 정밀도(APH) 및 NDS에서 상당한 개선을 보였다.

### **4.4 Multi-tasks Perception Results**

![image.png](../figs/2025-04-14-BEVFormer/image%2015.png)

- **공동 훈련:**
    - BEVFormer는 3D 객체 감지와 지도 세분화 작업을 동시에 학습할 수 있도록 설계되었다.
    - Joint training(감지와 세분화를 함께 학습) 시, BEVFormer는 단독 학습에 비해 감지 작업에서 11.0포인트, 차선 세분화에서 5.6포인트 향상된 성능을 보인다.
- **음의 전달 (Negative Transfer):**
    - 다중 작업을 함께 학습할 때, 도로 및 차선 세분화에서는 개별 작업 학습에 비해 성능이 약간 떨어지는 현상이 나타나기도 한다.

### **4.5 Ablation Study**

![image.png](../figs/2025-04-14-BEVFormer/image%2016.png)

- **공간 크로스 어텐션 효과:**
    - BEVFormer-S를 사용하여, 변형 가능한 어텐션(deformable attention) 기반 공간 크로스 어텐션이 전역 어텐션이나 단순 참조점 상호작용보다 우수함을 확인했다.
- **시간적 셀프 어텐션 효과:**
    - 시간 정보를 추가하면, 속도 추정 정확도, 물체 위치 및 방향 예측이 크게 향상되며, 특히 가시성이 낮은(occluded) 물체에 대한 재현율이 개선되었다.
- **모델 규모와 대기 시간:**
    - 다양한 구성(예: 인코더 레이어 수, BEV 쿼리의 크기, 다중 스케일 vs 단일 스케일)을 실험하여, 성능과 추론 속도 간의 균형을 평가하였다.
    - 예를 들어, 인코더 레이어 수를 1개로 줄이면 NDS는 약간 감소하지만, 추론 속도는 크게 개선되어 최대 7ms까지 단축된다.

![image.png](../figs/2025-04-14-BEVFormer/image%2017.png)

## **5 Discussion and Conclusion**

BEVFormer 논문은 **카메라 기반 3D 인지의 가능성**을 한 단계 끌어올렸다는 평가를 받는것 같다. 가장 큰 의의는, **굳이 깊이를 명시적으로 추정하지 않아도 학습만으로 BEV 표현을 만들 수 있다**는 것을 보여준 점이다. 이를 통해 **오차 누적의 감소**가 이루어졌고**,** 모델이 필요로 하는 정보만 골라보는 어텐션 덕분에, 중간 단계에서 잘못된 깊이나 3D 점이 만들어져서 발생하는 오류를 줄이고 **End-to-End로 최적화**할 수 있었다고 한다.

# 참고자료

https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file

https://docs.google.com/presentation/d/1fTjuSKpj_-KRjUACr8o5TbKetAXlWrmpaorVvfCTZUA/edit?usp=sharing

https://arxiv.org/abs/2203.17270

https://medium.com/@parkie0517/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-3b6f57b90b3a