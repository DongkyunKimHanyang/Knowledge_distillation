[Xu Lan et al., Knowledge Distillation by On-the-Fly Native Ensemble. ](https://arxiv.org/abs/1806.04606)  
 
논문 저자의 공식 [github 저장소](https://github.com/Lan1991Xu/ONE_NeurIPS2018)가 존재합니다.  
  
간단한 내용정리와 코드로 따라해보겠습니다.  

## 연구 동기
1. 기존의 Knowledge Distillation은 2단계로 나뉜다.  
  -High capacity를 갖는 Teacher 모델 생성  
  -Teacher 모델의 지식을 Student 모델에 증류  
  따라서 학습 프로세스의 시간도 길고 메모리 리소스도 커야한다.  
  
2. 위 문제를 해결 하기위한 peer online Distillation 기법들이 등장했다.  
  -작은 스튜던티끼리 서로 Distillation 하면서 학습하는 방식  
  하지만 스튜던트들이 Sub-optimal에 빠질수 있음, 연산량은 여전히 큼, forward, back propagation 과정이 너무 복잡함 등등의 단점이 존재  
  
## 연구 목적
연산량도 적고 간단하고 효과적인 Online Distillation 기법 제안.  
Teacher 모델을 따로 생성하는 프로세스도 생략하고, 모델의 일반화 성능도 향상시키고.

## 방법
<img width="701" alt="image" src="https://user-images.githubusercontent.com/87703352/159649545-d69a545b-74ba-4898-a548-8a4ad4ed0784.png">  

1.Resnet은 보통 4개의 블록으로 나눌수 있는데, 앞의 3개 블록을 Shared Layer로 배치 (그림 가장 왼쪽 회색)  

2.Shared layer 뒷단에 Branch Layer 여러개를 병렬로 이어 붙인다. Branch layer는 Resnet의 4번째 블록 + Linear layer로 구성된다  

3.Shared layer 뒷단에 Gate Layer를 하나 붙여서 Branch layer들의 Weighted average Ensemble에 사용   
  <img width="126" alt="image" src="https://user-images.githubusercontent.com/87703352/159650553-26b616cf-8dca-4c63-91b9-c22bdb91f41b.png">  
  즉, 위 식처럼 각 Branch layer 출력 z_i에 Gate layer의 출력을 계수로하는 앙상블 출력 생성  

4.앙상블 출력과 각 Branch의 출력을 Label과의 크로스 엔트로피로 학습  
<img width="241" alt="image" src="https://user-images.githubusercontent.com/87703352/159651385-82c83494-0486-4740-9787-cc76b7f2bc78.png">  

5.앙상블 출력과 Branch의 출력 사이에 Kullback-Leibler 적용. 즉, 각 Branch의 출력이 ensemble의 출력을 배우도록.  
<img width="305" alt="image" src="https://user-images.githubusercontent.com/87703352/159651639-a98085b0-fafc-4777-8812-b572bab3a01c.png">  

6.최종적으로 모든 loss를 더한 뒤, Back propagation.  
<img width="227" alt="image" src="https://user-images.githubusercontent.com/87703352/159651833-ed75aa72-f40f-4cce-b662-dff951833c01.png">  

결국에는 Ensemble 출력이 Teacher, 그리고 각 Branch의 출력이 Student가 되는 Online Distillation입니다.  

## CIFAR100 실험
위 방법론을 구현하여 CIFAR100 데이터셋에 실험해보겠습니다.  





