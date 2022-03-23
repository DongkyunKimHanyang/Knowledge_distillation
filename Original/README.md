# Distilling the Knowledge in a Neural Network
[논문 링크](https://arxiv.org/abs/1503.02531)  
위 논문은 Knowledge Distillation이 처음으로 제시된 논문입니다.  
해당 논문의 내용을 간단히 정리하고, 코드로 작성해보겠습니다.  
  
연구 동기: 사용자가 많은 시스템에 딥러닝 모델을 배포하는것은 서비스의 지연시간을 늘리고 큰 자원을 필요로 한다.  
따라서 지식증류 (Knowledge Distillation) 을 통해 성능이 좋은 모델의 지식을 작은 모델에게 전이하고자 한다.  
여기서 성능이 좋은 모델을 그냥 Teacher, 증류 결과물은 작은 모델을 Student라고 하겠습니다.


# Distillation
지식 증류를 위해서 Teacher가 출력한 "Soft label"을 Student가 학습하도록 합니다.  
Soft label을 사용하면 Hard label에 비해 정보량이 많고, 분산이적고, 더 적은 데이터와 큰 학습률로 모델을 학습시킬 수 있다고 합니다.  
이때 Soft label의 entropy가 커야 효과가 좋으므로, Teacher의 출력에 Teamperatur Softmax (TS)를 적용합니다.

<img width="185" alt="image" src="https://user-images.githubusercontent.com/87703352/159440776-c1fc8020-805a-48ed-a9f1-efbd928d95d4.png">
위 식은 Teacher의 Logits에 TS를 적용한것입니다. T가 클수록 entropy가 큰 lable이 생성 됩니다.  
  
아래 식이 전제 적인 Loss function입니다. Student - Teacher의 크로스엔트로피 그리고 Student-Target의 크로스엔트로피의 가중합을 사용합니다.
![image](https://user-images.githubusercontent.com/87703352/159444646-692a9503-60b7-4a54-975c-026d92ed3a9e.png)  
첫번째 텀의 logits에 대한 미분값이 1/T^2로 스케일링 되기 때문에 T^2라는 계수를 붙여주는것이 균형적이라고 합니다.  
두번째 텀에는 TS가 아닌 그냥 Softmax를 적용한 것입니다.

# MNIST 실험
논문에 나온데로 MNIST데이터를 가지고 실험해보겠습니다.  
Teacher 모델의 크기는 1200의 hidden_size를 가진 2개의 Linear layer에 Relu 그리고 높은 확률의 Dropout을 적용했습니다.  

```
python3 Train_teacher.py
```
Test set에서 97.3%의 정확도를 보이는 Teacher 모델을 만들었습니다.

다음으로 작은사이즈의 Student에 Distillation 해보겠습니다.  
Student도 2개의 linear hidden과 relu 활성함수를 사용합니다.

Hidden_units = 300으로 줄였을때 성능입니다.
|T|Without Distill|With Distill|
|---|---|---|
|1|97.9%|97.8%|
|2||98.0%|
|5||98.3%|
|10||98.4%|

크로스 엔트로피만 썼을때보다, Distill loss를 추가했을때 Temperature가 증가함에 따라서 성능이 향상됨을 보여줍니다. 
  
Hidden_units = 32 일때 성능입니다.
|T|Without Distill|With Distill|
|---|---|---|
|1|97.1%|96.9%|
|2||96.5%|
|5||96.7%|
|10||97.0%|

모델의 크기가 작을때는 Distill loss를 적용하지 않는것이 더 성능이 좋음을 보여줍니다. 즉, 그냥 hard label로 학습시키는것이 더 성능이 좋았습니다.  
Distillation이 정말 모델 압축에 효과가 있는것인지 좀더 다양하게 실험을 해봐야 할 것 같습니다.
