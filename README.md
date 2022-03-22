# Knowledge_distillation
Knowledge_distillation이란 거대 모델의 지식을 작은 모델로 증류하는것을 말합니다.
즉, 거대모델과 같은 성능을 보이는 더 작은 모델을 만드는것을 의미합니다.  
여기서 거대모델을 Teacher Network, 작은 모델을 Student Network라고 하는데요.  
모델 배포시, 거대 모델의 너무 큰 계산 복잡도가 문제가 되는 경우가 많아, 연산량이 작은 모델로 증류하여 배포하기 위함입니다.
![image](https://user-images.githubusercontent.com/87703352/159433587-987b81d5-8056-46ec-b07d-c98539d633d1.png)
