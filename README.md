# 2-4 스마트에디터의 그래머리 (문장 교정/교열) 기능 고도화

- 네이버 사용자가 작성한 문장을 문법적으로 맞는 문장으로 교정/교열 하는 모델을 만듭니다.

## Introduction
이 문제는 네이버 사용자가 작성한 문장을 문법적으로 맞는 문장으로 교정/교열하는 문제였습니다. 교정/교열의 종류는 띄어쓰기 교정, 붙여쓰기 교정, 시제 교정, 경어체 교정, 구두점 교정, 오탈자 교정 (위 분류에 없는 경우 모두 수렴), 윤문 처리 (더 매끄러운 문장)가 있었습니다.


이번 2라운드에서 Score 83.08502874637333으로 4등을 하였습니다. (내 300만원 ... ;ㅅ;)

<img width = "800" src = "https://user-images.githubusercontent.com/43025347/124918821-71f70c00-e030-11eb-980b-7571832a9e49.png">


## Model

### Data
- unlabeled corpus, labeled corpus가 주어졌습니다. 이번 대회에서는 unlabeled corpus가 매우 많아 이것을 어떻게 하면 효율적으로 사용할 것인지에 대해서 고민을 많이 한것 같습니다.

### Model Architecture
- <img width = "800" src = https://user-images.githubusercontent.com/43025347/131767613-675cfcaf-5904-45cf-8377-ca4906e99f0d.png>

- pytorch의 기본 transformer를 그대로 사용하였습니다.

### Train Step
- Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data를 기본 모델로 사용하였습니다.

- 첫번째로는 모델을 많은 단어를 이해하기 위해서 unlabeled corpus를 denoising Auto-encoder를 사용하여 학습합니다.

- 두번째로는 labeled corpus를 이용하여 모델을 학습한다.

- 세번째로는 학습된 모델을 가지고 unlabeled corpus를 beam search(n=5)를 사용하여 inference를 한후에 labeled corpus와 Semi-supervised learning을 진행하였습니다. 


## References
* https://arxiv.org/abs/1903.00138
* https://blog.est.ai/2020/11/ssl/


## AI rush가 끝나고...

대회를 시작하는 마음 가짐은 1라운드를 통과하는것을 목표로 잡았었습니다. 운이 좋게 1라운드를 통과하고 2라운드 스마트에디터 문법 교정 도우미 기능 고도화 부분에서 4위를 수상하였습니다.

대회를 진행하면서 저의 문제 접근 방법에 대한 아쉬운 부분을 설명하고자 합니다.

- 기본 모델을 너무 고수한게 제일 큰 문제가 아니였나 싶습니다. 1,2,3위 하신 분들의 발표자료를 대충 읽어봤는데 모두 모델을 수정하여 좋은결과를 낸것을 확인하였습니다.

- 저의 문제 접근 방법은 학습 방법, 추론방법 개선등에만 초점을 맞춰 극적인 결과를 내진 못했던것 같습니다.

이번 AI rush를 통해서 저의 부족한점을 알 수 있었고 보완하는 기회가 되었습니다. 


## Contacts
해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.


