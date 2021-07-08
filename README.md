# 2-4 스마트에디터의 그래머리 (문장 교정/교열) 기능 고도화

- 네이버 사용자가 작성한 문장을 문법적으로 맞는 문장으로 교정/교열 하는 모델을 만듭니다.

## Introduction
이 문제는 네이버 사용자가 작성한 문장을 문법적으로 맞는 문장으로 교정/교열하는 문제였습니다. 교정/교열의 종류는 띄어쓰기 교정, 붙여쓰기 교정, 시제 교정, 경어체 교정, 구두점 교정, 오탈자 교정 (위 분류에 없는 경우 모두 수렴), 윤문 처리 (더 매끄러운 문장)가 있었습니다.


이번 2라운드에서 Score 83.08502874637333으로 4등을 하였습니다. (내 300만원 ... ;ㅅ;)

<img width = "800" src = "https://user-images.githubusercontent.com/43025347/124918821-71f70c00-e030-11eb-980b-7571832a9e49.png">


## Model

### Data
- unlabeled corpus, labeled corpus가 주어졌습니다. 이번 대회에서는 unlabeled corpus가 매우 많아 이것을 어떻게 하면 효율적으로 사용할 것인지에 대해서 고민을 많이 한것 같습니다.

### train step
- 제작 중 입니다

