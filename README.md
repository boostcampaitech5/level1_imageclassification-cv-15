# Boostcamp-AI-Tech-Level1-BE1
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

- [부스트캠프 AI Tech](https://boostcamp.connect.or.kr/program_ai.html) - Level1. Mask Classification Competition  


<br />


## News

<<<<<<< HEAD
**`2023-04-19`**: 
- change age label threshold (60 -> 59)

=======
>>>>>>> a67a7da598a3c0ff98aef24e83da8287e9ac2c6e
**`2023-04-18` -  loss, optimizer Test**: 
- loss
  - before : f1 loss
  - ater : f1 loss * 1.5 + label smoothing loss * 1.0
- task change
  - before : 18 single labels
  - after : 3, 2, 3 multi labels
- optimizer : AdamW

**`2023-04-17` -  Augmentation Test**: 
<<<<<<< HEAD
- ~~Augmentation Setting : RandomRotation -15 ~ 15~~
=======
- Augmentation Setting : RandomRotation -15 ~ 15
>>>>>>> a67a7da598a3c0ff98aef24e83da8287e9ac2c6e
- face crop preprocessing
- test epoch : 30
- lr step : 10


**`2023-04-14` -  Model Performance Test**: 
- Final Model : ConvNext, MobileNet (For Fast Experiment)
- cross_entropy_loss -> F1-Loss
- optimizer Adam
- lr step : 4

**`2023-04-13` - 강나훈**: 
- baseline Sub Branch

<br />

## Repository 구조
- Repository 는 다음과 같은 구조로 구성되어있습니다. 

```
├── README.md
├── inference.py
├── train.py
├── loss.py
├── dataset.py
├── mask_data
│   ├── eval
│   ├── train
│   └── ._train
└── model.py
```



## 협업 규칙

- 커밋 메시지 컨벤션은 [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)을 따릅니다 
  - [commitizen](https://github.com/commitizen-tools/commitizen)을 사용하면 더욱 쉽게 커밋할 수 있습니다
- 작업은 기본적으로 별도의 브랜치를 생성하여 작업합니다. 작업이 완료되면 PR로 리뷰 받습니다
- PR 리뷰 후 머지 방식은 Squash & Merge를 따릅니다
  - Merge 전에 PR 제목을 되도록이면 convetional commit 형태로 만들어주세요



<br />

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/ejrtks1020"><img src="https://github.com/ejrtks1020.png" width="100px;" alt=""/><br /><sub><b>강나훈</b></sub></a><br /><a href="https://github.com/ejrtks1020" title="Code"></td>
    <td align="center"><a href="https://github.com/ejrtks1020"><img src="https://github.com/araseo.png" width="100px;" alt=""/><br /><sub><b>서아라</b></sub></a><br /><a href="https://github.com/araseo" title="Code"></td>
    <td align="center"><a href="https://github.com/adam1206"><img src="https://github.com/adam1206.png" width="100px;" alt=""/><br /><sub><b>이강민</b></sub></a><br /><a href="https://github.com/adam1206" title="Code"></td>
    <td align="center"><a href="https://github.com/Jeon-jisu"><img src="https://github.com/Jeon-jisu.png" width="100px;" alt=""/><br /><sub><b>전지수</b></sub></a><br /><a href="https://github.com/Jeon-jisu" title="Code"></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!