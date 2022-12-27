# MRC_with_non-answerable_questions

## 💡 경제기사 QA 데이터셋을 활용하여 지문 내에 답이 존재한다면 답의 위치를 추론하고, 지문 내에 답이 존재하지 않는다면 아무 답을 하지 않는 모델

모델의 최대 토큰 수는 512자이고 QA 데이터는 Paragraph와 Question이 한번에 들어가기 때문에 모델이 읽을 수 있는 지문이 짧습니다.  
지문을 단순히 자르게 되면 답이 있는 지문이라도 답이 잘린 지문 내에 포함되지 않을 수 있고 잘린 지문 사이에 존재하게 될 수도 있어 한 문장 정도가 겹치게 350자 단위로 잘라서 사용하였습니다.

### Retrospective Reader for Machine Reading Comprehension 참고

```python
@article{zhang2020survey,
title={Machine Reading Comprehension: The Role of Contextualized Language Models and Beyond},
author={Zhang, Zhuosheng and Zhao, Hai and Wang, Rui},
journal={arXiv preprint arXiv:2005.06249},
year={2020}
}
@inproceedings{zhang2021retrospective,
title={Retrospective reader for machine reading comprehension},
author={Zhang, Zhuosheng and Yang, Junjie and Zhao, Hai},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={35},
number={16},
pages={14506--14514},
year={2021}
}
```

Retro Reader 모델은 두개의 모듈을 병렬적으로 사용합니다.  
Sketch Module은 Sequence Classification 하위태스크를 사용해 응답가능 여부를 판단하고  
Intensive Module은 QuestionAnswering 하위태스크를 사용해 응답가능 여부를 판단하는 동시에 지문에서의 정답의 위치를 추론합니다.  
또한 Sketchy Module에서의 로짓과 Intensive Module에서 정답 시작 위치와 종료 위치 확률값을 점수로 활용해 앙상블하여 응답가능여부를 최종적으로 판단합니다.  
주어진 데이터셋이 경제기사 데이터이기 때문에 경제 금융 한국어 문서로 파인튜닝된 KB-ALBERT를 사전학습언어모델로 사용하였습니다.

### 성능
![그림1](https://user-images.githubusercontent.com/24906028/209667022-4161396f-cd03-48cc-a07f-d7a860121f87.jpg)
### 적용

![image](https://user-images.githubusercontent.com/24906028/209667039-a621185b-3902-44e4-a2d4-a095d13c8441.png)

기업여신상품설명서의 일부를 발췌하여 지문으로 넣고 질문을 입력했을 때 답을 잘 추론하는 것을 확인할 수 있었습니다.  

![image](https://user-images.githubusercontent.com/24906028/209667052-003e06b3-51c7-405f-9fa9-acfb909d04dd.png)

숫자를 추론해야하는 질문에도 답을 잘 추론합니다.  

![image](https://user-images.githubusercontent.com/24906028/209667060-2a7fd18b-a586-4d0e-82d2-9df62ccc3e87.png)

답을 알 수 없는 경우에는 답을 하지 않습니다.

### 라이브러리 수정사항

/usr/local/lib/python3.8/dist-packages/transformers/data/processors/squad.py

659행 `qas_id = qa["id"] → qas_id = qa["question_id"]`

/usr/local/lib/python3.8/dist-packages/transformers/data/metrics/squad_metrics.py

576~586행 `writer.write(json.dumps(all_predictions, indent=4) + "\n") → writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")`
