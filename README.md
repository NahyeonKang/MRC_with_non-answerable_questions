# MRC_with_non-answerable_questions

## ๐ก ๊ฒฝ์ ๊ธฐ์ฌ QA ๋ฐ์ดํฐ์์ ํ์ฉํ์ฌ ์ง๋ฌธ ๋ด์ ๋ต์ด ์กด์ฌํ๋ค๋ฉด ๋ต์ ์์น๋ฅผ ์ถ๋ก ํ๊ณ , ์ง๋ฌธ ๋ด์ ๋ต์ด ์กด์ฌํ์ง ์๋๋ค๋ฉด ์๋ฌด ๋ต์ ํ์ง ์๋ ๋ชจ๋ธ

๋ชจ๋ธ์ ์ต๋ ํ ํฐ ์๋ 512์์ด๊ณ  QA ๋ฐ์ดํฐ๋ Paragraph์ Question์ด ํ๋ฒ์ ๋ค์ด๊ฐ๊ธฐ ๋๋ฌธ์ ๋ชจ๋ธ์ด ์ฝ์ ์ ์๋ ์ง๋ฌธ์ด ์งง์ต๋๋ค.  
์ง๋ฌธ์ ๋จ์ํ ์๋ฅด๊ฒ ๋๋ฉด ๋ต์ด ์๋ ์ง๋ฌธ์ด๋ผ๋ ๋ต์ด ์๋ฆฐ ์ง๋ฌธ ๋ด์ ํฌํจ๋์ง ์์ ์ ์๊ณ  ์๋ฆฐ ์ง๋ฌธ ์ฌ์ด์ ์กด์ฌํ๊ฒ ๋  ์๋ ์์ด ํ ๋ฌธ์ฅ ์ ๋๊ฐ ๊ฒน์น๊ฒ 350์ ๋จ์๋ก ์๋ผ์ ์ฌ์ฉํ์์ต๋๋ค.

### Retrospective Reader for Machine Reading Comprehension ์ฐธ๊ณ 

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

Retro Reader ๋ชจ๋ธ์ ๋๊ฐ์ ๋ชจ๋์ ๋ณ๋ ฌ์ ์ผ๋ก ์ฌ์ฉํฉ๋๋ค.  
Sketch Module์ Sequence Classification ํ์ํ์คํฌ๋ฅผ ์ฌ์ฉํด ์๋ต๊ฐ๋ฅ ์ฌ๋ถ๋ฅผ ํ๋จํ๊ณ   
Intensive Module์ QuestionAnswering ํ์ํ์คํฌ๋ฅผ ์ฌ์ฉํด ์๋ต๊ฐ๋ฅ ์ฌ๋ถ๋ฅผ ํ๋จํ๋ ๋์์ ์ง๋ฌธ์์์ ์ ๋ต์ ์์น๋ฅผ ์ถ๋ก ํฉ๋๋ค.  
๋ํ Sketchy Module์์์ ๋ก์ง๊ณผ Intensive Module์์ ์ ๋ต ์์ ์์น์ ์ข๋ฃ ์์น ํ๋ฅ ๊ฐ์ ์ ์๋ก ํ์ฉํด ์์๋ธํ์ฌ ์๋ต๊ฐ๋ฅ์ฌ๋ถ๋ฅผ ์ต์ข์ ์ผ๋ก ํ๋จํฉ๋๋ค.  
์ฃผ์ด์ง ๋ฐ์ดํฐ์์ด ๊ฒฝ์ ๊ธฐ์ฌ ๋ฐ์ดํฐ์ด๊ธฐ ๋๋ฌธ์ ๊ฒฝ์  ๊ธ์ต ํ๊ตญ์ด ๋ฌธ์๋ก ํ์ธํ๋๋ KB-ALBERT๋ฅผ ์ฌ์ ํ์ต์ธ์ด๋ชจ๋ธ๋ก ์ฌ์ฉํ์์ต๋๋ค.

### ์ฑ๋ฅ
![๊ทธ๋ฆผ1](https://user-images.githubusercontent.com/24906028/209667022-4161396f-cd03-48cc-a07f-d7a860121f87.jpg)
### ์ ์ฉ

![image](https://user-images.githubusercontent.com/24906028/209667039-a621185b-3902-44e4-a2d4-a095d13c8441.png)

๊ธฐ์์ฌ์ ์ํ์ค๋ช์์ ์ผ๋ถ๋ฅผ ๋ฐ์ทํ์ฌ ์ง๋ฌธ์ผ๋ก ๋ฃ๊ณ  ์ง๋ฌธ์ ์๋ ฅํ์ ๋ ๋ต์ ์ ์ถ๋ก ํ๋ ๊ฒ์ ํ์ธํ  ์ ์์์ต๋๋ค.  

![image](https://user-images.githubusercontent.com/24906028/209667052-003e06b3-51c7-405f-9fa9-acfb909d04dd.png)

์ซ์๋ฅผ ์ถ๋ก ํด์ผํ๋ ์ง๋ฌธ์๋ ๋ต์ ์ ์ถ๋ก ํฉ๋๋ค.  

![image](https://user-images.githubusercontent.com/24906028/209667060-2a7fd18b-a586-4d0e-82d2-9df62ccc3e87.png)

๋ต์ ์ ์ ์๋ ๊ฒฝ์ฐ์๋ ๋ต์ ํ์ง ์์ต๋๋ค.

### ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์์ ์ฌํญ

/usr/local/lib/python3.8/dist-packages/transformers/data/processors/squad.py

659ํ `qas_id = qa["id"] โ qas_id = qa["question_id"]`

/usr/local/lib/python3.8/dist-packages/transformers/data/metrics/squad_metrics.py

576~586ํ `writer.write(json.dumps(all_predictions, indent=4) + "\n") โ writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")`
