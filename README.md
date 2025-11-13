### Data Analysis
- 원시(원본 그대로) 데이터를 분석하여 인사이트(가시성 증가 및 깊은 이해)로 변환하는 작업이다.
- 문제를 해결하기 위해 데이터를 사용해서 흐름 및 방향을 찾는 기술이다.
- 데이터 분석을 통해 비지니스 프로세스를 구성하고, 의사 결정을 개선하며, 비지니스 성장을 증진할 수 있다.

<img width="1430" height="570" alt="architecture" src="https://github.com/user-attachments/assets/a5f7c5d3-aedb-43de-83ff-c5f139fd1ced" />
** <sub>ETL은 Extract(추출), Transform(변환), Load(적재)를 의미한다. 여기 저기 흩어진 데이터를 하나로 모으기 위한 결합 과정이다.</sub>

---

### 기초 통계 (Basic statistics)
📌 통계는 아직 발생하지 않은 일을 예측하기 위해 사용한다.
- 통계학을 공부하는 데 있어 필요한 기본 개념이고,  
  수량적인 비교를 기초로 많은 사실을 관찰하고 처리하는 방법을 연구하는 학문이다.
- 불균형 데이터를 대상으로 규칙성과 불규칙성을 발견한 뒤 실생활에 적용할 수 있다.
- 
<img width="1690" height="999" alt="statistics01" src="https://github.com/user-attachments/assets/feac95a3-7bcc-4f27-b57d-7378ef58d09e" />

---

#### 변량 (Variable)
- 자료의 수치를 변량이라고 하며, 이는 데이터의 값을 의미한다.
```
# !pip install numpy pandas
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(151, 190, size=(10, 10)), \
        columns="서울,경기,인천,광주,대구,부산,전주,강릉,울산,수원".split(","))

display(df)
```

---

#### 계급 (Class)
- 변량을 일정 간격으로 나눈 구간을 의미한다.
- 변량의 최소값과 최대값을 잘 고려해서 계급을 정해야한다.
- 예를 들어, 계급이 (150, 160]일 경우, 151 ~ 160이 계급에 속한다. 즉 소괄호는 구간 미포함, 대괄호는 구간 포함이다.

```
df_seoul = df['서울']
df_class = pd.cut(df_seoul, bins=[150, 160, 170, 180, 190])
df_seoul_class = pd.DataFrame({"서울": df_seoul, "계급": df_class})

display(df_seoul_class)
```

---

#### 도수 (Frequency)
- 각 계급에 속하는 변량의 개수를 의미한다.
```
# observed: 결과가 0개일 때 표시 여부, False: 표시, True: 미표시
df_seoul_class.groupby('계급', observed=False).count()
```

---

#### 상대 도수 (Relative frequency)
- 각 계급에 속하는 변량의 비율을 의미한다.
```
df_seoul_class.groupby('계급', observed=False).count().apply(lambda x: x / 10)
```

---

#### 도수분포표(Frequency table)
- 주어진 자료를 계급별로 나눈 뒤 각 계급에 속하는 도수 및 상대 도수를 조사한 표이다.
- 구간별 분포를 한 번에 알아보기 좋지만 계급별 각 변량의 정확한 값이 생략되어 있다.
```
freq = df_seoul_class.groupby('계급', observed=False).count()['서울']
r_freq = df_seoul_class.groupby('계급', observed=False).count().apply(lambda x: x / 10)['서울']

freq_df = pd.DataFrame({'도수': freq, '상대도수': r_freq})
display(freq_df)
```
```
freq_df.reset_index(drop=False, inplace=True)

display(freq_df)
```

---

#### 히스토그램 (Histogram)
- 도수분포표를 시각화한 그래프이다.
```
# !pip install matplotlib
df_seoul_class['서울']
```
```
import matplotlib.pyplot as plt

df_seoul_class['서울'].hist(bins=4)
```

---

#### 산술 평균 (Mean)
- 변량의 합을 변량의 수로 나눈 값을 의미한다.

<img width="113" height="53" alt="pmf02" src="https://github.com/user-attachments/assets/b61c40ec-0846-4a26-97b7-285d0083f08e" />
```
df.mean(axis=0).to_frame(name="평균 키")
```

---
#### 편차 (Deviation)
- 변량에서 평균을 뺀 값이다.
- 각 변량의 편차를 구한 뒤 모두 합하면 0이 되기 때문에 편차의 평균은 구할 수 없다.
```
g_df = df['경기'].to_frame()
g_df['편차'] = g_df['경기'].apply(lambda x: x - g_df.mean())

display(g_df)

print(round(g_df['편차'].sum()))
```

---

#### 분산 (Variance)
- 변량이 평균으로부터 떨어져있는 정도를 보기 위한 통계량이다.
- 편차에 제곱하여 그 합을 구한 뒤 산술 평균을 낸다.
<img width="305" height="64" alt="variance" src="https://github.com/user-attachments/assets/88fbab0e-494d-410a-ad43-51994d9a93f9" />

```
g_df['편차의 제곱'] = g_df['편차'].apply(lambda x: x**2)
display(g_df)

variance = g_df['편차의 제곱'].mean()
print(f'분산: {round(variance, 2)}')
```

