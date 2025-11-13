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

---

#### 표준편차 (Standard deviation)
- 분산의 제곱근이며, 관측된 변량의 흩어진 정도를 하나의 수치로 나타내는 통계량이다.
- 표준편차가 작을 수록 평균 값에서 변량들의 거리가 가깝다고 판단한다.
<img width="205" height="40" alt="standard_deviation" src="https://github.com/user-attachments/assets/a05d7743-b983-42a5-88a2-0e78fb04b4cf" />

```
import math

std = math.sqrt(variance)
print(f'표준편차: {round(std, 2)}')
```

---

#### 확률변수 (Random variable)
- 머신러닝, 딥러닝 등 확률을 다루는 분야에 있어서 필수적인 개념이다.
- 확률(probability)이 있다는 뜻은 사건(event)이 있다는 뜻이며,  
  시행(trial)을 해야 시행의 결과인 사건(event)이 나타난다.
- 시행(trial)을 해서 어떤 사건(event)이 나타났는지에 따라 값이 정해지는 변수이다.
- 알파벳 대문자로 표현하며, X, Y, Z 또는 X<sub>1</sub>, X<sub>2</sub>, X<sub>3</sub>과 같이 표현한다.
- 확률변수는 집합이며, 원소를 확률변수값(Value of random variable)이라고 표현한다.  
  확률변수에서 사용한 알파벳의 소문자를 사용한다.
- Y = { y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub> }, 이 때 Y는 확률변수이고 원소인 y<sub>1</sub> ~ y<sub>3</sub>은 확률변수값이다.

![random_variable](https://github.com/user-attachments/assets/76751f58-f994-41f2-bae3-095fce3a7858)

---

#### 범주형 확률변수 (Categorical random variable)
- 범주형 확률변수값은 수치가 아닌 기호나 언어, 숫자등으로 표현하고, 기호나 언어는 순서를 가질 수도 있다.
- 유한집합으로 표현한다. 유한집합은 원소의 수가 유한한 집합을 의미한다.
- {앞면, 뒷면}, {동의, 비동의}, {선택, 미선택}, {봄, 여름, 가을, 겨울}

---

#### 이산형 확률변수 (Discrete random variable)
- 이산형 확률변수값은 수치로 표현하고 셀 수 있는 값이다. 이를 더 넓은 범위로,  
  양적 확률변수 또는 수치형 확률변수라고도 부른다.
- 유한집합 또는 셀 수 있는 무한집합으로 표현한다. 무한집합은 원소의 수가 무한한 집합을 의미한다.
- {0, 1, 2, 3}, {10, 20, 30}, {1, 2, 3, ...}, {100, 1000, 10000}

---

#### 연속형 확률변수 (Continuous random variable)
- 연속형 확률변수는 구간을 나타내는 수치로 표현한다. 이를 더 넓은 범위로,  
  양적 확률변수 또는 수치형 확률변수라고도 부른다.
- 셀 수 없는 무한집합으로 표현한다.
- 128.56 < X < 268.56

---

#### 확률분포 (Probability distribution)
- 사건에 대한 확률변수에서 정의된 모든 확률값의 분포이며, 서로 다른 모든 결과의 출현 확률을 제공한다.
  
> <strong>1) 동전 던지기 (시행)</strong>  
> <strong>2) { 0, 1 } (확률변수와 확률변수값)</strong>  
> <strong>3) 완벽한 형태의 동전일 경우 확률 분포</strong>  
>
> <img width="259" height="186" alt="probability_distribution01" src="https://github.com/user-attachments/assets/83847105-9356-4e94-8f44-17f15b626cd3" />

  
> <strong>1) 1 ~ 12까지 새겨진 주사위 던지기 (시행)</strong>  
> <strong>2) { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 } (확률변수와 확률변수값)</strong>  
> <strong>3) 완벽한 형태의 주사위일 경우 확률 분포</strong>
> 
> <img width="474" height="190" alt="probability_distribution02" src="https://github.com/user-attachments/assets/0208e721-78c6-4922-98ce-1e98f86c4b92" />

---

#### 확률분포표 (Probability distribution table)
- 확률변수의 모든 값(원소)에 대해 확률을 표로 표시한 것이다.
- 범주형 또는 이산형 확률변수의 확률분포를 표현하기에 적합한 방식이다.

```
import numpy as np
import pandas as pd

h_dist_df = pd.DataFrame(np.arange(0, 100) % 4 + 1, columns=['경주마 번호'])
h_dist_group_df = h_dist_df.groupby('경주마 번호')['경주마 번호'].count().reset_index(name='1등 횟수')

h_dist_group_df['1등할 확률'] = h_dist_group_df['1등 횟수'] / 100
display(h_dist_group_df)
```

---

#### 확률분포함수 (Probability distribution function)
- 확률변수의 분포를 나타내는 함수로서, 확률변수의 확률변수값이 나올 확률을 나타내는 함수이다.
- 확률질량함수, 확률밀도함수 등의 함수가 있다.

```
import matplotlib.pyplot as plt
import numpy as np

plt.hist(h_dist_df['경주마 번호'], bins=range(1, 6), \
            density=True, color='pink', edgecolor='red', linewidth=2)
```

---
#### 확률질량 함수 (Probability mass function, pmf)
- 확률변수 X의 분포를 나타내는 함수로서, x<sub>i</sub>가 나올 확률이다.
- 확률변수의 값을 매개변수로 전달받고, 해당 값이 나타날 확률을 구해서 리턴하는 함수이다.
- 범주형 확률변수와 이산형 확률변수에서 사용된다.
- 확률변수에서 각 값에 대한 확률을 나타내는 것이 마치 각 값이 "질량"을 가지고 있는 것처럼 보이기 때문에 확률질량 함수로 불린다.

> 확률질량 함수 f는 확률변수 X가 x를 변수값으로 가질 때의 확률이다.  
> <img width="157" height="33" alt="pmf01" src="https://github.com/user-attachments/assets/7b7c5284-6f2c-4331-b864-776b023cd1ee" />
> <img width="113" height="53" alt="pmf02" src="https://github.com/user-attachments/assets/7ed014b4-d048-478d-8023-351aee4e29fb" />
> <img width="122" height="25" alt="pmf03" src="https://github.com/user-attachments/assets/a4580dd7-2813-4901-853f-7a6cae59fb2f" />

```
import numpy as np
import pandas as pd

h_df = pd.DataFrame(np.random.randint(1, 5, size=(100, 1)), columns=['경주마 번호'])
h_group_df = h_df.groupby('경주마 번호')['경주마 번호'].count().reset_index(name='1등 횟수')

h_group_df['1등할 확률'] = h_group_df['1등 횟수'] / 100
display(h_group_df)
```
```
import matplotlib.pyplot as plt
import numpy as np

plt.hist(h_df['경주마 번호'], range(1, 6), \
            density=False, color='pink', edgecolor='red', linewidth=2)

plt.show()
```

---
#### 무한대 (Infinity)
- 끝없이 커지는 상태를 의미하고 기호로 ∞를 사용한다.

---
#### 무한소 0 (Infinitesimal)
- 거의 없다는 의미이고, 0에 매우 근접하지만 0이 아닌 상태를 의미한다.

---
#### 미분 (Differential)
- 기울기는 독립변수가 종속변수에 미치는 영향력의 크기를 의미한다.
- 변경 전의 독립변수 x<sub>1</sub>이라는 점과 변경 후의 x<sub>2</sub>라는 점을 지나는 직선의 기울기가 바로 변화에 대한 속도이다.
- 즉, 직선의 기울기가 4로 구해졌다면,  
  종속변수가 독립변수의 변화에 4배 속도로 변화된 것이다.
- 이 때, 두 점 사이가 무한히 가까워지면,  
  결국 거의 한 점과 같은 점에 대한 접선의 기울기가 되고 이는 순간적인 변화량이다.
- 미분을 통해서 독립변수가 미세하게 변화할 때 순간적으로 종속변수가 얼마나 빠르게 변화하는 지를 알 수 있다.

---
#### 적분 (Integral)
- 선분 = 높이(길이), 면적 = 가로 X 높이
- 면적을 구할 때 여러 사각형으로 나눈 뒤 합하여도 전체 면적이 나온다.
- 가로가 무한소 0인 사각형 즉, 선분과 거의 비슷한 사각형을 쌓은 뒤, 각 면적을 모두 합하는 것이 적분이다.

---
#### 확률밀도 함수 (Probability density function, pdf)
- 확률변수 X의 분포를 나타내는 함수로서, 특정 구간에 속할 확률이고 이는 특정 구간을 적분한 값이다.
- 확률변수값의 범위(구간)를 매개변수로 전달받고, 범위의 넓이를 구해서 리턴하는 함수이다.
- 연속형 확률변수에서 사용된다.
- 전체에 대한 확률이 아닌 구간에 포함될 확률을 나타내기 때문에 구간에 따른 밀도를 구하는 것이고,  
  이를 통해 확률밀도 함수라 불린다.

- ※ CDF(cumulative distribution function): 이하 확률

> 확률밀도 함수 f는 특정 구간에 포함될 확률을 나타낸다.  
> <img width="243" height="56" alt="pdf01" src="https://github.com/user-attachments/assets/6703a15a-edcd-48c4-be52-27b728b6d4ba" />
> <img width="122" height="25" alt="pdf02" src="https://github.com/user-attachments/assets/6a40c58b-58bb-4fee-aa5f-bff052aa398d" />
> <img width="142" height="59" alt="pdf03" src="https://github.com/user-attachments/assets/cc8fbc5e-5526-47a3-a478-f79c3ff8e128" />
> <img width="127" height="32" alt="pdf04" src="https://github.com/user-attachments/assets/0505e4e4-6e2e-4660-aad8-c9069b6c7343" />


```
!pip install scipy
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def pdf(x):
    mu = 0
    sigma = 1
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) / sigma) ** 2 / 2)

a = -1
b = 2

def calculate_probability_in_range(a, b):
    probability, _ = quad(pdf, a, b)
    return probability

probability = calculate_probability_in_range(a, b)
print(f'구간 [{a}, {b}]에서의 확률: {probability}')

x_values = np.linspace(-5, 5, 10000)
y_values = pdf(x_values)

plt.plot(x_values, y_values, label='pdf', color='blue')
plt.fill_between(x_values, y_values, \
                 where=(x_values >= a) & (x_values <= b), \
                 color='skyblue', alpha=0.5, label='area')
plt.axvline(x=a, linestyle='--', color='red', label='start')
plt.axvline(x=b, linestyle='--', color='red', label='end')
plt.xlabel('x')
plt.ylabel('density')
plt.title('Probability density function')
plt.grid(True)
plt.show()
```
---

#### 정규분포 (Normal distribution)
- 모든 독립적인 확률변수들의 평균은 어떠한 분포에 가까워지는데, 이 분포를 정규분포라고 한다.
- 즉, 비정규분포의 대부분은 극한상태에 있어서 정규분포에 가까워진다.

```
<img width="1920" height="1152" alt="normal_distribution01" src="https://github.com/user-attachments/assets/b8bda191-b147-4838-ba67-b6c8e2b4212d" /> <img width="1920" height="1152" alt="normal_distribution02" src="https://github.com/user-attachments/assets/7418cce4-c571-4792-8ace-9ec8dbc01e1f" />

- 평균 μ(mu)와 표준편차 σ(sigma)에 대해 아래의 확률밀도함수를 가지는 분포를 의미한다.

<div style="display: flex">
    <div>
        <img width="245" height="80" alt="normal_distribution03" src="https://github.com/user-attachments/assets/a435f408-ad4f-4b1c-899b-46e9602d70ba" />
    </div>
    <div>
        ![normal_distribution04](https://github.com/user-attachments/assets/fe91b493-0264-45e5-a05b-b0de0e918b21)
    </div>
</div>
```

---

#### 표준 정규분포 (Standard normal distribution)
- 정규분포는 평균과 표준편차에 따라서 모양이 달라진다.

<img width="545" height="339" alt="standard_normal_distribution01" src="https://github.com/user-attachments/assets/1627f6bf-776b-420d-8d87-ba132f9e0757" />

- 정규분포를 따르는 분포는 많지만 각 평균과 표준편차가 달라서 일반화할 수 없다.
- N(μ, σ) = N(0, 1)로 만든다면 모두 같은 특성을 가지는 동일한 확률분포로 바꿔서 일반화할 수 있다.
- 따라서 일반 정규분포를 표준 정규분포로 바꾼 뒤 표준 정규분포의 특정 구간의 넓이를 이용해서 원래 분포의 확률을 구할 수 있다.

<img width="501" height="376" alt="standard_normal_distribution02" src="https://github.com/user-attachments/assets/2aaf703a-9f34-4014-a885-362be79991fc" />

---

#### 표준화 (Standardization)
- 다양한 형태의 정규분포를 표준 정규분포로 변환하는 방법이다.
- 표준 정규분포에 대한 값(넓이)를 이용해 원래 분포의 확률을 구할 수 있다.

<img width="136" height="60" alt="standardization01" src="https://github.com/user-attachments/assets/1c05802f-9365-48d2-806d-7477ff10dc7f" />

<img width="679" height="644" alt="standardization02" src="https://github.com/user-attachments/assets/5619fe7e-2567-4f6a-b662-e43aa63f9a6d" />

---

#### 모집단과 모수 (Population and population parameter)
- 모집단이란, 정보를 얻고자 하는 대상의 전체 집합을 의미한다.
- 모수란, 모집단의 수치적 요약값을 의미한다. 평균 또는 표준편차와 같은 모집단의 통계값을 모수라고 한다.

---

#### 표본과 샘플링 (Sample and Sampling)
- 표본이란, 모집단의 부분집합으로서 표본의 통계량을 통해 모집단의 통계량을 추론할 수 있다.
- 모집단의 통계량을 구할 수 없는 상황 즉, 전수 조사가 불가능한 상황에서 임의의 표본을 추출하여 분석한다.
- 이렇게 표본(sample)을 추출하는 작업을 샘플링(sampling)이라고 한다.
