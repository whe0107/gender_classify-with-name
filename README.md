## 나이즈 베이즈 분류로 이름으로 성별 분류하기

> 나이브 베이즈 분류(Naïve Bayes Classification)는 텍스트 분류에 사용된다.<br/>
> 대표적으로 스팸 메일을 필터링하는 데 사용되고 있다.


## 이름 성별 분류하기

> 남성과 여성의 이름에는 몇 가지 독특한 특성이 있다..<br/>
> 이름으로 남성인지, 여성인지 분류해보자.

### 0. 필요 모듈 설치

```python
import nltk  #자연어 처리 모듈
from nltk.corpus import names
nltk.download('names')
 ```
 ![zero](/images/zero.png)

 ### 1. 첫 글자, 마지막 글자를 반환함
 
```python
 def gender_feature(word):
    bigram = [("".join(gram).lower(), "".join(gram).lower()) for gram in nltk.bigrams(word)] + \
             [('last_letter', word[-1]), ('first_letter', word[0])]
    return dict(bigram)
```

### 2. 학습 데이터 준비

> * 머신러닝을 학습할 데이터를 준비. male.txt 파일과 female.txt 파일에서 머신러닝이 학습할 데이터를 가져옴
> * 해당 파일은 <github.com/tomazas>에서 가져옴 

```python
if __name__ == "__main__":
    import random
    names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(names) #랜덤으로 섞음
```

### 3. 분류기 학습하기

> * gender_features() 함수로 학습 데이터 처리하여 feature set를 만듬 
> * feature set에서 학습 세트(train set)와 테스트 세트(test set)로 나눔
> * 학습 세트는 나이브 베이즈 분류 머신러닝을 학습시키는 데 사용됨
> * 테스트 세트는 학습된 머신러닝을 검증하는 데 사용함
> * 의사결정트리도 확인해봄

```python
featuresets = [(gender_feature(n), g) for (n, g) in names]
train_set, test_set = featuresets[900:], featuresets[:900]

model = nltk.classify.NaiveBayesClassifier.train(train_set) #나이즈베이즈
model1 = nltk.classify.DecisionTreeClassifier.train(train_set) # 의사결정트리
```

### 4. 테스트 하기

> * 학습 데이터에 없는 이름을 가지고 테스트. Neo는 남성, Trinity와 Jane은 여성이라는 결과가 나온다.

```python
print(model.classify(gender_feature('Neo')))
print(model.classify(gender_feature('Trinity')))
print(model.classify(gender_feature('jane')))
```

 ![one](/images/one.png)

> * 테스트 세트를 이용하여 나이브 베이즈 분류기의 정확도를 확인(약 80프로의 정확성을 보인다.)

```python
print(nltk.classify.accuracy(model, test_set))
print(nltk.classify.accuracy(model1, test_set))
```

 ![two](/images/two.png)

> * 마지막으로 show_most_informative_features() 함수를 사용하면 이름의 성별을 구별하는 기준을 확인할 수 있음
```python
print(model.show_most_informative_features())
```

 ![three](/images/three.png)
