import nltk
from nltk.corpus import names #자연어 처리 모듈
nltk.download('names')

##1단계 첫 글자, 마지막 글자를 반환함
def gender_feature(word):
    bigram = [("".join(gram).lower(), "".join(gram).lower()) for gram in nltk.bigrams(word)] + \
             [('last_letter', word[-1]), ('first_letter', word[0])]
    return dict(bigram)

##2단계 학습 데이터 준비
# 머신러닝을 학습할 데이터를 준비. male.txt 파일과 female.txt 파일에서 머신러닝이 학습할 데이터를 가져옴
# 해당 파일은 <github.com/tomazas>에서 가져옴
if __name__ == "__main__":
    import random
    names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(names) #랜덤으로 섞음

##print(names)

###3단계 분류기 학습하기
#gender_features() 함수로 학습 데이터 처리하여 feature set를 만듬
# 그리고 feature set에서 학습 세트(train set)와 테스트 세트(test set)로 나눔
# 학습 세트는 나이브 베이즈 분류 머신러닝을 학습시키는 데 사용됨
# 그리고 테스트 세트는 학습된 머신러닝을 검증하는 데 사용함
featuresets = [(gender_feature(n), g) for (n, g) in names]
train_set, test_set = featuresets[900:], featuresets[:900]

model = nltk.classify.NaiveBayesClassifier.train(train_set)

##4단계 테스트 하기 약 80%~83%
#학습 데이터에 없는 이름을 가지고 테스트. Neo는 남성, Trinity와 Jane은 여성이라는 결과가 나옴
print(model.classify(gender_feature('Neo')))
print(model.classify(gender_feature('Trinity')))
print(model.classify(gender_feature('jane')))

# model = nltk.classify.DecisionTreeClassifier.train(train_set)

#테스트 세트를 이용하여 나이브 베이즈 분류기의 정확도를 확인
print(nltk.classify.accuracy(model, test_set))

#마지막으로 show_most_informative_features() 함수를 사용하면 이름의 성별을 구별하는 기준을 확인할 수 있음
print(model.show_most_informative_features())