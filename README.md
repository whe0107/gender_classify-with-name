{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 나이즈 베이즈 분류로 이름으로 성별 분류하기\n",
    "\n",
    "> 나이브 베이즈 분류(Naïve Bayes Classification)는 텍스트 분류에 사용된다.</br>\n",
    "> 대표적으로 스팸 메일을 필터링하는 데 사용되고 있다.\n",
    "\n",
    "\n",
    "## 이름 성별 분류하기\n",
    "\n",
    "> 남성과 여성의 이름에는 몇 가지 독특한 특성이 있다.</br>\n",
    "> 이름으로 남성인지, 여성인지 분류해보자."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0. 필요 모듈 설치"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package names to\n",
      "[nltk_data]     C:\\Users\\whe01\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package names is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk  #자연어 처리 모듈\n",
    "from nltk.corpus import names\n",
    "nltk.download('names')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. 첫 글자, 마지막 글자를 반환함"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gender_feature(word):\n",
    "    bigram = [(\"\".join(gram).lower(), \"\".join(gram).lower()) for gram in nltk.bigrams(word)] + \\\n",
    "             [('last_letter', word[-1]), ('first_letter', word[0])]\n",
    "    return dict(bigram)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. 학습 데이터 준비\n",
    "\n",
    "> * 머신러닝을 학습할 데이터를 준비. male.txt 파일과 female.txt 파일에서 머신러닝이 학습할 데이터를 가져옴 </br>\n",
    "> * 해당 파일은 <github.com/tomazas>에서 가져옴 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    import random\n",
    "    names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])\n",
    "    random.shuffle(names) #랜덤으로 섞음"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. 분류기 학습하기\n",
    "\n",
    "> * gender_features() 함수로 학습 데이터 처리하여 feature set를 만듬 </br>\n",
    "> * feature set에서 학습 세트(train set)와 테스트 세트(test set)로 나눔 </br>\n",
    "> * 학습 세트는 나이브 베이즈 분류 머신러닝을 학습시키는 데 사용됨 </br>\n",
    "> * 테스트 세트는 학습된 머신러닝을 검증하는 데 사용함"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "featuresets = [(gender_feature(n), g) for (n, g) in names]\n",
    "train_set, test_set = featuresets[900:], featuresets[:900]\n",
    "\n",
    "model = nltk.classify.NaiveBayesClassifier.train(train_set)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. 테스트 하기\n",
    "\n",
    "> * 학습 데이터에 없는 이름을 가지고 테스트. Neo는 남성, Trinity와 Jane은 여성이라는 결과가 나옴"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "male\n",
      "female\n",
      "female\n"
     ]
    }
   ],
   "source": [
    "print(model.classify(gender_feature('Neo')))\n",
    "print(model.classify(gender_feature('Trinity')))\n",
    "print(model.classify(gender_feature('jane')))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> * 테스트 세트를 이용하여 나이브 베이즈 분류기의 정확도를 확인"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8077777777777778\n"
     ]
    }
   ],
   "source": [
    "print(nltk.classify.accuracy(model, test_set))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> * 마지막으로 show_most_informative_features() 함수를 사용하면 이름의 성별을 구별하는 기준을 확인할 수 있음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Most Informative Features\n",
      "             last_letter = 'k'              male : female =     41.1 : 1.0\n",
      "             last_letter = 'a'            female : male   =     35.9 : 1.0\n",
      "                      rv = 'rv'             male : female =     28.9 : 1.0\n",
      "                      lt = 'lt'             male : female =     22.1 : 1.0\n",
      "                      fo = 'fo'             male : female =     20.1 : 1.0\n",
      "                      hu = 'hu'             male : female =     19.4 : 1.0\n",
      "             last_letter = 'v'              male : female =     18.7 : 1.0\n",
      "                      sp = 'sp'             male : female =     17.6 : 1.0\n",
      "             last_letter = 'f'              male : female =     16.0 : 1.0\n",
      "                      rw = 'rw'             male : female =     15.3 : 1.0\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(model.show_most_informative_features())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
