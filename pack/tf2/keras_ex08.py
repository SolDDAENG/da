# 선형회귀
import tensorflow as tf
from tensorflow.keras.models import Sequential  # 완전연결 모델
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# 자료 읽기
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Advertising.csv')
del data['no']
print(data.head(2))
print(data.corr())

# 정규화 : 0 ~ 1 사이로 scaling - 정확도를 높이기 위한 작업(정규화를 하는 이유)
#scaler = MinMaxScaler(feature_range=(0, 1))    # 기본값이 0 ~ 1
# scaler = MinMaxScaler()
# xy = scaler.fit_transform(data) # scaler.inverse_transform(xy) 를 사용하면 원래값으로 환원환다.
# print(xy[:2])

xy = minmax_scale(data, axis=0, copy=True)  # 위랑 같지만 이게 더 편하다.
print(xy[:2])

# 데이터 분리 : train_test_split - 과적합 방지. 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(xy[:, 0:-1], xy[:,-1],  # x = xy의 마지막 열을 제외한 모든 열의 모든행, y = xy의 마지막 열의 모든 행
                                                test_size=0.3, random_state=123)  
print(xtrain[:2], ' ', xtrain.shape)    # (140, 3)
print(ytrain[:2], ' ', ytrain.shape)    # (140, )


# 모델 생성
model = Sequential()
model.add(Dense(20, input_dim = 3, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))    # 레이어 3개


# 모델 파라미터(구성정보) 확인
print(model.summary())


# 모델을 사진으로 저장
tf.keras.utils.plot_model(model, 'abc.png') # GraphBiz가 설치되어야 가능


# 학습 process 생성(컴파일)
model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mse'])


# 모델 학습 (fitting)
history = model.fit(xtrain, ytrain, batch_size=32, epochs=100, verbose=1, validation_split=0.2)
    # batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정. 속도가 빨라짐.    
    # validation_split=0.2 : 학습데이터가 들어오면 알아서 k-fold작업을 한다.(나눠서 들어온 데이터가 아니여도 내부에서 나눈다.) train데이터를 다시 8:2로 나눠서 8을 학습데이터, 2를 검정데이터로 사용한다.
print('train loss : ', history.history['loss'])

    
# 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss = model.evaluate(xtest, ytest, batch_size=32)
print('test loss : ', loss)

# 설명력(결정계수)
from sklearn.metrics import r2_score
print('r2_score(설명력) : ', r2_score(ytest, model.predict(xtest)))    # r2_score(실제값, 예측값) : 0.916

# 학습결과 예측값 출력
pred = model.predict(xtest)
print('실제값 : ', ytest[:5])
print('예측값 : ', pred[:5].flatten())


# 시각화














