# 와인의 맛, 등급, 산도 등을 측정해 레트와 화이트 와인 분류 모델 장성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# seed 값 고정(난수 발생)
np.random.seed(3)
tf.random.set_seed(3)

wdf = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/wine.csv', header=None)
df = wdf.sample(frac=0.5)  # frac : 전체 개수의 비율만큼 샘플을 반환. n과 중복사용 불가
print(df.head(2))
print(df.info())
print(df.iloc[:, 12].unique())  # [0 1] => binary crossentropy : binary 이항분류
dataset = df.values
x = dataset[:, 0:12]  # feature
y = dataset[:, -1]  # label(class)

# 모델
model = Sequential()
# model.add(Dense(30, input_dim=12, activation='relu'))  # ReLU는 정류 선형 유닛이다. 입력값이 0보다 작으면 0으로 출력, 0보다 크면 입력값 그대로 출력하는 유닛
# model.add(Dense(12, activation='relu'))
# model.add(Dense(8, activation='relu'))

model.add(Dense(30, input_dim=12, activation='elu'))  # ReLU는 정류 선형 유닛이다. 입력값이 0보다 작으면 0으로 출력, 0보다 크면 입력값 그대로 출력하는 유닛
# model.add(tf.keras.layers.BatchNormalization())     # 배치 정규화 - 그레디언트 소실과 폭주문제 해결
model.add(Dense(12, activation='elu'))
# model.add(tf.keras.layers.BatchNormalization())     # 너무 급격하게 변화할 때 적합. 현재 여기에는 적합하지 않다.
model.add(Dense(8, activation='elu'))
# model.add(tf.keras.layers.BatchNormalization()) 
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 데이터를 두 개의 그룹으로 분류하는 문제에서 가장 기본적인 방법

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit() 이전의 훈련되지 않은 모델 정확도
lass, acc = model.evaluate(x, y, verbose=0)
print(' 훈련되지 않은 모델 정확도 : {:5.2f}%'.format(acc * 100))  # {:5.2f} => 총 자릿수 5, 소수점 2자리까지

# 모델 저장 폴더 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):  # 내 디렉토리에 model이라는 폴더가 없으면 만든다.
    os.mkdir(MODEL_DIR)
    
modelpath = 'model/{epoch:02d}-{val_loss:4f}.hdf5'

# 모델 학습 시 모니터링의 결과를 파일로 저장
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=5)  # val_loss가 없을땐 loss를 써도 된다.

# 모델 실행 - fitting
history = model.fit(x, y, validation_split=0.3, epochs=100000, batch_size=128,
                    callbacks=[early_stop, chkpoint])

lass, acc = model.evaluate(x, y, verbose=0)
print(' 훈련된 모델 정확도 : {:5.2f}%'.format(acc * 100))

# 시각화
y_vloss = history.history['val_loss']  # val_loss가 없을땐 loss를 써도 된다.
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c='red', ms=3)  # 오차
plt.plot(x_len, y_acc, 'x', c='blue', markersize=3)  # 정확도
plt.show()
