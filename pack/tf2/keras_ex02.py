# 복수 레이어 : Deep Learning
# XOR을 하려면 두 개 이상의 레이어가 필요하다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # AVX를 지원하지 않는다는 에러 발생 시 기술함

# Keras 기본 개념 및 모델링 순서
# 참고 - http://cafe.daum.net/flowlife/S2Ul/10

# 1) 논리회로(XOR gate) 모델 작성
# 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0]) # [[0],[1],[1],[0]] 이차원으로 만들어도 된다.
print(x)
print(y)

# 2) 모델 생성(설정)

    # 모델 생성 방법 1
# model = Sequential([
#     Dense(input_dim = 2, units=5),    # 입력 2개, units(뉴런) 출력5개 
#     Activation('relu'),   # 렐루 : 0보다 큰건 전부 1, 나머진 전부 0
#     Dense(units=1),    # 출력 1개. 두번 째 모델에서 입력을 적을 필요 없다(1차 출력때 5개이기 때문). 
#     Activation('sigmoid')   # 시그모이드
# ])

    # 모델 생성 방법 2
model = Sequential()
# model.add(Dense(5, input_dim=2))
# model.add(Activation('relu'))    # 입력은 x, y 두 개가 들어와서 출력으로 relu 5개가 나간다.
model.add(Dense(5, input_dim=2, activation='relu')) # 위 두줄을  한줄로 입력 가능. 
model.add(Dense(5, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))   # 입력 5개, 출력 1개로 시그모이드가 나간다.

# 3) 학습 process 생성(컴파일)
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])  # Adam클래스를 사용. <- Adam이 가장 권장됨.


# 3-1) 모델 파라미터 확인
print(model.summary())  # 상세 정보를 보여줌


# 4) 모델 학습 (fitting)
history = model.fit(x, y, epochs=100, batch_size=1, verbose=2)  # epochs : 훈련 횟수, verbose : 1.진행과정 보기 0.진행과정 보지않고 빠르게 결과가 나옴. 
    # batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정 - 배치 크기는 학습에 큰 영향을 미침

    
# 5) 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_matrics = model.evaluate(x, y)
print(loss_matrics) # 분류 정확도


# 6) 학습결과 예측값 출력
pred2 = (model.predict(x) > 0.5).astype('int32')
print('pred2 : ', pred2)


print('\n------------------------------------------------------')
print(model.weights)    # dense/kernel, bias 확인
print('\n******************************************************')
#print(history.history)  # loss, acc를 확인 가능
print('history loss : ', history.history['loss'])
print('history acc : ', history.history['acc'])


# 시각화 방법1
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])   # loss(실수)는 점점 떨어진다.
plt.plot(history.history['acc'])    # acc(정확도)는 올라간다.
plt.xlabel('epoch')
plt.show()

# 시각화 방법2
import pandas as pd
# print(pd.DataFrame(history.history))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()





