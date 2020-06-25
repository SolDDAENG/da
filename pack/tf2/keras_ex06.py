# 단순선형회귀 모델 : 작성방법 3가지
# 공부시간에 따른 성적 점수 예측
# http://cafe.daum.net/flowlife/S2Ul/22

import tensorflow as tf
from tensorflow.keras.models import Sequential  # 완전연결 모델
from tensorflow.keras.layers import Dense 
from tensorflow.keras import optimizers
import numpy as np

x_data = np.array([1,2,3,4,5], dtype=np.float32)
y_data = np.array([11,32,53,64,70], dtype=np.float32)
print(np.corrcoef(x_data, y_data))  # 상관관계 0.97    상관관계가 0.3 이하면 하지않는 것이 좋다.

# 모델 작성 1 : 완전연결 모델
print('\n모델 작성 1------------------------------------------------------------------------------------')
model = Sequential()    # 대칭구조
model.add(Dense(1, input_dim=1, activation='linear'))    # 입력 1, 출력 1, 출력 모형은 Linear(단순선형회귀)
model.add(Dense(1, activation='linear'))    # 레이어 추가

# 학습 process 생성(컴파일)
opti = optimizers.SGD(lr=0.001) # 경사하강법으로 갈거기 때문에 SGD를 사용한다.    lr=(학습률)
model.compile(opti, loss='mse', metrics='mse')  # mse(mean squared error)(평균제곱오차) : 잔차의 제곱에 평균을 취함. 수치가 작을수록 정확성이 높다.

# 모델 학습 (fitting)
model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=1)  # epochs : 학습횟수

# 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_metrics = model.evaluate(x_data, y_data)
print('loss_metrics : ', loss_metrics)

# 학습결과 예측값 출력
from sklearn.metrics import r2_score    # 결정계수(설명력)
print('설명력 : ', r2_score(y_data, model.predict(x_data)))    # 설명력 :  0.94
print('실제값 : ', y_data) # 실제값 :  [11. 32. 53. 64. 70.]
print('예측값 : ', model.predict(x_data).flatten())    # flatten() : 차원축소. 2차원 -> 1차원 
print('새로운 값 예측 : ', model.predict([6.5, 2.1]).flatten())

# 시각화
# import matplotlib.pyplot as plt
# plt.plot(x_data, model.predict(x_data), 'b', x_data, y_data, 'ko')
# plt.show()


# ==================================================================================================================================
# 모델 작성 2 : function api를 사용 - 방법 1에 비해 유연한 모델을 작성
print('\n모델 작성 2------------------------------------------------------------------------------------')
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

inputs = Input(shape=(1,))

#outputs = Dense(1, activation='linear')(inputs) # 히든 레이어 1개
output1 = Dense(2, activation='linear')(inputs)
outputs = Dense(1, activation='linear')(output1) # 히든 레이어 2개

model2 = Model(inputs, outputs)

# 학습 process 생성(컴파일)    - 이하 부분은 위 모델1과 같다.
opti = optimizers.SGD(lr=0.001) # 경사하강법으로 갈거기 때문에 SGD를 사용한다.    lr=(학습률)
model2.compile(opti, loss='mse', metrics='mse')  # mse(mean squared error)(평균제곱오차) : 잔차의 제곱에 평균을 취함. 수치가 작을수록 정확성이 높다.

# 모델 학습 (fitting)
model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=1)  # epochs : 학습횟수

# 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_metrics = model2.evaluate(x_data, y_data)
print('loss_metrics : ', loss_metrics)

# 학습결과 예측값 출력
print('실제값 : ', y_data) # 실제값 :  [11. 32. 53. 64. 70.]
print('예측값 : ', model2.predict(x_data).flatten())    # flatten() : 차원축소. 2차원 -> 1차원 
# ==================================================================================================================================


# ==================================================================================================================================
# 모델 작성 3-1 : sub classing 사용 : Model을 상속
print('\n모델 작성 3-1----------------------------------------------------------------------------------')

# Model class 생성
class MyModel(Model):
    def __init__(self): # 생성자
        super(MyModel, self).__init__()
        self.d1 = Dense(5, activation='linear')
        self.d2 = Dense(1, activation='linear') # 레이어 2개
        
    def call(self, x):  # x : 입력 매개변수    <== 모델.fit(), 모델.evaluate(), 모델.predict()를 사용할 때 호출한다.
        x = self.d1(x)
        return self.d2(x)

model3 = MyModel()  # 생성자 호출
    
# 학습 process 생성(컴파일)    - 이하 부분은 위 모델1과 같다.
opti = optimizers.SGD(lr=0.001) # 경사하강법으로 갈거기 때문에 SGD를 사용한다.    lr=(학습률)
model3.compile(opti, loss='mse', metrics='mse')  # mse(mean squared error)(평균제곱오차) : 잔차의 제곱에 평균을 취함. 수치가 작을수록 정확성이 높다.

# 모델 학습 (fitting)
model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=1)  # epochs : 학습횟수

# 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_metrics = model3.evaluate(x_data, y_data)
print('loss_metrics : ', loss_metrics)

# 학습결과 예측값 출력
print('실제값 : ', y_data) # 실제값 :  [11. 32. 53. 64. 70.]
print('예측값 : ', model3.predict(x_data).flatten())    # flatten() : 차원축소. 2차원 -> 1차원 

# ==================================================================================================================================

# ==================================================================================================================================
# 모델 작성 3-2 : sub classing 사용 : Layer를 상속
print('\n모델 작성 3-2----------------------------------------------------------------------------------')
from tensorflow.keras.layers import Layer

# Linear class 생성
class Linear(Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape):   # call함수 호출
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer = 'random_normal',
                                 trainable = True)  # trainable = True Back propergation 진행

        self.b = self.add_weight(shape=(self.units),    # b = bias
                                 initializer = 'zeros',
                                 trainable = True)  # trainable = True Back propergation 진행
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
        
class MyMLP(Model):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.linear1 = Linear(1)    # 레이어 1개
        # self.linear1 = Linear(2)    # 레이어 2개
        # self.linear2 = Linear(1)
        
    def call(self, inputs): # Linear의 build 호출 
        return self.linear1(inputs)    # 레이어 1개의 경우
        # x = self.linear1(inputs)   # 레이어 2개의 경우
        # return self.linear1(x) 
        
model4 = MyMLP()

# 학습 process 생성(컴파일)    - 이하 부분은 위 모델1과 같다.
opti = optimizers.SGD(lr=0.001) # 경사하강법으로 갈거기 때문에 SGD를 사용한다.    lr=(학습률)
model4.compile(opti, loss='mse', metrics='mse')  # mse(mean squared error)(평균제곱오차) : 잔차의 제곱에 평균을 취함. 수치가 작을수록 정확성이 높다.

# 모델 학습 (fitting)
model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=1)  # MyMLP의 call 호출

# 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_metrics = model4.evaluate(x_data, y_data)
print('loss_metrics : ', loss_metrics)

# 학습결과 예측값 출력
print('실제값 : ', y_data) # 실제값 :  [11. 32. 53. 64. 70.]
print('예측값 : ', model4.predict(x_data).flatten())    # flatten() : 차원축소. 2차원 -> 1차원 

# ==================================================================================================================================






