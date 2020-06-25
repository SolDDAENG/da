# 다중 선형회귀, 텐서보드(모델의 구조 밒 학습 진행 결과 시각화 툴)
# http://cafe.daum.net/flowlife/S2Ul/22

import tensorflow as tf
from tensorflow.keras.models import Sequential  # 완전연결 모델
from tensorflow.keras.layers import Dense 
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

x_data = np.array([[70, 85, 80], [71, 89, 88], [50, 45, 70], [99, 90, 90], [50, 15, 10]])
y_data = np.array([80, 85, 55, 95, 20])

model = Sequential()
# model.add(Dense(1, input_dim=3, activation='linear'))   # 레이어 1개
model.add(Dense(6, input_dim=3, activation='linear'))   # 레이어 복수 - 3개
model.add(Dense(3, activation='linear'))
model.add(Dense(1, activation='linear'))
print(model.summary())

# 학습 process 생성(컴파일)
opti = optimizers.Adam(lr=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])


# 텐서보드 : 시행착오를 알고리즘을 이용해서 최소화 할 수 있다.
from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(
        log_dir = '.\\mylog',
        histogram_freq = True,
        write_graph = True
    )
history = model.fit(x_data, y_data, batch_size=1, epochs=1000, verbose=1,
                    callbacks=[tb])

# 시각화
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 결과
from sklearn.metrics import r2_score    # 결정계수(설명력)
print('설명력 : ', r2_score(y_data, model.predict(x_data)))    # 설명력 :  0.94
print('예측값 : ', model.predict(x_data).flatten())
x_new = np.array([[20, 30, 70], [100, 70, 30]])
print('새로운 값으로 예측 : ', model.predict(x_new).flatten())




















