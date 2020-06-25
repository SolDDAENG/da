import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# 1) 논리회로(OR gate) 모델 작성
# 데이터 수집 및 가공
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
print(x)
print(y)

# 2) 모델 생성(설정)
# model = Sequential([
#     Dense(input_dim = 2, units=1),
#     Activation('sigmoid')
# ])
# 위를 아래처럼 풀어서 쓸 수 있다.
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))

# 3) 학습 process 생성 (컴파일)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 4) 모델 학습
model.fit(x, y, epochs=1000, batch_size=1, verbose=1)  # 학습 중 출력되는 문구를 설정하는 것

# 5) 모델 평가
loss_metrics = model.evaluate(x, y)
print(loss_metrics)  # 분류 정확도

# 6) 예측값 출력
pred1 = model.predict(x)
print('pred1 : ', pred1)
pred2 = (model.predict(x) > 0.5).astype('int32')
print('pred2 : ', pred2)
pred3 = model.predict_classes(x)
print('pred3 : ', pred3)