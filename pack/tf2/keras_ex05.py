# tf 1.x 와 2.x : 단순선형회귀/로지스틱회귀 소스 코드
# http://cafe.daum.net/flowlife/S2Ul/17

# ex1) 단순선형회귀 - 경사하강법 함수 사용 1.x 
# 실습 소스)

import tensorflow.compat.v1 as tf   # tensorflow 1.x 소스 실행 시
tf.disable_v2_behavior()            # tensorflow 1.x 소스 실행 시

import matplotlib.pyplot as plt

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]

x = tf.placeholder(tf.float32)  # placeholder : 
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x * w + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

print('\n경사하강법 메소드 사용------------')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # Gradient : 경사하강법
train = optimizer.minimize(cost)    # cost의 최소화

# ------------여기까지 모형의 설계 ---------------------------------------------------------------

sess = tf.Session()   # Launch the graph in a session.
sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for i in range(501):
    _, curr_cost, curr_w, curr_b = sess.run([train, cost, w, b], {x:x_data, y:y_data})
    w_val.append(curr_w)
    cost_val.append(curr_cost)
    if i  % 10 == 0:
        print(str(i) + ' cost:' + str(curr_cost) + ' weight:' + str(curr_w) +' b:' + str(curr_b))

# 시각화
plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('--회귀분석 모델로 Y 값 예측------------------')
print(sess.run(hypothesis, feed_dict={x:[5]}))        # [5.0563836]
print(sess.run(hypothesis, feed_dict={x:[2.5]}))      # [2.5046895]
print(sess.run(hypothesis, feed_dict={x:[1.5, 3.3]})) # [1.4840119 3.3212316]


# --------여기까지 tensorflow 1.x 버전 -----------------------------------------------------------------------


# --------여기부터 tensorflow 2.x 버전 -----------------------------------------------------------------------
print('\n\n----2.x-------------------------------------------')
# ex2) 선형회귀분석 기본  - Keras 사용 2.x 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]

model=Sequential()   # 계층구조(Linear layer stack)를 이루는 모델을 정의
model.add(Dense(1, input_dim=1, activation='linear'))   # 입력 : 1, 출력 : 1

# activation function의 종류 : https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations
sgd=optimizers.SGD(lr=0.01)  # 학습률(learning rate, lr)은 0.01
model.compile(optimizer=sgd, loss='mse',metrics=['mse'])    # 정확도가 아니기 때문에 mse나 msa를 사용한다.
# loss_metrics = model.eval‎uate(x_data,y_data)
# print(loss_metrics)

# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용.
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 100번 시도.
model.fit(x_data, y_data, batch_size=1, epochs=100, shuffle=False, verbose=2)

from sklearn.metrics import r2_score    # r2_score : 결정계수(설명력)
print('설명력 : ', r2_score(y_data, model.predict(x_data)))
print('예상 수 : ', model.predict([5]))         # [[4.801656]]
print('예상 수 : ', model.predict([2.5]))       # [[2.490468]]
print('예상 수 : ', model.predict([1.5, 3.3]))  # [[1.565993][3.230048]]









