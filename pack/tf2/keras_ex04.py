# 선형회귀 모형 계산
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # AVX를 지원하지 않는다는 에러 발생 시 기술함

x = [1.,2.,3.,4.,5.]
y = [1.2,2.0,3.0,3.5,5.5]

w = tf.Variable(tf.random.normal((1,))) # tuple
b = tf.Variable(tf.random.normal((1,)))
opti = tf.keras.optimizers.SGD()

# cost function : 내부적으로 미분을 계산
def train_step(x, y):   
    with tf.GradientTape() as tape: # tape에 녹음을 한다? 담아둔다?    경사하강법
        hypo = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
    grad = tape.gradient(loss, [w, b])  # 여기서 미분을 하고있다.
    opti.apply_gradients(zip(grad, [w,b]))  # zip() : 짝을 지어서 넘긴다.
    return loss
    
w_vals = []
loss_vals = []

# 학습
for i in range(100):
    loss_val = train_step(x, y)   # 최적의 w_val을 얻을 수 있다.
    loss_vals.append(loss_val.numpy())
    w_vals.append(w.numpy()) # 랜덤한 값이 저장된다.
    #print(loss_val)

print('w_vals : ', w_vals)
print('loss_vals : ', loss_vals)


# 시각화
import matplotlib.pyplot as plt
plt.plot(w_vals, loss_vals, 'o--')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()


# 예측
y_pred = tf.multiply(x, w) + b  # 경사 하강법을 통해서 w와 b를 얻음
print(y_pred.numpy())

plt.plot(x, y, 'ro')
plt.plot(x, y_pred, 'b-')   # 잔차(loss)가 최소가 되는 회귀식 - 이제 이런 내용에 keras를 적용시켜 보자
plt.show()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        