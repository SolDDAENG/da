# 변수
import tensorflow as tf
import numpy as np

f = tf.Variable(1.0)  # 변수형 텐서에 scala 값 기억    # Variable class type
v = tf.Variable(tf.ones((2,)))
m = tf.Variable(tf.ones((2, 1)))
print(f)
print(v, v.numpy())
print(m)

print()
v1 = tf.Variable(1)
print(v1)
v1.assign(10)
print(v1, ' ', v1.numpy(), ' ', type(v1))

print()
v2 = tf.Variable(tf.ones(shape=(1)))  # 1차원 텐서
v2.assign([20])  # 1차원 텐서이므로 배열값 할당
print(v2, ' ', type(v2))

print()
v3 = tf.Variable(tf.ones(shape=(1, 2)))  # 2차원 텐서
v3.assign([[30, 40]])  # 2차원 텐서이므로 2차원 배열값으로 할당 - 차원 맞춰야한다.
print(v3, ' ', type(v3))

print()
v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
print(v3.numpy())

var = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
result1 = var + 10
print(result1)

w = tf.Variable(tf.ones(shape=(1,)))
b = tf.Variable(tf.ones(shape=(1,)))
w.assign([3])
b.assign([2])


def func1(x):
    return w * x + b


out_a1 = func1([3])
print('out_a1 : ', out_a1)

print()

w = tf.Variable(tf.ones(shape=(1, 2)))
b = tf.Variable(tf.ones(shape=(1,)))
w.assign([[2, 3]])


@tf.function  # autograph 가능 (내부적으로 tf.Graph + tf.Session) : 속도가 빨라짐    = 일반 함수를 텐서의 그래프로 바꿔주고 세션으로 돌린다.
# 단점은 디버깅이 불편함. 처음에는 없이 구현하고 다 끝나면 그때 적용해주면 좋다.
def func2(x):
    return w * x + b


out_a2 = func2([3])
print(out_a2)

print('\n -----------------------------------')
w = tf.Variable(tf.keras.backend.random_normal([5, 5], mean=0, stddev=0.3))  # random_normal : 정규분포를 따르는 난수 발생
print(w.numpy().mean())
print(np.mean(w.numpy()))
print(w)
b = tf.Variable(tf.zeros([5]))
print(b * w)

print()
rand1 = tf.random.normal([4], 0, 1)  # 평균 : 0, 표준편차 : 1
print('rand1 : ', rand1)
rand2 = tf.random.uniform([4], 0, 1)  # 최소값 : 0, 최대값 : 1 위의 normal과 다름
print('rand2 : ', rand2)

# 변수 치환 좀 더....
aa = tf.ones((2, 1))
print(aa.numpy())

m = tf.Variable(tf.zeros((2, 1)))
m.assign(aa)  # 치환
print(m.numpy())

m.assign(m + 10)
print(m.numpy())

m.assign_add(aa)    # 값을 증가 +=
print(m.numpy())

m.assign_sub(aa)    # 값을 감소 -=
print(m.numpy())