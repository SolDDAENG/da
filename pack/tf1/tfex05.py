# 연산자와 기본 함수
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = 3

result1 = tf.cond(x > y, lambda:tf.add(x, y), lambda:tf.subtract(x, y))
print(result1.numpy())

f1 = lambda :tf.constant(1)
print(f1())
f2 = lambda :tf.constant(2)
a = tf.constant(3)
b = tf.constant(4)

# 안녕하세요 저는 최한솔입니다.

result2 = tf.case([(tf.less(a, b), f1)], default=f2)
print(result2.numpy())

print('관계연산자')
print(tf.equal(1, 2).numpy())
print(tf.not_equal(1, 2))
print(tf.less(1, 2))
print(tf.greater(1, 2))
print(tf.greater_equal(1, 2))

print('논리연산자')
print(tf.logical_and(True, False))  # logical_or    logical_not

print()
kbs = tf.constant([1, 2, 2, 2, 3])  # contant : 정수
val, ind = tf.unique(kbs)
print(val)
print(ind)

print('차원 관련 ---')
ar = [[1, 2], [3, 4]]
print(tf.reduce_sum(ar))
print(tf.reduce_mean(ar))  # reduce_mean : 텐서의 차원에서 요소의 평균을 계산
print(tf.reduce_mean(ar, axis=0))
print(tf.reduce_mean(ar, axis=1))

print()
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]) 
print(t.shape)  # (2, 2, 3) : 2연 2행 3열 => 3차원 
print(tf.reshape(t, shape=[2, 6]))
print(tf.reshape(t, shape=[-1, 6]))  # -1 하면 자동으로 행의 개수 정렬
print(tf.reshape(t, shape=[2, -1]))

print()
print(tf.squeeze(t))  # 차원 축소 (열 요소 수가 1인 경우만 해당)
aa = np.array([[1], [2], [3], [4]])
print(aa.shape)  # 2차원
bb = tf.squeeze(aa)
print(bb.shape)  # 1차원

print()
tarr = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.shape(tarr))  # [2, 3
sbs = tf.expand_dims(tarr, 0)  # expand_dims(늘린변수, 0) : 0번째 차원을 확대
print(sbs, ' ', tf.shape(sbs))  # (1, 2, 3)
sbs = tf.expand_dims(tarr, 1)  # expand_dims(늘린변수, 1) : 1번째 차원을 확대
print(sbs, ' ', tf.shape(sbs))  # [2 1 3]
sbs = tf.expand_dims(tarr, 2)  # expand_dims(늘린변수, 2) : 2번째 차원을 확대
print(sbs, ' ', tf.shape(sbs))  # [2 3 1]
sbs = tf.expand_dims(tarr, -1)  # expand_dims(늘린변수, -1) : -1번번째 차원을 확대
print(sbs, ' ', tf.shape(sbs))  # [2 3 1]

print()
print(tf.one_hot([0, 1, 2, 0], depth=2))  # 0 : [1. 0.]    1 : [0. 1.]    2 : [0. 0.] => 0, 1, 2는 숫자가 아닌 종류로 바야한다.
print(tf.one_hot([2, 5, 1, 1], depth=4))  # depth : 열의 갯수

print()
print(tf.cast([1, 2, 3, 5], tf.float32))  # 정수를 실수로 형변환
a = 5
print(tf.cast(a > 7, tf.float32))  # 조건이 참이면 1.0, 거짓이면 0.0

print()
x = [1, 4]
y = [2, 5]
z = [3, 6]
print(x, y, z)
print(tf.stack([x, y, z]))  # stack : 여러 조각의 행렬을 합칠 때 사용
print(tf.stack([x, y, z], axis=0))
print(tf.stack([x, y, z], axis=1))

print()
x = np.array([[0, 1, 2], [2, 1, 0]])
print(x)
print(tf.ones_like(x))
print(tf.zeros_like(x))

# ...
