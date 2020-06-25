# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import tensorflow as tf

print(tf.__version__)  # 2.2.0
print(tf.keras.__version__)  # 2.3.0-tf
# print('GPU 사용 가능 여부 : ', '가능' if tf.test.is_gpu_available() else '불가능')
print('GPU 사용 가능 여부 : ', '가능' if tf.config.list_physical_devices('GPU') else '불가능')

# tensor의 이해 : tf의 기본 구성 요소. 데이터를 위한 컨테이너로 대개의 경우 수치 데이터를 다루는 수치용 컨테이너
# 임의 차원 갯수를 가지는 행렬의 일반화된 객체.
# 상수 정의 (상수 텐서를 생성)
print(tf.constant(1))  # scala : 0차원 텐서
print(tf.constant([1]))  # vector : 1차원 텐서    shape(1,)
print(tf.constant([[1]]))  # matrix : 2차원 텐서
print(tf.constant([[1, 2]]))  # matrix : 2차원 텐서

print(tf.rank(tf.constant(1.)), ' ', tf.rank(tf.constant([[1]])))  # rack() : 차원을 알 수 있다.
# tf.Tensor(0, shape=(), dtype=int32) ==> 0차원 ,  tf.Tensor(2, shape=(), dtype=int32) ==> 2차원
print(tf.constant(1.).get_shape(), ' ', tf.constant([[1]]).get_shape())  # constant() : 크기를 알 수 있다. 

print()
a = tf.constant([1, 2])
b = tf.constant([3, 4])
# c = a + b
c = tf.add(a, b)  # tf가 지원하는 연산함수
print(c, type(c))  # tensor
print(c, c.numpy())  # numpy

print('-------------------')
# d = tf.constant([3])  # 1차원
# d = 3  # 텐서로 변환
d = tf.constant([[3]])  # 2차원
e = c + d
print(e)  # Broadcast 연산

# 상수를 텐서화
print(7, type(7))
print(tf.convert_to_tensor(7))
print(tf.cast(7, dtype=tf.float32))

# numpy의 ndarray와 tensor 사이에 자동 변환
import numpy as np
arr = np.array([1, 2])
print(arr, ' ', type(arr))
tfarr = tf.add(arr, 5)
print(tfarr)
print(tfarr.numpy(), ' ', type(tfarr.numpy()))
print(np.add(tfarr, 3))