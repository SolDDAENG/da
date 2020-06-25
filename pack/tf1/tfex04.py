# constant() : 텐서(일반적인 상수 값)를 직접 기억 
# Variable() : 텐서가 저장된 주소를 기억

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # AVX를 지원하지 않는다는 에러 발생 시 기술함

a = 10
print(a, type(a))
print('--------------')
b = tf.constant(10)
print(b, type(b))
print('--------------')
c = tf.constant(10)
print(c, type(c))

print()
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# node1 = tf.Variable(3.0, tf.float32)
# node2 = tf.Variable(4.0)
print(node1)
print(node2)
node3 = tf.add(node1, node2)
print(node3)

print('-----------------------')
# v = tf.Variable(1)
v = tf.Variable(2)


@tf.function
def find_next_odd():
    abc()  # autograph 지원 함구가 다른 함수를 호출하면 해당 함수도 autograph가 됨
    v.assign(v + 1)  # v는 함수 밖에서 정의했으므로 전역함수
    if tf.equal(v % 2, 0):  # v를 2로 나눈 나머지가 0이라면
        v.assign(v + 10)


def abc():
    print('abc')


find_next_odd()
print(v.numpy())

print('----1 ~ 3 까지 숫자 증가---------------------')


def func():
    imsi = tf.constant(0)  # imsi = 0
    su = 1
    for _ in range(3):
        imsi = tf.add(imsi, su)  # 누적
    return imsi


kbs = func()
print(kbs.numpy(), ' ', np.array(kbs))

print('----------------------')
imsi = tf.constant(0)


def func2():
    su = 1
    global imsi
    for _ in range(3):
        imsi = tf.add(imsi, su)
    return imsi 


# mbc = func2()
# print(mbc.numpy(), ' ', np.array(mbc))
mbc = func2
print(mbc().numpy())

print('$$$$$$$$$$$$$$$$$')


def func3():
    imsi = tf.Variable(0)
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)  # 누적        
        imsi.assign_add(su)
    return imsi


kbs = func3()
print(kbs.numpy())

print()
imsi = tf.Variable(0)  # 데코레이터(ex__ @tf.function) 위에 변수 써주기 
@tf.function
def func4():
    # imsi = tf.Variable(0)    # ValueError
    su = 1
    for _ in range(3):
        imsi.assign_add(su)
    return imsi

mbc = func4()
print(mbc.numpy())

print('구구단 출력 ---------------')
# @tf.function
def gugu1(dan):
    su = 0
#     aa = tf.constant(5)
#     print(aa.numpy())   # autograph 내에서는 .numpy()를 쓸 수 없다.
    
    for _ in range(9):
        su = tf.add(su, 1)
        print('{} x {} = {:2}'.format(dan, su, dan * su))   # .format()도 안된다.
        # TypeError: unsupported format string passed to Tensor.__format__
        
gugu1(3)

# @tf.function
print()
def gugu2(arg):
    for i in range(1, 10):
        result = tf.multiply(arg, i)
        print('{} x {} = {:2}'.format(arg, i, result))

gugu2(5)
    