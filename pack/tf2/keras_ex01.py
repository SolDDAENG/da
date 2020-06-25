import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # AVX를 지원하지 않는다는 에러 발생 시 기술함

# Keras 기본 개념 및 모델링 순서
# 참고 - http://cafe.daum.net/flowlife/S2Ul/10

# 1) 논리회로(OR gate) 모델 작성
# 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])
print(x)
print(y)

# 2) 모델 생성(설정)

    # 모델 생성 방법 1
# model = Sequential([
#     Dense(input_dim = 2, units=1),    # 입력 2개, units(뉴런) 출력1개 
#     Activation('sigmoid')   # 시그모이드
# ])

    # 모델 생성 방법 2
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))    # 입력은 x, y 두 개가 들어와서 출력으로 시그모이드 1개가 나간다.

# 3) 학습 process 생성(컴파일)
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])    # loss='binary_crossentropy' : 모델의 종류. 0 아니면 1이기 때문에 사용
    # optimizer='sgd' : 확률적 경사 하강법으로 코스트를 최소화한다.    metrics=['accuracy'] : 오차값, 분류 정확도를 확인
                                                        # 평균 오차를 보고싶은 경우 mse(mean square error) 적는다.
                                                        
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    # sgd -> rmsprop -> adam 순으로 보완해 만든 모델
    
    # optimizer의 클래스를 직접 사용
# model.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])  # SGD클래스를 직접 사용. lr(learning rate(학습률))를 부여
# model.compile(optimizer=SGD(lr=0.1, momentum=0.8), loss='binary_crossentropy', metrics=['accuracy'])  # SGD에는 momentum이 없다.
# model.compile(optimizer=RMSprop(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])  # RMSprop클래스를 사용.
model.compile(optimizer=Adam(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])  # Adam클래스를 사용. <- Adam이 가장 권장됨.
 
    
    
# 4) 모델 학습 (fitting)
model.fit(x, y, epochs=1000, batch_size=1, verbose=1)  # epochs : 훈련 횟수, verbose : 1.진행과정 보기 0.진행과정 보지않고 빠르게 결과가 나옴. 
    # batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정 - 배치 크기는 학습에 큰 영향을 미침
    
# 5) 모델 평가 - 여기서 결과가 이상하면 오버피팅. 못풀면 모델을 여러 개 써보자.
loss_matrics = model.evaluate(x, y)
print(loss_matrics) # 분류 정확도

# 6) 학습결과 예측값 출력
pred1 = model.predict(x)
print('pred1 : ', pred1)
pred2 = (model.predict(x) > 0.5).astype('int32')
print('pred2 : ', pred2)


print('\n\n--모델 저장 및 읽기----------------------------------------------------------')
# 모델 저장
model.save('test.hdf5') # 확장자  hdf5 : 대용량의 데이터 파일
del model   # 모델을 저장하고 지운다.

# pred3 = (model.predict(x) > 0.5).astype('int32')    # 모델을 지워서 읽을 수 없기 때문에 에러.
# print('pred3 : ', pred3)


# 저장한 모델 읽기
from tensorflow.keras.models import load_model
model2 = load_model('test.hdf5')

pred3 = (model2.predict(x) > 0.5).astype('int32')    # 모델을 지워서 읽을 수 없기 때문에 에러.
print('pred3 : ', pred3)








