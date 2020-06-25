# Keras로 자동차 연비 예측
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers

dataset = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/auto-mpg.csv')
print(dataset.head(2))
del dataset['car name']
print(dataset.head(2))
print(dataset.info())  # 각 칼럼 정보 확인 : horsepower(object) 자료 중에 ? 포함

# 강제 형 변환 시 ValueError 를 무시하기 : errors='coerce'
dataset['horsepower'] = dataset['horsepower'].apply(pd.to_numeric, errors='coerce')
print(dataset.info())
print(dataset.isna().sum())  # NaN 확인    horsepower      6
dataset = dataset.dropna()
print(dataset.isna().sum())  # NaN 확인    horsepower      0

# 시각화
sns.pairplot(dataset[['mpg', 'weight', 'horsepower']], diag_kind='kde')
plt.show()

# train / test 데이터 분리  - train_test_split이외의 방법
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)  # train의 70%를 제외한 나머지 30% 
print(train_dataset.shape)  # (274, 8)
print(test_dataset.shape)  # (118, 8)

train_stat = train_dataset.describe()  # 요약 통계값 확인
train_stat.pop('mpg')
train_stat = train_stat.transpose()
print(train_stat)

# label : 'mpg'
train_labels = train_dataset.pop('mpg')
print(train_labels[:2])
test_labels = test_dataset.pop('mpg')
print(test_labels[:2])


# 표준화 작업
def st_func(x):  # 표쥰화 처리 함수 : (요소값 - 평균) / 표준편차
    return ((x - train_stat['mean']) / train_stat['std'])


# print('st_func(10) : ', st_func(10))
print(st_func(train_dataset[:3]))   
st_train_data = st_func(train_dataset)
st_test_data = st_func(test_dataset)

# 모델 작성 후 예측
print('\n모델 작성 후 예측 -------------------------------------------')


    # 모델 작성용 함수
def build_model():
    network = tf.keras.Sequential([  # model의 이름을 network라고 주었다.
        layers.Dense(units=32, activation=tf.nn.relu, input_shape=[7]),  # 미리 없앤 car name을 빼고, mpg를 제외한 나머지 7개 컬럼.
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # 레이어 개수 : 3        
        ])
    # 학습 process 생성(컴파일)
    # opti = tf.keras.optimizers.RMSprop(0.001)
    opti = tf.keras.optimizers.Adam(0.01)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])  # mse, mae
    
    return network


model = build_model()
print(model.summary())

# fit() 전에 모델을 실행해 볼 수도 있다.
print(model.predict(st_train_data[:3]))  # 결과는 신경쓰지 않음. 실행이 되는지만 확인

# 모델 학습 (fitting)
epochs = 5000

# 학습 조기 종료
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # patience = 5 : val_loss에서 같은 값이 5번 나오면 조기종료

history = model.fit(st_train_data, train_labels,
                    epochs=epochs, verbose=1, validation_split=0.2,
                    callbacks=[early_stop])

df = pd.DataFrame(history.history)
print(df.head(3))
print(df.columns)
# ['loss', 'mean_squared_error', 'mean_absolute_error',     <= mae때문에 이 값도 보인다.
# 'val_loss', 'val_mean_squared_error', 'val_mean_absolute_error']    <= val_~ 값들은 validation_split를 사용해서 나오는 것이다.

# from IPython.display import display    # 참고 : jupiter에서 실행하면 칼럼명 모두 보임
# display(df.head(3)) 


# history 객체에 저장된 통계치를 사용해 모델의 훈련 과정을 시각화 코드 
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8, 12))
    
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)

# 모델 평가
loss, mae, mse = model.evaluate(st_test_data, test_labels)
# ma : 절대 오차(Absolute Error)는 측정값에서 오차의 크기. 측정값과 실제값의 차이
# mae : 평균 절대 오차(MAE) : 모든 절대 오차의 평균 - 일반적인 회귀 지표는 평균 절대 오차(MAE)
# mse : 평균 제곱 오차 : 잔차(오차)의 제곱에 대한 평균을 취한 값 - 평균 제곱 오차(MSE)는 회귀에서 자주 사용되는 손실 함수
print('test dataset으로 평가 mae : '.format(mae))
print('test dataset으로 평가 mse : '.format(mse))
print('test dataset으로 평가 loss : '.format(loss))

# 예측 : 주의 - 새로운 데이터로 예측을 원한다면 표준화 작업을 선행
test_pred = model.predict(st_test_data).flatten()
print(test_pred)

plt.scatter(test_labels, test_pred)
plt.xlabel('True value[mpg]')
plt.ylabel('pred value[mpg]')
plt.show()

# 오차 분포 확인 (정규성 : 잔차향이 정규분포를 따르는지 확인)
err = test_pred
plt.hist(err, bins=20)
plt.xlabel('pred error[mpg]')
plt.show()