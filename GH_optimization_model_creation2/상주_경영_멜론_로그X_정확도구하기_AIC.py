import pandas as pd
import os
import numpy as np
from datetime import date, datetime, timedelta
import datetime
import tensorflow
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pickle import load
from pickle import dump
from sklearn.metrics import f1_score, recall_score, accuracy_score,mean_squared_error,mean_absolute_error
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def AIC(Y, df):

    custom_x = dict()

    for i in Y:
    ## 전진 선택법
        variables = df[['tp_x', 'hd_x',	'co2',	'tp_y',	'sr',	'wd',	'ws',	'pc']].columns.tolist() ## 설명 변수 리스트
        
        y = df[i] ## 반응 변수
        selected_variables = [] ## 선택된 변수들
        sl_enter = 0.05
        
        sv_per_step = [] ## 각 스텝별로 선택된 변수들
        adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
        steps = [] ## 스텝
        step = 0
        while len(variables) > 0:
            remainder = list(set(variables) - set(selected_variables))
            pval = pd.Series(index=remainder) ## 변수의 p-value
            ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서 
            ## 선형 모형을 적합한다.
            for col in remainder: 
                X = df[selected_variables+[col]]
                X = sm.add_constant(X)
                model = sm.OLS(y,X).fit()
                pval[col] = model.pvalues[col]
        
            min_pval = pval.min()
            if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
                selected_variables.append(pval.idxmin())
                
                step += 1
                steps.append(step)
                adj_r_squared = sm.OLS(y,sm.add_constant(df[selected_variables])).fit().rsquared_adj
                adjusted_r_squared.append(adj_r_squared)
                sv_per_step.append(selected_variables.copy())
            else:
                print(i)  
                print(selected_variables)
                custom_x[i] = selected_variables
                # fig = plt.figure(figsize=(10,10))
                # fig.set_facecolor('white')
                
                # font_size = 15
                # plt.xticks(steps,[f'step {s}\n'+'\n'.join(sv_per_step[i]) for i,s in enumerate(steps)], fontsize=12)
                # plt.plot(steps,adjusted_r_squared, marker='o')
                    
                # plt.ylabel('Adjusted R Squared',fontsize=font_size)
                # plt.grid(True)
                # plt.show()
                break
    return custom_x 

# 연속형이면 y 로그변환 범주형이라면 그대로 유지
# def data_indicator(i,targets):
#     if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
#         targets = targets
#     else:
#         targets = np.log1p(targets)

#     return targets

# # 연속형이면 예측값 e제곱 범주형이라면 그대로 유지
# def pred_indicator(i,pred):
#     if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
#         pred = pred
#     else:
#         pred = np.expm1(pred)

#     return pred

def custom_scaling_make_dataset(key, input_data, targets, Ntrain, T, D):
    
    #x값 스케일링
    X_scaler = MinMaxScaler()
    X_scaler.fit(input_data[:Ntrain])

    input_data_train_scaled = X_scaler.transform(input_data[:Ntrain])
    input_data_valid_scaled = X_scaler.transform(input_data[Ntrain:])

    #타겟값 로그변환
    #targets = data_indicator(key,targets)

    targets_train = targets[:Ntrain]
    targets_valid = targets[Ntrain:]

    sc_input_data = np.vstack((input_data_train_scaled, input_data_valid_scaled))
    sc_targets = np.vstack((targets_train, targets_valid))

    X=np.zeros((len(input_data) - T - 1, T, D))
    Y=np.zeros((len(input_data) - T - 1, targets.shape[1]))
    
    for t in range(len(input_data) - T - 1):

        X[t, :, :]= sc_input_data[t:t + T]
        
        Y[t]=(sc_targets[t + T + 1][0])  #정답인 수 를 넣기 
          
    #학습
    X_train = X[:Ntrain]
    Y_train = Y[:Ntrain]

    #검증
    X_valid = X[Ntrain:]
    Y_valid = Y[Ntrain:]

    return X_train, Y_train, X_valid, Y_valid, X_scaler

def threshold(x):
  if x < 0.5:
    x = 0.0
  else:
    x = 1.0
  return x

def numerical_model(T, input_data, X_train, Y_train, X_valid, Y_valid,i):

        # 모델학습

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        # 콜백함수
        mc = ModelCheckpoint("./상주_멜론_모델파일_AIC/상주_멜론_모델파일_AIC_"+i+".h5", save_best_only=True,monitor='val_loss')
        # 모델구성
        model = Sequential([LSTM(16, input_shape=[T, input_data.shape[1]], activation='relu')])
        # repeat vector
        model.add(RepeatVector(1))

        # decoder layer
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam',
                    loss='mean_squared_error',metrics=['mse','mae'])
        r = model.fit(
            X_train,
            Y_train,  # 훈련할 데이터 넣기
            batch_size=64,
            epochs=1000,
            validation_data=(X_valid, Y_valid),
            callbacks=[early_stop,mc]
        )
        return r,model

def binary_model(T, input_data, X_train, Y_train, X_valid, Y_valid,i):

        # 모델학습

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
        # 콜백함수
        mc = ModelCheckpoint("./상주_멜론_모델파일_AIC/상주_멜론_모델파일_AIC_"+i+".h5", save_best_only=True,monitor='val_accuracy')
        # 모델구성
        model = Sequential([LSTM(16, input_shape=[T, input_data.shape[1]], activation='relu')])
        #repeat vector
        model.add(RepeatVector(1))

        #decoder layer
        model.add(LSTM(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',metrics=['accuracy'])
        r = model.fit(
            X_train,
            Y_train,  # 훈련할 데이터 넣기
            batch_size=64,
            epochs=1000,
            validation_data=(X_valid, Y_valid),
            callbacks=[early_stop,mc]
        )
        return r,model


if __name__ == "__main__":

    df  = pd.read_csv('D:/theimz/조석준/NAVER_WORKS/스마트팜_혁신밸리/22년12월/모델고도화코드개발/스마트팜_혁신밸리_고도화_모델코드(전달)/상주_온실별_작기별_데이터/상주_경영_멜론_2_5_iqr_이상치대체_10분단위_1220_ver1.csv')

    Y = ['sky_window','cg_curtain','flow_fan', 'hmdfc', 'crc_pump', '3way_valve','heat_cooler']
    
    custom_x = AIC(Y, df)

    predicted_value = pd.DataFrame()
    test_value = pd.DataFrame()

    with tensorflow.device("/device:GPU:0"):

        for i in custom_x.keys():
            print(i)

            k = list
            k = custom_x[i]
            k.append(i)
            print(k)
            # x값
            input_data = df[k].values
            # 예측해야하는 y값
            targets = df[[i]].values

            # 데이터 train,valid 나누기(여기서는 70% train, 30% valid data)
            Ntrain = len(input_data)* 8//10

            tensorflow.random.set_seed(2022)

            # 예측할때 사용할 시계열 데이터(여기서는 10분 주기의 데이터 48개 => 480분을 가지고 미래의 10분을 예측하도록 설정)
            T = 48

            # y컬럼의 갯수
            D = input_data.shape[1]


            # train,valid 데이터셋 분리
            X_train, Y_train, X_valid, Y_valid, scaler = custom_scaling_make_dataset(i, input_data, targets, Ntrain, T, D)
            # 스케일러 저장
            dump(scaler, open('./상주_멜론_스케일러_AIC/상주_멜론_스케일러_AIC_'+i+'.pkl', 'wb'))
            # test_value = pd.concat([test_value, pd.DataFrame(Y_valid, columns =[i])] ,axis=1  )

            print(i+'구하는 모델')

            #연속형,범주형에따라 함수 구분
            if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
                r,GH_optimization_model = binary_model(T, input_data, X_train, Y_train, X_valid, Y_valid,i)
            else:
                r,GH_optimization_model = numerical_model(T, input_data, X_train, Y_train, X_valid, Y_valid,i)

            test_value = pd.concat([test_value, pd.DataFrame(Y_valid, columns =[i])] ,axis=1  )
            #스케일링을 풀어주는 함수
            #predicted_value = pd.concat([predicted_value, pd.DataFrame(pred_indicator(i,GH_optimization_model.predict(X_valid).reshape(-1,1)) ,columns = [i])] ,axis=1)

            #모델 불러오기
            model = tensorflow.keras.models.load_model('D:/theimz/조석준/NAVER_WORKS/스마트팜_혁신밸리/22년12월/모델고도화코드개발/스마트팜_혁신밸리_고도화_모델코드(전달)/상주_온실별_작기별_데이터/상주_멜론_모델파일_AIC/상주_멜론_모델파일_AIC_'+i+'.h5')

            #스케일링 유지
            #연속형,범주형에따라 함수 구분
            if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
                predicted_value = pd.concat([predicted_value, pd.DataFrame((model.predict(X_valid) > 0.5).astype(int).reshape(-1,1) ,columns = [i])] ,axis=1)
                
            else:
                predicted_value = pd.concat([predicted_value, pd.DataFrame(model.predict(X_valid).reshape(-1,1) ,columns = [i])] ,axis=1)

            

        # test_value.to_csv('상주_test_value.csv',index=False, header = True)
        # predicted_value.to_csv('상주_predicted_value.csv',index=False, header = True)

        y_pred = predicted_value[['flow_fan',	'hmdfc',		'crc_pump','heat_cooler']].astype(int)
        y_true = test_value[['flow_fan',	'hmdfc',	'crc_pump','heat_cooler']].astype(int)



        상주_범주형_성능지표 = pd.DataFrame(columns = ['flow_fan',	'hmdfc','crc_pump',	'heat_cooler'])
        
        count = 0
        list_of_lists = []

        for i in 상주_범주형_성능지표.columns:

            acc = accuracy_score(y_true[i], y_pred[i])
            list_of_lists.append(acc)

        상주_범주형_성능지표.loc[count] = list_of_lists

        count = count + 1
        list_of_lists = []

        for i in 상주_범주형_성능지표.columns:

            recall = recall_score(y_true[i], y_pred[i])
            list_of_lists.append(recall)

        상주_범주형_성능지표.loc[count] = list_of_lists

        count = count + 1
        list_of_lists = []

        for i in 상주_범주형_성능지표.columns:

            f1 = f1_score(y_true[i], y_pred[i])
            list_of_lists.append(f1)

        상주_범주형_성능지표.loc[count] = list_of_lists

        상주_연속형_성능지표 = pd.DataFrame(columns = ['sky_window','cg_curtain','3way_valve'])

        #스케일링 풀지 않고 구함

        상주_연속형_성능지표.loc[0] = [mean_squared_error(test_value['sky_window'], predicted_value['sky_window']),mean_squared_error(test_value['cg_curtain'], predicted_value['cg_curtain']),\
            mean_squared_error(test_value['3way_valve'], predicted_value['3way_valve'])]

        상주_연속형_성능지표.loc[1] = [mean_absolute_error(test_value['sky_window'], predicted_value['sky_window']),mean_absolute_error(test_value['cg_curtain'], predicted_value['cg_curtain']),\
            mean_absolute_error(test_value['3way_valve'], predicted_value['3way_valve'])]


        상주_범주형_성능지표.to_csv('상주_멜론_범주형_성능지표.csv',index=False, header = True)
        상주_연속형_성능지표.to_csv('상주_멜론_연속형_성능지표.csv',index=False, header = True)