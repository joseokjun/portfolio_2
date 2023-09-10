# ----------------------------------------------------------------
# python 3.8 ver
# 2022.12.16 최초 작성
# [스마트팜 - 혁신밸리] 온실데이터 예측 모델 고도화
# 1차 개발 - theimc 서울본부 R팀 조석준, 구지은
# 2차 개발 - theimc 서울본부 R팀 조석준, 구지은
# 해당코드는 서울서버 경로기준으로 작성되었습니다.
# 10분마다 실행 모델
#
# author : jsj
# ----------------------------------------------------------------
import os
import sys
import inspect
import pymysql
import warnings
import traceback
import numpy as np
import pandas as pd
from pickle import load
import logging.config
from config import Config
from multiprocessing import Pool
from datetime import timedelta, datetime

import tensorflow
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout, RepeatVector
import pytz


warnings.filterwarnings('ignore')
f = open('/home/seoul/joseokjun/smartfarm/smartfarm_ver2/smartfarm_ver2_code/logging.conf','r',encoding='utf8')
logging.config.fileConfig(f)
f.close()
logger = logging.getLogger('GH_LOG')
tz = pytz.timezone('Asia/Seoul')
cur_time = datetime.now(tz)
simple_cur_time = cur_time.strftime("%Y:%m:%d:%H:%M")
now_time = datetime.strptime(simple_cur_time, "%Y:%m:%d:%H:%M")

# 데이터 스케일링 & train_test 데이터셋 만들기
def scaling_make_dataset(self,df,var,i):
    try:
        #x값 스케일링
        X_scaler = load(open('/home/seoul/joseokjun/smartfarm/smartfarm_ver2/smartfarm_ver2_code/model_scaler/sangu_'+crop_nm_indicator(self)+'_scaler/sangu_'+crop_nm_indicator(self)+'_scaler_'+i+'.pkl', 'rb'))
        Pred_data = X_scaler.transform(df.reset_index()[var].values)
        Pred_data = Pred_data.reshape(1, 48, Pred_data.shape[1])

        return Pred_data
    except:
        logger.error(traceback.format_exc())
        logging.error("train 데이터셋 만드는 중 오류 발생")
        return

# 연속형이면 예측값 e제곱 범주형이라면 그대로 유지
# def pred_indicator(i,pred):
#     if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
#         pred = pred
#     else:
#         pred = np.expm1(pred)

#     return pred
# 연속형이면 값 반올림, 범주형이면 0.5기준 0,1나누는 함수
def value_standard(i,output):
  if i == 'flow_fan' or i == 'hmdfc' or i == 'crc_pump' or i == 'heat_cooler':
      output = threshold(output)
  else:
      output = round(output)

  return output

# 확률로 나온 결과 값을 0.5를 기준으로 0,1로 구분하는 함수
def threshold(x):
    if x >= 0.5:
        x = 1
    else:
        x = 0
    return x
def crop_nm_indicator(self):
    if self.crop_nm == '딸기':
        return "strawberry"
    elif self.crop_nm == '토마토':
        return "tomato"
    elif self.crop_nm == '멜론':
        return "melon"
    else:
        return "mandarin"
# DB 연결
def db_connect():
    try:
        conn = pymysql.connect(host=Config.DB_HOST, port=Config.DB_PORT, user=Config.DB_USER, password=Config.DB_PW,
                               db=Config.DATABASE, charset="utf8", autocommit=True)
        logger.debug("DB 연결 정상")
        return conn
    except:
        logger.error(traceback.format_exc())
        logger.error("DB 연결 오류 발생")

class GHModel():
    def __init__(self, model):
        self.fclty_nm = model[0]
        self.crop_nm = model[1]
        

    # db에서 데이터 가져오기
    def get_data(self):
        try:
            conn = db_connect()

            sql_in = "SELECT * FROM (SELECT * FROM gh_indoor_env_info WHERE fclty_nm="+"'"+self.fclty_nm+"'"+\
                     " ORDER BY reg_date DESC LIMIT 48) AS a ORDER BY a.reg_date asc;"
            sql_out = "SELECT * FROM (SELECT * FROM gh_outdoor_env_info WHERE fclty_nm=" +"'"+self.fclty_nm[:3]+"'"+\
                      " ORDER BY reg_date DESC LIMIT 9) AS a ORDER BY a.reg_date asc;"
            sql_con = "SELECT * FROM (SELECT * FROM gh_indoor_mal_info WHERE fclty_nm="+"'"+self.fclty_nm+"'"+\
                      " ORDER BY reg_date DESC LIMIT 48) AS a ORDER BY a.reg_date asc;"

            df_in = pd.read_sql(sql_in, conn)
            df_out = pd.read_sql(sql_out, conn)
            df_con = pd.read_sql(sql_con, conn)

            return df_in, df_out, df_con
        except:
            logger.error(traceback.format_exc())
            logger.error("DB 에서 데이터 가져오기 오류 발생")
            return

    
    def refine_data(self, df_in, df_out, df_con):
        logger.debug("["+str(os.getpid()) + "]"+inspect.currentframe().f_code.co_name + " start")

        if '메론' in df_in['crop_nm'].values:
            df_in = df_in.replace({'crop_nm':'메론'}, '멜론')
    
        if '메론' in df_con['crop_nm'].values:
            df_con = df_con.replace({'crop_nm':'메론'}, '멜론')
        
        # 필요한 열만 추출
        df_in = df_in[['reg_date', 'fclty_nm', 'crop_nm', 'tp', 'hd', 'co2']]      # tp_x, hd_x
        df_out = df_out[['reg_date', 'tp', 'sr', 'wd', 'ws', 'pc']]            # tp_y, hd_y
        df_con = df_con[['reg_date', 'sky_window', 'cg_curtain', 'flow_fan', 'hmdfc', 'crc_pump', '3way_valve','heat_cooler']]

        return df_in, df_out, df_con
    # iloc->loc로 수정
    def preprocess_in(self,df_in):
        try:
            # 현재 시간 단위로 모든 데이터를 수정
            for i in range(0, 48):
                # 10분 * i 전의 시간
                temp = now_time - timedelta(minutes=10*(47-i))

                # 날짜 데이터 수정
                df_in['reg_date'].loc[i] = datetime(temp.year, temp.month, temp.day, temp.hour, (temp.minute - temp.minute % 10))
            
            return df_in
        except:
            logger.error(traceback.format_exc())
            logger.error("preprocess_in오류 발생")

    # iloc->loc로 수정
    def preprocess_con(self,df_con):
    # 현재 시간 단위로 모든 데이터를 수정
        try:
            for i in range(0, 48):
                # 10분 * i 전의 시간
                temp = now_time - timedelta(minutes=10 * (47 - i))

                # 날짜 데이터 수정
                df_con['reg_date'].loc[i] = datetime(temp.year, temp.month, temp.day, temp.hour, (temp.minute - temp.minute % 10))
            
            return df_con
        except:
            logger.error(traceback.format_exc())
            logger.error("preprocess_con오류 발생")

    def preprocess_out(self,df_in,df_con,df_out):
        # 최근 시간에 맞추어 전처리 필요
        for i in range(len(df_out)):
            # 1 * n 시간 전의 시간
            temp = now_time - timedelta(hours= (len(df_out) - (i+1)))

            # 날짜 데이터 수정
            df_out['reg_date'].iloc[i] = datetime(temp.year, temp.month, temp.day, temp.hour)

        # 기존 한 시간 단위의 시간에서 10분 단위에 시간을 표현하기 위한 컬럼 추가
        df_out['new_date'] = df_out['reg_date']

        # 10분 단위로 시간을 바꾸기 위해서 df_out 각각의 행을 6개로 복사
        for j in range(0, 5):
            for i in range(0, 9):
                df_out = df_out.append(df_out.loc[i], ignore_index=True)

        # 시간 순으로 정렬
        df_out = df_out.sort_values(by='new_date', ascending=True).reset_index(drop=True)

        # 1시간 단위로 복사한 행을 10분 단위로 수정
        for i in range(len(df_out)):
            # 정각 시간은 제외
            if i % 6 == 0:
                continue
            # 전 행에 10분을 더해줌
            else:
                df_out['new_date'].iloc[i] = df_out['new_date'].iloc[i-1] + timedelta(minutes=10)

        result = pd.merge(df_in, df_con, on='reg_date')
        result = pd.merge(result, df_out, left_on='reg_date', right_on='new_date')
        return result

    # 결측치 처리
    def process_missing_val(self, df):
        logger.debug(inspect.currentframe().f_code.co_name + " start")

        df.drop(df[['reg_date_x', 'reg_date_y']], axis=1, inplace=True)

        df = df[['new_date', 'fclty_nm', 'crop_nm', 'tp_x', 'hd', 'co2', 
                 'tp_y', 'sr', 'wd', 'ws', 'pc', 
                 'sky_window', 'cg_curtain', 'flow_fan', 'hmdfc', 'crc_pump', '3way_valve','heat_cooler']]

        #hd-> hd_x로 변경
        df.rename(columns = {'hd':'hd_x'},inplace = True)

        # 결측치 처리
        df = df.replace('-', np.NaN).replace('\\N', np.NaN).replace('', np.NaN)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)

        df[['tp_x', 'hd_x','co2','tp_y','sr','wd','ws','pc']] = df[['tp_x','hd_x','co2','tp_y','sr','wd','ws','pc']].astype(float)
        df = df.replace({'wd':'서'}, 270)   # 이상치 처리
        df = df.replace({'wd':'동북'}, 45)
        df = df.replace({'wd':'동남'}, 135)
        df = df.replace({'wd':'동'}, 90)

        df = df.set_index('new_date')

        return df

    # 예측하기
    def pridict_model(self, df):
        logger.debug("["+str(os.getpid()) + "]"+inspect.currentframe().f_code.co_name + " start")
        
        
        predicted_value = []
        try:
            for i in Config.custom_x[crop_nm_indicator(self)].keys():
                
                tensorflow.random.set_seed(2022)
                var = []
                var.clear()
                var = Config.custom_x[crop_nm_indicator(self)][i].copy()
                var.append(i)
        
                
                Pred_data = scaling_make_dataset(self,df,var,i)
                

                #모델 불러오기
                GH_optimization_model = tensorflow.keras.models.load_model('/home/seoul/joseokjun/smartfarm/smartfarm_ver2/smartfarm_ver2_code/model_scaler/sangu_'+crop_nm_indicator(self)+'_model/sangu_'+crop_nm_indicator(self)+'_model_'+i+'.h5')

                # 예측
                
                output = pd.DataFrame(GH_optimization_model.predict(Pred_data).reshape(-1,1))
                output  = value_standard(i,output[0][0])
                predicted_value.append(output)
                # 예측한 날짜
                prd_date = df.reset_index()['new_date'][47] + timedelta(minutes=10)
            
                
            
            return predicted_value, prd_date

        except Exception as e:
            print(traceback.format_exc())
            logger.error("예측하기 오류 발생")
            return

    # 결과 csv 파일로 내보내기
    # def export_csv(self, predicted_value, prd_date):
    #     logger.debug("["+str(os.getpid()) + "]"+inspect.currentframe().f_code.co_name + " start")
    #     print(predicted_value)
    #     # 예측할 값을 넣을 데이터 프레임
    #     GH_env_monitor = pd.DataFrame(
    #         columns=['sn', 'fclty_nm', 'crop_nm', 'sky_window', 'side_window', 'double_window', 'cg_curtain',
    #                  'kw_curtain', 'side_curtain', 'out_curtain','flow_fan', 'heater', 'hmdfc', 'light_beam', 
    #                  'cooler', 'crc_ws', 'co2_occur', 'crc_pump','kp_ctrl', '3way_valve', 'exhst_fan',
    #                  'vent', 'kp_sttus', 'pump_sttus', 'hd_manage', 'heat_cooler', 'fumi_gator', 'co2_heater',
    #                  'nutr_sys', 'reg_date', 'prd_date'])

    #     GH_env_monitor.loc[0] = [0,   self.fclty_nm, self.crop_nm, predicted_value[0], 0,   0,                  predicted_value[1],
    #                              0,   0,             0,            predicted_value[2], 0,   predicted_value[3], 0, 
    #                              0,   0,             0,            predicted_value[4], 0,   predicted_value[6], 0,
    #                              0,   0,             0,            0,                  0,   0,                  0, 
    #                              0,   np.NaN,        prd_date]
    #     try:
    #         result_path = './result/' + self.fclty_nm + "_" + self.crop_nm + '_온실모니터링_10분뒤_예측.csv'
    #         GH_env_monitor.to_csv(result_path, encoding='utf-8-sig')
    #         logger.info("결과 파일 저장 완료 - " + result_path)
    #     except:
    #         logger.error(traceback.format_exc())
    #         logger.error("결과 파일 저장 오류 발생 - " + result_path)

    # 결과 DB 저장
    def insert_into_db(self, predicted_value, prd_date):
        try:
            cursor = db_connect().cursor()
            query = """
                        INSERT INTO smartfarm.gh_indoor_mal_info_predict_test(fclty_nm, crop_nm, sky_window, cg_curtain, flow_fan, hmdfc, crc_pump, 3way_valve, heat_cooler, prd_date)
                        VALUES('{}', '{}', {}, {}, {}, {}, {}, {}, {}, '{}')
                    """.format(self.fclty_nm, self.crop_nm, predicted_value[0], predicted_value[1], predicted_value[2], predicted_value[3], predicted_value[4], predicted_value[5], predicted_value[6], prd_date)

            cursor.execute(query)
            cursor.close()
            logger.info(self.fclty_nm + "_" + self.crop_nm + " - 결과 DB 저장 완료")
        except:
            logger.error(traceback.format_exc())
            logger.error(self.fclty_nm + "_" + self.crop_nm + " - 결과 DB 저장 오류 발생")

    # 모델 실행
    def run(self):
        try:
            logger.info("["+str(os.getpid()) + "]GH Model 실행 : " + self.fclty_nm + "_" + self.crop_nm + ":")

            df_in, df_out, df_con = GHModel.get_data(self)

            df_in, df_out, df_con = GHModel.refine_data(self, df_in, df_out, df_con)

            df_in = GHModel.preprocess_in(self, df_in)
            if df_in is None or df_in.empty: raise Exception("데이터 전처리 및 병합 오류 발생")

            df_con = GHModel.preprocess_con(self, df_con)
            if df_con is None or df_con.empty: raise Exception("데이터 전처리 및 병합 오류 발생")

            df = GHModel.preprocess_out(self, df_in,df_con,df_out)
            if df is None or df.empty: raise Exception("데이터 전처리 및 병합 오류 발생")
            
            df = GHModel.process_missing_val(self, df)
            if df is None: raise Exception("결측치 제거시 오류 발생")

            predicted_value, prd_date = GHModel.pridict_model(self, df)
            if predicted_value is None: raise Exception("예측값 None")

            # GHModel.export_csv(self, predicted_value, prd_date)

            GHModel.insert_into_db(self, predicted_value, prd_date)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("* 모듈 실행중 오류 발생 : " + str(e))
        return

def execute_model(model):
    obj = GHModel(model)
    obj.run()

if __name__ == "__main__":
    logger.info("GH optimization model : Start")

    model_list = [["경영형-01","딸기"],
                  ["경영형-02","토마토"],
                  ["경영형-03","멜론"],
                  ["경영형-04","만다린"],
                  ["교육형-01","딸기"],
                  ["교육형-02", "토마토"],
                  ["교육형-03", "멜론"],
                  ["교육형-04", "만다린"],
                  ["임대형1-01", "딸기"],
                  ["임대형1-02", "딸기"],
                  ["임대형1-03", "딸기"],
                  ["임대형1-04", "딸기"]]


    start_time = datetime.now()

    p = Pool(1)
    p.map(execute_model, model_list)
    p.close()   # 프로세스 관련 자원 해제
    p.join()    # 프로세스 종료 대기

    end_time = datetime.now()
    logger.info('** 모듈 실행 완료 > 총 실행시간 : ' + str(end_time-start_time))
