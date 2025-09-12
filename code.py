import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df_raw=pd.read_csv("raw_total_data.csv")

df_raw.info()
# =============================================================================
#   #   Column                        Non-Null Count   Dtype         
# ---  ------                        --------------   -----         
#   0   Timestamp                     579297 non-null  datetime64[ns]
#   1   OUTPUT_COUNT_DAY_1            472200 non-null  float64       
#   2   KO6_MOTOR_SET_FREQ            31 non-null      float64       
#   3   CUTTING_SET_FREQ              30 non-null      float64       
#   4   STATUS                        138 non-null     float64       
#   5   KO5_MOTOR_SET_FREQ            31 non-null      float64       
#   6   METAL_OIL_SUPPLY_PRESS_CONTR  577321 non-null  float64       
#   7   KO4_MOTOR_SET_FREQ            31 non-null      float64       
#   8   KO2_MOTOR_SET_FREQ            31 non-null      float64       
#   9   MAIN_MOTOR_CURR               553059 non-null  float64       
#   10  KO3_MOTOR_SET_FREQ            31 non-null      float64       
#   11  TRANS_POS_UP_SET_H            31 non-null      float64       
#   12  TRANS_POS_RIGHT_SET_L         31 non-null      float64       
#   13  TONGS_INVERTER_ALM_ERR_CD     30 non-null      float64       
#   14  KO1_MOTOR_SET_FREQ            31 non-null      float64       
#   15  KO3_MOTOR_INVERTER_ALM        31 non-null      float64       
#   16  MAIN_MOTOR_RPM                287 non-null     float64       
#   17  TRANS_CURR                    30 non-null      float64       
#   18  KO1_MOTOR_CURR                31 non-null      float64       
#   19  TRANS_INVERTER_ALM_ERR_CD     30 non-null      float64       
#   20  TONGS_CAST_SET_FREQ           68 non-null      float64       
#   21  TRANS_POS_LEFT_SET_H          31 non-null      float64       
#   22  KO4_MOTOR_INVERTER_ALM        31 non-null      float64       
#   23  TRANS_POS_DOWN_SET_L          31 non-null      float64       
#   24  KO6_MOTOR_CURR                31 non-null      float64       
#   25  OIL_SUPPLY_PRESS              577536 non-null  float64       
#   26  KO2_MOTOR_INVERTER_ALM        31 non-null      float64       
#   27  KO3_MOTOR_CURR                31 non-null      float64       
#   28  TRANS_POS_UP                  498646 non-null  float64       
#   29  TONGS_POS                     48 non-null      float64       
#   30  WORK_OIL_SUPPLY_PRESS         519487 non-null  float64       
#   31  METAL_TEMP_CONTROL            2162 non-null    float64       
#   32  TONGS_CAST_CURR               541 non-null     float64       
#   33  CUTTING_INVERTER_ALM_ERR_CD   32 non-null      float64       
#   34  KO6_MOTOR_INVERTER_ALM        31 non-null      float64       
#   35  TRANS_POS_RIGHT_SET_H         31 non-null      float64       
#   36  TRANS_POS_UP_SET_L            31 non-null      float64       
#   37  TRANS_POS_LEFT                503450 non-null  float64       
#   38  KO4_MOTOR_CURR                31 non-null      float64       
#   39  METAL_OIL_SUPPLY_PRESS_CUT    577220 non-null  float64       
#   40  MAIN_AIR_PRESS                579297 non-null  float64       
#   41  TRANS_POS_LEFT_SET_L          31 non-null      float64       
#   42  TRANS_SET_FREQ                30 non-null      float64       
#   43  METAL_TEMP_CUT                2145 non-null    float64       
#   44  KO5_MOTOR_INVERTER_ALM        31 non-null      float64       
#   45  MAIN_MOTOR_SET_FREQ           30 non-null      float64       
#   46  OIL_PRESS_LEVEL_ALM           30 non-null      float64       
#   47  CUTTING_CURR                  30 non-null      float64       
#   48  KO5_MOTOR_CURR                31 non-null      float64       
#   49  KO2_MOTOR_CURR                31 non-null      float64       
#   50  KO1_MOTOR_INVERTER_ALM        33 non-null      float64       
#   51  TRANS_POS_DOWN_SET_H          31 non-null      float64       
#   52  OUTPUT_COUNT_DAY_2            472194 non-null  float64       
#   53  OUTPUT_COUNT_SUM              11640 non-null   float64       
#   54  TRANS_POS_DOWN                492863 non-null  float64       
#   55  TRANS_POS_RIGHT               546188 non-null  float64       
#   56  MAIN_MOTOR_ALM                32 non-null      float64       
# =============================================================================

df_raw.shape # (579297, 57)

# 측정 시각 파악
df_raw['Timestamp'].head()
# 0    2022-05-02 06:32:33 
df_raw['Timestamp'].tail()
# 579296    2022-05-14 04:34:46 

# 불필요한 index, column 제거
df_raw.drop(df_raw.columns[0], axis=1,inplace=True)
df_raw.columns = [col.replace('.xlsx', '') for col in df_raw.columns]
df_raw.columns.sort_values()
# =============================================================================
# Index(['CUTTING_CURR', 'CUTTING_INVERTER_ALM_ERR_CD', 'CUTTING_SET_FREQ',
#        'KO1_MOTOR_CURR', 'KO1_MOTOR_INVERTER_ALM', 'KO1_MOTOR_SET_FREQ',
#        'KO2_MOTOR_CURR', 'KO2_MOTOR_INVERTER_ALM', 'KO2_MOTOR_SET_FREQ',
#        'KO3_MOTOR_CURR', 'KO3_MOTOR_INVERTER_ALM', 'KO3_MOTOR_SET_FREQ',
#        'KO4_MOTOR_CURR', 'KO4_MOTOR_INVERTER_ALM', 'KO4_MOTOR_SET_FREQ',
#        'KO5_MOTOR_CURR', 'KO5_MOTOR_INVERTER_ALM', 'KO5_MOTOR_SET_FREQ',
#        'KO6_MOTOR_CURR', 'KO6_MOTOR_INVERTER_ALM', 'KO6_MOTOR_SET_FREQ',
#        'MAIN_AIR_PRESS', 'MAIN_MOTOR_ALM', 'MAIN_MOTOR_CURR', 'MAIN_MOTOR_RPM',
#        'MAIN_MOTOR_SET_FREQ', 'METAL_OIL_SUPPLY_PRESS_CONTR',
#        'METAL_OIL_SUPPLY_PRESS_CUT', 'METAL_TEMP_CONTROL', 'METAL_TEMP_CUT',
#        'OIL_PRESS_LEVEL_ALM', 'OIL_SUPPLY_PRESS', 'OUTPUT_COUNT_DAY_1',
#        'OUTPUT_COUNT_DAY_2', 'OUTPUT_COUNT_SUM', 'STATUS', 'TONGS_CAST_CURR',
#        'TONGS_CAST_SET_FREQ', 'TONGS_INVERTER_ALM_ERR_CD', 'TONGS_POS',
#        'TRANS_CURR', 'TRANS_INVERTER_ALM_ERR_CD', 'TRANS_POS_DOWN',
#        'TRANS_POS_DOWN_SET_H', 'TRANS_POS_DOWN_SET_L', 'TRANS_POS_LEFT',
#        'TRANS_POS_LEFT_SET_H', 'TRANS_POS_LEFT_SET_L', 'TRANS_POS_RIGHT',
#        'TRANS_POS_RIGHT_SET_H', 'TRANS_POS_RIGHT_SET_L', 'TRANS_POS_UP',
#        'TRANS_POS_UP_SET_H', 'TRANS_POS_UP_SET_L', 'TRANS_SET_FREQ',
#        'Timestamp', 'WORK_OIL_SUPPLY_PRESS'],
#       dtype='object')
# =============================================================================


# 빈 시간 체크하고자 처음부터 끝까지 매초가 index로 존재하는 full_time_df 생성
full_time_df=df_raw.copy(deep=True)
full_time_df['Timestamp'] = pd.to_datetime(full_time_df['Timestamp'])
full_time_df.set_index('Timestamp', inplace=True)
# full_range shape이 (1029734,)이고
# df_raw shape이 (579297,_)이므로 없는 게 조금 더 적음
full_range = pd.date_range\
    (start=full_time_df.index.min(), end=full_time_df.index.max(), freq='S')
full_time_df= full_time_df.reindex(full_range)
# raw data에 없던 시간 넣어본 결과, 행이 생겼고 모두 NaN으로 돼있음
full_time_df.loc['2022-05-13 19:23:30']

# index로 돼있는 Timestamp를 다시 컬럼으로
full_time_df_idx = full_time_df.reset_index().rename\
    (columns={'index': 'Timestamp'})
full_time_df_idx['Timestamp'] = full_time_df_idx['Timestamp'].astype(str)

# Timestamp 제외 row 모든 값이 nan인지
is_all_nan=full_time_df_idx.drop(columns='Timestamp').isna().all(axis=1)

# 시간순 '분별로 data 몇 개 뽑혔는지' 그래프
all_timestamps_min = set(full_time_df_idx.Timestamp.str[:-3])
full_time_df_idx.loc[~is_all_nan].Timestamp.str[:-3].value_counts().reindex\
    (all_timestamps_min, fill_value=0).sort_index().plot(style='o',ms=3)
# 매 초마다 데이터가 있지 않음
# 작업 정지 시간이 일정하지 않음
# 중간에 하루 이상을 중단한 시기도 존재



# 전처리 시작

# 자료형 관련
# Timestamp 제외 모두 float64. Timestamp 자료형 datetime으로 만들고 index로. 
# float64들 경우, target인 status 제외 모두 범주형 아닌 수치형이므로 그대로
df_raw.info()
# 0   Timestamp                     579297 non-null  object 
df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'])
df_raw.info()
# 0   Timestamp                     579297 non-null  datetime64[ns]

# 표준편차 0 columns 삭제
df_raw_std=df_raw.drop(columns=['Timestamp']).std()
list(df_raw_std[df_raw_std==0].index)
# =============================================================================
# ['TRANS_POS_RIGHT_SET_L',
#  'TRANS_CURR',
#  'KO1_MOTOR_CURR',
#  'TRANS_POS_DOWN_SET_L',
#  'KO6_MOTOR_CURR',
#  'KO3_MOTOR_CURR',
#  'TRANS_POS_UP_SET_L',
#  'KO4_MOTOR_CURR',
#  'TRANS_POS_LEFT_SET_L',
#  'OIL_PRESS_LEVEL_ALM',
#  'CUTTING_CURR',
#  'KO5_MOTOR_CURR',
#  'KO2_MOTOR_CURR']
# =============================================================================
# 알람은 표준편차가 0이더라도
# 그 타이밍에 어떤 문제가 발생했다는 것을 알려주는 기능을 하니 의미가 있을 수도 있음
# 울렸을 때 status를 확인하고자 함

error_indices = df_raw[df_raw['OIL_PRESS_LEVEL_ALM'].notna()].index
# 정상이거나 정상에 가까운 때 울린 알람이 전부
df_raw['STATUS'].loc[error_indices].value_counts()
# STATUS 2.0    12, 0.0    18

# 혹시 좀 시간이 지나면서 안 좋아지는 걸 수도 있으니 어떻게 되나 보면
# 40 index 뒤까지도 동일
df_raw['STATUS'].ffill().iloc[error_indices + 40].value_counts()
# 이때쯤부터 변함 STATUS 2.0 13, 0.0 17
df_raw['STATUS'].ffill().iloc[error_indices + 50].value_counts()
# STATUS 2.0 20, 0.0 10
df_raw['STATUS'].ffill().iloc[error_indices + 100].value_counts()
# 점점 정상이 많아지는 식으로, 알람이 무의미하다는 걸 알 수 있음
# 제거해도 되겠음


df_raw_pre1 = df_raw.drop(columns=list(df_raw_std[df_raw_std==0].index))
df_raw_pre1.shape # (579297, 44)

df_raw_pre1.count().sort_values()
# =============================================================================
# CUTTING_SET_FREQ                    30
# MAIN_MOTOR_SET_FREQ                 30
# TRANS_INVERTER_ALM_ERR_CD           30
# TRANS_SET_FREQ                      30
# TONGS_INVERTER_ALM_ERR_CD           30
# KO2_MOTOR_INVERTER_ALM              31
# TRANS_POS_LEFT_SET_H                31
# KO6_MOTOR_INVERTER_ALM              31
# TRANS_POS_RIGHT_SET_H               31
# KO3_MOTOR_INVERTER_ALM              31
# KO1_MOTOR_SET_FREQ                  31
# KO3_MOTOR_SET_FREQ                  31
# KO4_MOTOR_INVERTER_ALM              31
# KO2_MOTOR_SET_FREQ                  31
# KO4_MOTOR_SET_FREQ                  31
# KO5_MOTOR_INVERTER_ALM              31
# KO5_MOTOR_SET_FREQ                  31
# KO6_MOTOR_SET_FREQ                  31
# TRANS_POS_DOWN_SET_H                31
# TRANS_POS_UP_SET_H                  31
# CUTTING_INVERTER_ALM_ERR_CD         32
# MAIN_MOTOR_ALM                      32
# KO1_MOTOR_INVERTER_ALM              33
# TONGS_POS                           48
# TONGS_CAST_SET_FREQ                 68
# STATUS                             138
# MAIN_MOTOR_RPM                     287
# TONGS_CAST_CURR                    541
# METAL_TEMP_CUT                    2145
# METAL_TEMP_CONTROL                2162
# OUTPUT_COUNT_SUM                 11640
# OUTPUT_COUNT_DAY_2              472194
# OUTPUT_COUNT_DAY_1              472200
# TRANS_POS_DOWN                  492863
# TRANS_POS_UP                    498646
# TRANS_POS_LEFT                  503450
# WORK_OIL_SUPPLY_PRESS           519487
# TRANS_POS_RIGHT                 546188
# MAIN_MOTOR_CURR                 553059
# METAL_OIL_SUPPLY_PRESS_CUT      577220
# METAL_OIL_SUPPLY_PRESS_CONTR    577321
# OIL_SUPPLY_PRESS                577536
# MAIN_AIR_PRESS                  579297
# Timestamp                       579297
# =============================================================================


# 같은 것들 하나만 남김
cols= list(df_raw_pre1.drop(columns=['Timestamp']).columns.sort_values())
# 같은 컬럼들 저장할 dict. 
# 컬럼 a,d,e,f가 같다면, key a에 대해 [d,e,f]가 value로 들어오고
# key d,e,f는 존재하지 않도록 해 중복 없음
same_things=dict()
no_need_to_do=set()
for i in range(len(cols)):
    # 이미 고려된 컬럼 돌지 않게 해 key 중복 안 되게 함
    if cols[i] not in no_need_to_do:
         for j in range(i + 1, len(cols)):
        # nan==nan 을 false로 간주하므로 둘 다 nan일 때를 true로 보는 식을 덧붙임
            if ((df_raw_pre1[cols[i]]==df_raw_pre1[cols[j]]) |\
                (df_raw_pre1[cols[i]].apply(math.isnan) &\
                 df_raw_pre1[cols[j]].apply(math.isnan)) ).all():
                if cols[i] not in same_things.keys():
                    same_things[cols[i]]=[cols[j]]
                else:
                    same_things[cols[i]].append(cols[j])
                # 이미 고려됐으니 돌 필요 없는 컬럼으로 추가
                no_need_to_do.add(cols[j])
same_things 
# =============================================================================
# {'CUTTING_SET_FREQ': ['TRANS_SET_FREQ'],
#  'KO1_MOTOR_SET_FREQ': ['KO2_MOTOR_SET_FREQ',
#   'KO3_MOTOR_SET_FREQ',
#   'KO4_MOTOR_SET_FREQ',
#   'KO5_MOTOR_SET_FREQ',
#   'KO6_MOTOR_SET_FREQ'],
#  'KO2_MOTOR_INVERTER_ALM': ['KO3_MOTOR_INVERTER_ALM',
#   'KO4_MOTOR_INVERTER_ALM',
#   'KO5_MOTOR_INVERTER_ALM',
#   'KO6_MOTOR_INVERTER_ALM'],
#  'TONGS_INVERTER_ALM_ERR_CD': ['TRANS_INVERTER_ALM_ERR_CD'],
#  'TRANS_POS_DOWN_SET_H': ['TRANS_POS_LEFT_SET_H',
#   'TRANS_POS_RIGHT_SET_H',
#   'TRANS_POS_UP_SET_H']}
# =============================================================================

# 정리한 same_things 이용해 중복열 삭제
df_raw_pre2=df_raw_pre1.copy(deep=True)
for col in same_things.keys():
    df_raw_pre2.drop(columns=same_things[col],inplace=True)
df_raw_pre2.shape

df_raw_pre2.count().sort_values()
# =============================================================================
# CUTTING_SET_FREQ                    30
# TONGS_INVERTER_ALM_ERR_CD           30
# MAIN_MOTOR_SET_FREQ                 30
# KO1_MOTOR_SET_FREQ                  31
# KO2_MOTOR_INVERTER_ALM              31
# TRANS_POS_DOWN_SET_H                31
# MAIN_MOTOR_ALM                      32
# CUTTING_INVERTER_ALM_ERR_CD         32
# KO1_MOTOR_INVERTER_ALM              33
# TONGS_POS                           48
# TONGS_CAST_SET_FREQ                 68
# STATUS                             138
# MAIN_MOTOR_RPM                     287
# TONGS_CAST_CURR                    541
# METAL_TEMP_CUT                    2145
# METAL_TEMP_CONTROL                2162
# OUTPUT_COUNT_SUM                 11640
# OUTPUT_COUNT_DAY_2              472194
# OUTPUT_COUNT_DAY_1              472200
# TRANS_POS_DOWN                  492863
# TRANS_POS_UP                    498646
# TRANS_POS_LEFT                  503450
# WORK_OIL_SUPPLY_PRESS           519487
# TRANS_POS_RIGHT                 546188
# MAIN_MOTOR_CURR                 553059
# METAL_OIL_SUPPLY_PRESS_CUT      577220
# METAL_OIL_SUPPLY_PRESS_CONTR    577321
# OIL_SUPPLY_PRESS                577536
# MAIN_AIR_PRESS                  579297
# Timestamp                       579297
# =============================================================================

# 결측치가 모두 존재하지 않는 행의 개수 파악
rows_without_nan = df_raw.dropna()
len(rows_without_nan) # 30
# 즉, count 30인 컬럼들은 모든 값 찍힐 때만 찍혔다!

# count 30들 결과를, 인덱스와 함께 보면 
# 0 한 번 나왔다가, 또는 바로 0이 아닌 수치가 나옴. 그리고 그 타이밍이 셋 다 동일
# 그리고 절반값 나오는 타이밍도 셋 다 동일
df_raw[['MAIN_MOTOR_SET_FREQ','CUTTING_SET_FREQ',\
        'TONGS_INVERTER_ALM_ERR_CD']][df_raw['MAIN_MOTOR_SET_FREQ'].notna()]
# =============================================================================
# index       MAIN_MOTOR_SET_FREQ  CUTTING_SET_FREQ  TONGS_INVERTER_ALM_ERR_CD
# 0                    2166.5            3000.0                     4497.5

# 79758                2166.5            3000.0                     4497.5

# 126127                  0.0               0.0                        0.0
# 126128               4333.0            6000.0                     8995.0

# 205011                  0.0               0.0                        0.0
# 205012               4333.0            6000.0                     8995.0

# 246466                  0.0               0.0                        0.0
# 246467               4333.0            6000.0                     8995.0

# 319921               4333.0            6000.0                     8995.0
# 320009               4333.0            6000.0                     8995.0
# 321756               4333.0            6000.0                     8995.0
# 323561               4333.0            6000.0                     8995.0

# 324760                  0.0               0.0                        0.0
# 324761               4333.0            6000.0                     8995.0

# 370640                  0.0               0.0                        0.0
# 370641               4333.0            6000.0                     8995.0

# 395549               2166.5            3000.0                     4497.5

# 416839                  0.0               0.0                        0.0
# 416840               4333.0            6000.0                     8995.0

# 425643               4333.0            6000.0                     8995.0
# 428126               4333.0            6000.0                     8995.0
# 429378               4333.0            6000.0                     8995.0
# 447627               4333.0            6000.0                     8995.0
# 452240               4333.0            6000.0                     8995.0

# 463203                  0.0               0.0                        0.0
# 463204               4333.0            6000.0                     8995.0

# 535043               4333.0            6000.0                     8995.0
# 561163               4333.0            6000.0                     8995.0
# 564082               4333.0            6000.0                     8995.0
# 573012               4333.0            6000.0                     8995.0
# =============================================================================


# 모든 값 찍힐 때가 붙어있는 때도 있는 걸 보고, 모든 값 찍힐 때의 전후 시간을 확인해봤음


# 모든 값 찍힐 때 전후 시간
# 다 찍힐 때가 바로 다음 초에 이어서 나왔을 땐 한 번만 출력하도록 작성함
prev_idx = 0
for idx in rows_without_nan.index:
    if idx==0:
        print(df_raw.iloc[idx:idx+2,0])
    if idx==df_raw.index[-1]:
        print(df_raw.iloc[idx-1:idx+1,0])
    if idx-prev_idx>2:
        print(df_raw.iloc[idx-1:idx+2,0])
    prev_idx = idx
# 아래 들어간 부분들: 전과의 간격이 시간 단위 
# 아래 튀어나오게 표시한 부분들: 전과의 간격이 분, 초 단위
# =============================================================================
# 0   2022-05-02 06:32:33
# 1   2022-05-02 06:32:34
# Name: Timestamp, dtype: datetime64[ns]
# 79757   2022-05-03 04:43:32
# 79758   2022-05-03 06:27:02
# 79759   2022-05-03 06:27:03
# Name: Timestamp, dtype: datetime64[ns]
# 126126   2022-05-03 19:22:48
# 126127   2022-05-04 06:33:18
# 126128   2022-05-04 06:33:19
# Name: Timestamp, dtype: datetime64[ns]
# 205010   2022-05-05 04:32:45
# 205011   2022-05-05 16:59:24
# 205012   2022-05-05 16:59:25
# Name: Timestamp, dtype: datetime64[ns]
# 246465   2022-05-06 04:31:53
# 246466   2022-05-06 06:32:53
# 246467   2022-05-06 06:32:54
# Name: Timestamp, dtype: datetime64[ns]
                        # 319920   2022-05-07 03:01:24
                        # 319921   2022-05-07 03:01:39
                        # 319922   2022-05-07 03:01:40
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 320008   2022-05-07 03:03:06
                        # 320009   2022-05-07 03:03:21
                        # 320010   2022-05-07 03:03:22
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 321755   2022-05-07 03:32:55
                        # 321756   2022-05-07 03:33:08
                        # 321757   2022-05-07 03:33:09
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 323560   2022-05-07 04:04:05
                        # 323561   2022-05-07 04:04:20
                        # 323562   2022-05-07 04:04:21
                        # Name: Timestamp, dtype: datetime64[ns]
# 324759   2022-05-07 04:24:51
# 324760   2022-05-09 06:31:30
# 324761   2022-05-09 06:31:31
# Name: Timestamp, dtype: datetime64[ns]
# 370639   2022-05-09 19:22:10
# 370640   2022-05-10 06:30:37
# 370641   2022-05-10 06:30:38
# Name: Timestamp, dtype: datetime64[ns]
                        # 395548   2022-05-10 13:26:54
                        # 395549   2022-05-10 13:28:13
                        # 395550   2022-05-10 13:28:14
                        # Name: Timestamp, dtype: datetime64[ns]
# 416838   2022-05-10 19:25:08
# 416839   2022-05-11 06:34:40
# 416840   2022-05-11 06:34:41
# Name: Timestamp, dtype: datetime64[ns]
                        # 425642   2022-05-11 09:02:44
                        # 425643   2022-05-11 09:02:58
                        # 425644   2022-05-11 09:02:59
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 428125   2022-05-11 09:46:05
                        # 428126   2022-05-11 09:46:28
                        # 428127   2022-05-11 09:46:29
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 429377   2022-05-11 10:07:37
                        # 429378   2022-05-11 10:07:50
                        # 429379   2022-05-11 10:07:51
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 447626   2022-05-11 15:17:38
                        # 447627   2022-05-11 15:17:53
                        # 447628   2022-05-11 15:17:54
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 452239   2022-05-11 16:36:25
                        # 452240   2022-05-11 16:36:40
                        # 452241   2022-05-11 16:36:41
# Name: Timestamp, dtype: datetime64[ns]
# 463202   2022-05-11 19:40:51
# 463203   2022-05-12 06:38:00
# 463204   2022-05-12 06:38:01
# Name: Timestamp, dtype: datetime64[ns]
                        # 535042   2022-05-13 02:41:04
                        # 535043   2022-05-13 02:41:18
                        # 535044   2022-05-13 02:41:19
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 561162   2022-05-13 19:23:28
                        # 561163   2022-05-13 19:23:46
                        # 561164   2022-05-13 19:23:47
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 564081   2022-05-13 21:02:48
                        # 564082   2022-05-13 21:03:16
                        # 564083   2022-05-13 21:03:19
                        # Name: Timestamp, dtype: datetime64[ns]
                        # 573011   2022-05-14 00:40:16
                        # 573012   2022-05-14 00:40:21
                        # 573013   2022-05-14 00:40:23
                        # Name: Timestamp, dtype: datetime64[ns]
# =============================================================================


# 위에서 구한 두 결과(count 30들 값과 모든 값 찍힐 때 전후 시간)를 함께 보면, 재밌는 규칙을 찾게 됨
# 전과의 간격 분 단위일 때는 바로 정상적으로 됨 0 한 번 나오는 적 한 번도 없음
# 또한 시간 단위, 분 단위 상관없이 중간 주파수로 설정시, 시간 단위여도 0 안 나옴
# 따라서 주파수 높게 설정하고 간격 시간 단위 때만 0 나옴. 


# 위 세 경우들에서 다른 컬럼들도 차이를 보이는지 그래프로 확인해보고자 함

# count 30이
# 간격 시간 단위시 높게 켜서 0 나온 때
# 주파수 반으로 설정한 때
# 그 외
# 세 경우에 대해서
# 그래프를 그리는 컬럼의 시계열 데이터와의 교점과 그 위치에서의 수직선을 서로 다른 세 가지 색으로 나타냄


# 각 케이스들 인덱스 정리

# 0 나올 때가, 간격 시간 단위고 high_freq일 때. 다음 초까지 연달아 나옴
# 2차원으로 만든 list 1차원으로 만들기 위해
# ravel 함수 쓰고자 np.array로 잠깐 바꿨다가, set으로
cases_work_start_high_f=set(np.array([[idx,idx+1]\
                                      for idx in df_raw['MAIN_MOTOR_SET_FREQ']\
                                          [df_raw['MAIN_MOTOR_SET_FREQ']==0]\
                                              .index]).ravel())
# 2166.5로 0과 4333의 중간 주파수로 설정할 때
cases_low_freq=set(df_raw['MAIN_MOTOR_SET_FREQ']\
                   [df_raw['MAIN_MOTOR_SET_FREQ']==2166.5].index)
# 차집합으로 남는 경우 구함
cases_else=set(df_raw['MAIN_MOTOR_SET_FREQ']\
               [df_raw['MAIN_MOTOR_SET_FREQ'].notna()].index)\
    -cases_work_start_high_f-cases_low_freq

# 세 종류 케이스들이 가질 수직선,교점의 색
colors={'cases_work_start_high_f':'red','cases_low_freq'\
        :'green','cases_else':'blue'}


# index가 위 세 케이스 종류 중 어디 속하는지 판단 함수. 아래 쓰임 
def find_case(idx):
    if idx in cases_work_start_high_f:
        return 'cases_work_start_high_f'
    elif idx in cases_low_freq:
        return 'cases_low_freq'
    else:
        return 'cases_else'
    

for col in df_raw_pre2.drop(columns=['Timestamp']).columns.sort_values():
    _, ax= plt.subplots()
    ax.plot(df_raw_pre2.index, df_raw_pre2[col], 'o', ms=3)
    ax.set_title(col)
    for idx in df_raw['MAIN_MOTOR_SET_FREQ'][df_raw['MAIN_MOTOR_SET_FREQ']\
                                             .notna()].index:
        if not math.isnan(df_raw_pre2[col][idx]):
            case=find_case(idx)
            plt.axvline(x=idx, color=colors[case], linestyle='-',linewidth=0.5)
            ax.plot(idx, df_raw_pre2[col][idx], 'o', markerfacecolor='none',\
                    markeredgewidth=0.7, markersize=6, color=colors[case])
    plt.show()
  
# 나온 그래프들을 해석해보자 (아래 설명들을 보고서에 쓰실 때 그래프들도 첨부해주세요)

# 빨간색의 경우, 모든 컬럼에서 붙어있는 두 값 중 첫 번째 값에는 0으로 나왔고, 두 번째 값에는 추세에 맞는 데이터들이 나왔음
# 따라서 붙어있는 두 값 중 첫 번째에 해당하는 인덱스의 row들만 제거

# 초록색의 경우,
# output count day1/day2/sum, metal temp cut/control 반토막 나서 추세와 맞지 않게 단 한 번 이상하게 나오는 걸로 봐서
# 주파수를 반으로 설정한 게 아니라, 주파수는 똑같은데 0 대신 중간 값이 이상치로 나온 것일 뿐
# 다른 그래프들도 이때 이상치가 나왔음
# 따라서 초록색에 해당하는 행은 제거

# 파란색의 경우, 
# 항상 추세에 맞는 정상적인 데이터가 나왔음. 작업 중 주파수도 정상적으로 찍히는 경우이니 
# 다른 값들도 마찬가지로 정상적으로 찍히는 듯


# 위 정리대로 '빨간색 각 그룹 둘 중 첫번째', 초록색에 해당하는 row들을 삭제
idx_to_drop = cases_low_freq |\
    set([idx for idx in df_raw['MAIN_MOTOR_SET_FREQ']\
         [df_raw['MAIN_MOTOR_SET_FREQ']==0].index])
df_raw_pre3= df_raw_pre2.drop(idx_to_drop)

# 행 제거 후 표준편차 0인 column 만들어짐 -> 제거
df_raw_pre3_std=df_raw_pre3.drop(columns=['Timestamp']).std()
list(df_raw_pre3_std[df_raw_pre3_std==0].index)
# =============================================================================
# ['CUTTING_SET_FREQ',
#  'TONGS_INVERTER_ALM_ERR_CD',
#  'KO1_MOTOR_SET_FREQ',
#  'KO2_MOTOR_INVERTER_ALM',
#  'MAIN_MOTOR_SET_FREQ',
#  'TRANS_POS_DOWN_SET_H']
# =============================================================================
# 그런데 알람은 std 0이여도 그 타이밍에 어떤 문제가 발생했다는 것을 알려주는 기능을 하니
# 의미가 있을 수도 있음
# 울렸을 때 status를 확인하고자 함

error_indices = df_raw_pre3[df_raw_pre3['TONGS_INVERTER_ALM_ERR_CD']\
                            .notna()].index
# STATUS 2.0 13개 0.0 7개.
df_raw_pre3['STATUS'].ffill().loc[error_indices - 40].value_counts()
# STATUS 2.0 12개 0.0 8개. 
# 정상이거나 정상에 가까운 때 울린 알람이 전부. 알람 울리기 전과 상태 변한 건 1개 뿐
df_raw_pre3['STATUS'].loc[error_indices].value_counts()
# 혹시 좀 시간이 지나면서 안 좋아지는 걸 수도 있으니 어떻게 되나 보면
# 40 index 뒤까지도 동일
df_raw_pre3['STATUS'].ffill().loc[error_indices + 40].value_counts()
# 이때쯤부터 변함 STATUS 2.0 13 0.0 7
df_raw_pre3['STATUS'].ffill().loc[error_indices + 50].value_counts()
# STATUS 2.0 15 0.0 5
df_raw_pre3['STATUS'].ffill().loc[error_indices + 100].value_counts()
# 점점 정상이 많아지는 식으로, 알람이 무의미하다는 걸 알 수 있음


# 다른 알람도
error_indices = df_raw_pre3[df_raw_pre3['KO2_MOTOR_INVERTER_ALM']\
                            .notna()].index
# STATUS 2.0 13개 0.0 8개.
(df_raw_pre3['STATUS'].ffill()).loc[error_indices-40].value_counts()
# STATUS 2.0 12 0.0 9. 
# 정상이거나 정상에 가까운 때 울린 알람이 전부. 알람 울리기 전과 상태 변한 건 1개 뿐
(df_raw_pre3['STATUS'].ffill()).loc[error_indices].value_counts()
# 좀 시간이 지나면서 어떻게 되나 보면
# 40 index 뒤까지도 동일
df_raw_pre3['STATUS'].ffill().loc[error_indices + 40].value_counts()
# 이때쯤부터 변함 STATUS 2.0 13 0.0 8
df_raw_pre3['STATUS'].ffill().loc[error_indices + 50].value_counts()
# STATUS 2.0 16 0.0 5
df_raw_pre3['STATUS'].ffill().loc[error_indices + 100].value_counts()
# 점점 정상이 많아지는 식으로, 알람이 무의미하다는 걸 알 수 있음

# 두 알람 다 제거해도 되겠음

df_raw_pre4 = df_raw_pre3.drop(columns=list(df_raw_pre3_std\
                                            [df_raw_pre3_std==0].index))
df_raw_pre4.shape # (579287, 24)
df_raw_pre4.count().sort_values()

# =============================================================================
# MAIN_MOTOR_ALM                      22
# CUTTING_INVERTER_ALM_ERR_CD         22
# KO1_MOTOR_INVERTER_ALM              23
# TONGS_POS                           38
# TONGS_CAST_SET_FREQ                 58
# STATUS                             128
# MAIN_MOTOR_RPM                     277
# TONGS_CAST_CURR                    531
# METAL_TEMP_CUT                    2135
# METAL_TEMP_CONTROL                2152
# OUTPUT_COUNT_SUM                 11630
# OUTPUT_COUNT_DAY_2              472184
# OUTPUT_COUNT_DAY_1              472190
# TRANS_POS_DOWN                  492853
# TRANS_POS_UP                    498636
# TRANS_POS_LEFT                  503440
# WORK_OIL_SUPPLY_PRESS           519477
# TRANS_POS_RIGHT                 546178
# MAIN_MOTOR_CURR                 553049
# METAL_OIL_SUPPLY_PRESS_CUT      577210
# METAL_OIL_SUPPLY_PRESS_CONTR    577311
# OIL_SUPPLY_PRESS                577526
# MAIN_AIR_PRESS                  579287
# Timestamp                       579287
# =============================================================================

# 그래프들 다시 확인
for col in df_raw_pre4.drop(columns=['Timestamp']).columns.sort_values():
    plt.figure()
    df_raw_pre4[col].plot(title=col, style='o',ms=3)
    plt.show()


# 정보성이 있는지 의심가는 컬럼들은 아래와 같음

# 1. TONGS POS
# 공정상 TONGS POS는 계속 변할 것. 
# 그런데 이에 비해 주어진 데이터는 58만개의 데이터 동안 간헐적으로 찍힌 38개 뿐
# 더군다나 왔다갔다할 때 두 곳의 위치일 뿐.
# 데이터가 가치가 없음.
df_raw_pre4['TONGS_POS'].count()
df_raw_pre4['TONGS_POS'].plot(style='o',ms=3)


# 2. main motor rpm
# MAIN_MOTOR_RPM 확대 (선형 증가/감소 형태)
full_time_df['MAIN_MOTOR_RPM']['2022-05-04 07:58:11':'2022-05-04 08:07:13']\
    .plot(style='o',ms=3)
# 위 plot에서 조그만하게 있고 분별 안 가는 부분 확대
# 선형적인 산 형태를 알 수 있음
full_time_df['MAIN_MOTOR_RPM']['2022-05-04 08:04:11':'2022-05-04 08:06:13']\
    .plot(style='o',ms=3)
# 선형적 변화 없는 다른 곳에서는 꾸준히 높은 rpm들로 찍혔음
# 정보성이 있다고 판단됨


# 3. tongs cast curr
# 그래프 형태를 보면
# 특정 시기에 몰려서 찍힘
df_raw_pre4['TONGS_CAST_CURR'].plot(style='o',ms=3)
# 몰리는 시기 파악해서 아래에서 확대된 그래프 그려보고자 0 찍힌 index 찍어봄
df_raw_pre4['TONGS_CAST_CURR'][df_raw_pre4['TONGS_CAST_CURR']==0].index
# =============================================================================
# [126128, 130909, 131776, 205012, 246467, 274964, 274989, 319921, 320009,
#        321756, 323561, 324761, 324811, 370641, 416836, 416840, 416877, 416904,
#        425643, 427956, 428126, 429378, 447627, 452240, 463204, 463540, 518466,
#        535043, 561163, 564082, 573012]
# =============================================================================
# 세부 형태 보면
df_raw_pre4['TONGS_CAST_CURR'][126128-20:126128+5].plot(style='o',ms=3)
df_raw_pre4['TONGS_CAST_CURR'][427330:427400].plot(style='o',ms=3)
df_raw_pre4['TONGS_CAST_CURR'][324761-20:324761+10].plot(style='o',ms=3)
df_raw_pre4['TONGS_CAST_CURR'][130909-20:131776+10].plot(style='o',ms=3)
df_raw_pre4['TONGS_CAST_CURR'][130909-20:130909+10].plot(style='o',ms=3)
df_raw_pre4['TONGS_CAST_CURR'][131776-20:131776+10].plot(style='o',ms=3)
# 어떨 때는 70~80 또는 60~70 선에서 비슷한 값이 10초 정도 이어지고 
# 그 앞뒤로 0 또는 절반의 값이 출력되며 끝나는 형태
# 그러나 확실한 규칙은 없음
# COUNT로 보면 1/1000 상황에서만 값이 나왔음.
# 규칙이 확실하지도 않고 1/1000 상황에서만 값이 나왔기에
# 결측치를 채울 마땅한 방법이 없다고 판단


# 4. 알람들
# 'CUTTING_INVERTER_ALM_ERR_CD', 'KO1_MOTOR_INVERTER_ALM', 'MAIN_MOTOR_ALM'
for col in df_raw_pre4.filter(regex='ALM').columns:
    df_raw_pre4[col].plot(title=col,style='o',ms=3)
    plt.show()
    
# 알람은 그 타이밍에 어떤 문제가 발생했다는 것을 알려주는 기능을 함
# 울린 순간과 전후의 status를 확인 해
# 실제로 문제가 생기는 걸 알려주는 기능을 하는지 확인해볼 것

# 그 전에 세 알람이 비슷해보임
df_raw_pre4.filter(regex='ALM').count()
# =============================================================================
# CUTTING_INVERTER_ALM_ERR_CD    22
# KO1_MOTOR_INVERTER_ALM         23
# MAIN_MOTOR_ALM                 22
# =============================================================================


# count 제일 많은 KO1_MOTOR_INVERTER_ALM 이 notna일 때 세 컬럼 값들 찍어봄 
df_raw_pre4[['CUTTING_INVERTER_ALM_ERR_CD','KO1_MOTOR_INVERTER_ALM'\
             ,'MAIN_MOTOR_ALM']][df_raw_pre4['KO1_MOTOR_INVERTER_ALM'].notna()]
# =============================================================================
#           CUTTING_INVERTER_ALM_ERR_CD  KO1_MOTOR_INVERTER_ALM  MAIN_MOTOR_ALM
# 79759                           NaN                     1.0             NaN
# 126128                          1.0                     1.0             0.0
# 205012                          1.0                     1.0             0.0
# 246467                          1.0                     1.0             0.0
# 275477                       8995.0                  8995.0          2570.0
# 319921                       8995.0                  8995.0          2570.0
# 320009                       8995.0                  8995.0          2570.0
# 321756                       8995.0                  8995.0          2570.0
# 323561                       8995.0                  8995.0          2570.0
# 324761                       8995.0                  8995.0          2570.0
# 333780                          1.0                     1.0             0.0
# 370641                          1.0                     1.0             0.0
# 416840                          1.0                     1.0             0.0
# 425643                          1.0                     1.0             0.0
# 428126                          1.0                     1.0             0.0
# 429378                          1.0                     1.0             0.0
# 447627                          1.0                     1.0             0.0
# 452240                          1.0                     1.0             0.0
# 463204                          1.0                     1.0             0.0
# 535043                          1.0                     1.0             0.0
# 561163                          1.0                     1.0             0.0
# 564082                          1.0                     1.0             0.0
# 573012                          1.0                     1.0             0.0
# =============================================================================
# 세 가지 경우로 나뉨
# 1. KO1_MOTOR_INVERTER_ALM만 찍힌 79759 인덱스
# 2. 모두 큰 값(컬럼따라 8995 또는 2570)이 찍힌 인덱스들
# 3. 모두 작은 값(컬럼따라 0또는 1)이 찍힌 인덱스들

# 이제 각 세 경우에 대해 울린 순간부터 그 이후의 status를 확인해서 
# 실제로 문제가 생기는 걸 알려주는 기능을 하는지 확인해볼 것

# 1. KO1_MOTOR_INVERTER_ALM만 찍힌 79759 인덱스
# 79759 인덱스 전에 이미 status 0이었는데 그 도중에 알람이 울리고
# 조금 지나면 status 2, 즉 정상이 돼 버렸음
(df_raw_pre4['STATUS'].ffill()).loc[79759-40:79759+100]\
    .plot(title=col,style='o',ms=3)
# 따라서 의미가 없음

# 2. 모두 큰 값(컬럼따라 8995 또는 2570)이 찍힌 인덱스들
large_val_idx = df_raw_pre4['KO1_MOTOR_INVERTER_ALM']\
    .loc[df_raw_pre4['KO1_MOTOR_INVERTER_ALM']==8995.0].index
# STATUS 2.0 5개 0.0 1개
(df_raw_pre4['STATUS'].ffill()).loc[large_val_idx-2].value_counts()
# STATUS 2.0 4개 0.5 1개 0.0 1개
(df_raw_pre4['STATUS'].ffill()).loc[large_val_idx].value_counts()
# STATUS 2.0 5개 0.5 1개
(df_raw_pre4['STATUS'].ffill()).loc[large_val_idx+100].value_counts()
# STATUS 2.0 6개
(df_raw_pre4['STATUS'].ffill()).loc[large_val_idx+500].value_counts()
# 딱 한 번만 도움이 됐고 다른 때는 도움 안 됨

# 3. 모두 작은 값(컬럼따라 0또는 1)이 찍힌 인덱스들
small_val_idx = df_raw_pre4['KO1_MOTOR_INVERTER_ALM']\
    .loc[df_raw_pre4['KO1_MOTOR_INVERTER_ALM']==1.0].index
# STATUS 2.0 9개 0.0 8개
(df_raw_pre4['STATUS'].ffill()).loc[small_val_idx-2].value_counts()
# STATUS 2.0 8개 0.0 9개
(df_raw_pre4['STATUS'].ffill()).loc[small_val_idx].value_counts()
# STATUS 2.0 11개 0.0 6개
(df_raw_pre4['STATUS'].ffill()).loc[small_val_idx+100].value_counts()
# STATUS 2.0 13개 0.0 4개
(df_raw_pre4['STATUS'].ffill()).loc[small_val_idx+300].value_counts()
# 도움 안 됨

# 세 알람 모두 제거



# 위에서 제거하기로 판단했던 열 제거
df_raw_pre7=df_raw_pre4.drop(columns=['TONGS_POS','TONGS_CAST_CURR',
                                      'CUTTING_INVERTER_ALM_ERR_CD',
                                      'KO1_MOTOR_INVERTER_ALM',
                                      'MAIN_MOTOR_ALM'])
df_raw_pre7.shape # (579287, 19)
df_raw_pre7.count().sort_values()

# =============================================================================
# TONGS_CAST_SET_FREQ                 58
# STATUS                             128
# MAIN_MOTOR_RPM                     277
# METAL_TEMP_CUT                    2135
# METAL_TEMP_CONTROL                2152
# OUTPUT_COUNT_SUM                 11630
# OUTPUT_COUNT_DAY_2              472184
# OUTPUT_COUNT_DAY_1              472190
# TRANS_POS_DOWN                  492853
# TRANS_POS_UP                    498636
# TRANS_POS_LEFT                  503440
# WORK_OIL_SUPPLY_PRESS           519477
# TRANS_POS_RIGHT                 546178
# MAIN_MOTOR_CURR                 553049
# METAL_OIL_SUPPLY_PRESS_CUT      577210
# METAL_OIL_SUPPLY_PRESS_CONTR    577311
# OIL_SUPPLY_PRESS                577526
# MAIN_AIR_PRESS                  579287
# Timestamp                       579287
# =============================================================================


# 이제 이상치 제거 단계

# 이상치 제거가 필요하다고 보이는 컬럼 확인 위해
# 그래프들 다시 확인
for col in df_raw_pre7.drop(columns=['Timestamp']).columns.sort_values():
    plt.figure()
    df_raw_pre7[col].plot(title=col, style='o',ms=3)
    plt.show()
# main air press, main motor curr, metal oil supply press 두 가지 다, oil supply press, 
# trans pos 네 가지 다, work oil supply press 가 이상치 제거 필요하게 보임
# 이들은 count가 490000 이상인 컬럼들 중 timestamp 제외한 전부에 해당


# 이상치 제거가 필요하다고 보이는 컬럼 리스트 생성
columns_outliers = df_raw_pre7.count().sort_values()\
    [df_raw_pre7.count().sort_values()>490000].index
columns_outliers = list(columns_outliers.difference(pd.Index(["Timestamp"])))

# 이상치들이 항상 특정한 status를 가지는 등 규칙성이 있으면 이상치도 특별한 정보를 갖고 있는 것이기 때문에
# 이상치 제거 전에 이를 확인하고자 함
# 이상치들만 status 따라 색깔 다른 점 찍어봄

# 이상치들만 status 따라 색깔 다르게 점 찍는 추가적인 작업을 할 거기 때문에
# 컬럼마다 이상치가 아닌 범위를 구해놓는 과정이 필요


def outlier_boundary(x):
    Q1=x.quantile(1/4)
    Q3=x.quantile(3/4)
    IQR=Q3-Q1
    LL=Q1-(1.5*IQR)
    UU=Q3+(1.5*IQR)
    return (LL,UU)

# 위에서 이상치를 제거하기로 결정한 컬럼들을 key로
# outlier boundary를 value로 갖는 딕셔너리 생성
outlier_bound_dict=dict()
for col in columns_outliers:
    LL,UU=outlier_boundary(df_raw_pre7[col])
    outlier_bound_dict[col]=[LL,UU]

outlier_bound_dict

# STATUS ffill
df_raw_pre8=df_raw_pre7.copy(deep=True)
df_raw_pre8['STATUS']=df_raw_pre8['STATUS'].ffill()

df_raw_pre8['STATUS'].plot(style='o',ms=3)
df_raw_pre8['STATUS'].isna()
# 앞쪽에 status 없는 부분 제거
df_raw_pre8=df_raw_pre8.dropna(subset=['STATUS'])
df_raw_pre8['STATUS'].notna().all() # True

# 그래프
status_colors = {0: 'pink', 0.5: 'red', 1: 'green', 2: 'gray'}
for col in columns_outliers:
    _, ax= plt.subplots()
    ax.plot(df_raw_pre8.index, df_raw_pre8[col], 'o', ms=3)
    ax.set_title(col)
    for idx in df_raw_pre8[col][df_raw_pre8[col].notna()].index:
        # 이상치에 해당하면
        if df_raw_pre8[col][idx]<outlier_bound_dict[col][0]\
            or df_raw_pre8[col][idx]>outlier_bound_dict[col][1]:
            color = status_colors[df_raw_pre8['STATUS'][idx]]
            ax.plot(idx, df_raw_pre8[col][idx], 'o', markersize=4, color=color)
    plt.show()
# 결과 분석    
# 그래프들 이상치에 규칙성이 보이는 것들 있음 (그래프 보고서에 설명)
# status에 대한 정보를 주지만, 갯수로 치면 미미한 경우들일 뿐임


# 정상치에 속한 status 0.5까지 빨간 점 찍은 그래프
df_raw_pre8_red=df_raw_pre8.copy(deep=True)
df_raw_pre8_red.loc[df_raw_pre8_red['STATUS'] != 0.5] = np.nan
status_colors = {0: 'pink', 0.5: 'red', 1: 'green', 2: 'gray'}
for col in columns_outliers:
    _, ax= plt.subplots()
    ax.plot(df_raw_pre8.index, df_raw_pre8[col], 'o', ms=3)
    ax.plot(df_raw_pre8.index, df_raw_pre8_red[col], 'o', ms=3,\
            color=status_colors[0.5])
    ax.set_title(col)
    for idx in df_raw_pre8[col][df_raw_pre8[col].notna()].index:
        # 이상치에 해당하면
        if df_raw_pre8[col][idx]<outlier_bound_dict[col][0]\
            or df_raw_pre8[col][idx]>outlier_bound_dict[col][1]:
            color = status_colors[df_raw_pre8['STATUS'][idx]]
            ax.plot(idx, df_raw_pre8[col][idx], 'o', markersize=4, color=color)
    plt.show()

 
# 빨간 점 중 이상치 비율 구해봄
# num_red는 전체 데이터 중 status 0.5 갯수
num_red = len(df_raw_pre8[df_raw_pre8['STATUS']==0.5])
for col in columns_outliers:
    # num_red_outliers는 각 컬럼에서 status 0.5이면서 이상치인 애들
    num_red_outliers = ((df_raw_pre8_red[col] < outlier_bound_dict[col][0])\
                        | (df_raw_pre8_red[col] > outlier_bound_dict[col][1]))\
        .sum()
    print(f'{col} : {num_red_outliers/num_red:.4f}')
    
# =============================================================================
# MAIN_AIR_PRESS : 0.0026
# MAIN_MOTOR_CURR : 0.0026
# METAL_OIL_SUPPLY_PRESS_CONTR : 0.0000
# METAL_OIL_SUPPLY_PRESS_CUT : 0.0000
# OIL_SUPPLY_PRESS : 0.0000
# TRANS_POS_DOWN : 0.0017
# TRANS_POS_LEFT : 0.0031
# TRANS_POS_RIGHT : 0.0261
# TRANS_POS_UP : 0.0022
# WORK_OIL_SUPPLY_PRESS : 0.0022
# =============================================================================

# 그 시간대 정상치 내에 status 같은 값들이 훨씬 많이 속해 있음
# 이들을 이용해 status를 맞출 줄 모른다면
# 그건 심각하게 망한 알고리즘이 될 것. 따라서 어차피 정상치로 예측이 가능해야 함


# 이상치 제거
df_raw_pre9=df_raw_pre8.copy(deep=True)
for col in columns_outliers:
    df_raw_pre9[col].loc[df_raw_pre9[col] < outlier_bound_dict[col][0]] = np.nan
    df_raw_pre9[col].loc[df_raw_pre9[col] > outlier_bound_dict[col][1]] = np.nan

df_raw_pre9.shape # (579205, 19)
df_raw_pre9.count().sort_values()

# =============================================================================
# TONGS_CAST_SET_FREQ                 56
# MAIN_MOTOR_RPM                     277
# METAL_TEMP_CUT                    2135
# METAL_TEMP_CONTROL                2152
# OUTPUT_COUNT_SUM                 11630
# OUTPUT_COUNT_DAY_2              472179
# OUTPUT_COUNT_DAY_1              472185
# WORK_OIL_SUPPLY_PRESS           474836
# TRANS_POS_DOWN                  491367
# TRANS_POS_UP                    497685
# TRANS_POS_LEFT                  502843
# TRANS_POS_RIGHT                 539206
# MAIN_MOTOR_CURR                 552890
# METAL_OIL_SUPPLY_PRESS_CONTR    575778
# METAL_OIL_SUPPLY_PRESS_CUT      575790
# OIL_SUPPLY_PRESS                576525
# MAIN_AIR_PRESS                  578882
# STATUS                          579205
# Timestamp                       579205
# =============================================================================

# 이상치 처리 종료 후 그래프들
for col in df_raw_pre9.drop(columns=['Timestamp']).columns.sort_values():
    plt.figure()
    df_raw_pre9[col].plot(title=col, style='o',ms=0.3)
    plt.show()

# main motor curr
# 그래프 아래 점 몇 개 있지도 않은데 범위만 엄청 잡아먹음. 확대해서 그려봄
df_raw_pre9['MAIN_MOTOR_CURR'].loc[df_raw_pre9['MAIN_MOTOR_CURR'] < 4000]\
    .plot(style='o',ms=3)
len(df_raw_pre9['MAIN_MOTOR_CURR'].loc[df_raw_pre9['MAIN_MOTOR_CURR'] < 3750])
# 3750 아래로점 8개도 안 되는데 범위 엄청 차지하므로 제거
df_raw_pre9['MAIN_MOTOR_CURR'].loc[df_raw_pre9['MAIN_MOTOR_CURR'] < 3750]=np.nan
df_raw_pre9['MAIN_MOTOR_CURR'].plot(title=col, style='o',ms=0.3)

# df_raw_pre9['MAIN_MOTOR_CURR'].plot(title='MAIN_MOTOR_CURR', style='o',ms=0.3)

# 결측치 처리
df_raw_pre10=df_raw_pre9.copy(deep=True)

# tongs cast set freq
df_raw_pre10['TONGS_CAST_SET_FREQ'].plot(style='o',ms=0.3)
#ffill
df_raw_pre10['TONGS_CAST_SET_FREQ'].ffill(inplace=True)

df_raw_pre10['TONGS_CAST_SET_FREQ'].plot(style='o',ms=0.3)

# 6000이 499410 경우, 2000이 62 경우가 되는 걸 볼 때
# 2000은 1분만 사용됐다는 건데 말이 안 된다고 생각
# 정말로 set freq한 게 아니라고 보여짐
df_raw_pre10['TONGS_CAST_SET_FREQ'].value_counts()
# 6000.0    499410
# 2000.0        62


# 열 제거
df_raw_pre10.drop(columns=['TONGS_CAST_SET_FREQ'], inplace=True)
df_raw_pre10.columns.sort_values()

# 남은 column 확인
# =============================================================================
# Index(['MAIN_AIR_PRESS', 'MAIN_MOTOR_CURR', 'MAIN_MOTOR_RPM',
#        'METAL_OIL_SUPPLY_PRESS_CONTR', 'METAL_OIL_SUPPLY_PRESS_CUT',
#        'METAL_TEMP_CONTROL', 'METAL_TEMP_CUT', 'OIL_SUPPLY_PRESS',
#        'OUTPUT_COUNT_DAY_1', 'OUTPUT_COUNT_DAY_2', 'OUTPUT_COUNT_SUM',
#        'STATUS', 'TRANS_POS_DOWN', 'TRANS_POS_LEFT', 'TRANS_POS_RIGHT',
#        'TRANS_POS_UP', 'Timestamp', 'WORK_OIL_SUPPLY_PRESS'],
#       dtype='object')
# =============================================================================


# output count day1 / day2 / sum
for col in df_raw_pre10.filter(regex='^OUTPUT').columns:
    df_raw_pre10[col].plot(title=col,style='o',ms=0.3)
    plt.show()
# 추세가 단순,명확. interpolate 

# interpolate 위해 인덱스가 timestamp인 df 생성
df_raw_pre10 = df_raw_pre10.set_index('Timestamp')  
#  interpolate 
for col in df_raw_pre10.filter(regex='^OUTPUT').columns:  
    df_raw_pre10[col].interpolate(title=col,method='index',inplace=True)
    df_raw_pre10[col].plot(title=col,style='o',ms=0.3)
    plt.show()


# metal temp control
df_raw_pre10['METAL_TEMP_CONTROL'].plot(style='o',ms=0.3)
# 추세가 단순,명확. interpolate
df_raw_pre10['METAL_TEMP_CONTROL'].interpolate(method='index',inplace=True)

df_raw_pre10['METAL_TEMP_CONTROL'].plot(style='o',ms=0.3)


# metal temp cut
df_raw_pre10['METAL_TEMP_CUT'].plot(style='o',ms=0.3)
# 추세가 단순,명확. interpolate
df_raw_pre10['METAL_TEMP_CUT'].interpolate(method='index',inplace=True)

df_raw_pre10['METAL_TEMP_CUT'].plot(style='o',ms=0.3)


# main motor rpm
# 위에서 살펴봤듯 확대해서 보면
# 선형 증가/감소 형태
full_time_df['MAIN_MOTOR_RPM']\
    ['2022-05-04 07:58:11':'2022-05-04 08:07:13'].plot(style='o',ms=3)

# 위 plot에서 조그만하게 있고 분별 안 가는 부분 확대
# 선형적인 산 형태를 알 수 있음
full_time_df['MAIN_MOTOR_RPM']\
    ['2022-05-04 08:04:11':'2022-05-04 08:06:13'].plot(style='o',ms=3)

# 선형적 변화 없는 다른 곳에서는 꾸준히 높은 rpm들로 찍혔음
df_raw_pre9['MAIN_MOTOR_RPM'].plot(style='o',ms=3)

# interpolate가 적절하게 보임
df_raw_pre10['MAIN_MOTOR_RPM'].interpolate(method='index',inplace=True)

df_raw_pre10['MAIN_MOTOR_RPM'].plot(style='o',ms=3)

# 위에서 봤던 구간
df_raw_pre10['MAIN_MOTOR_RPM']\
    ['2022-05-04 07:58:11':'2022-05-04 08:07:13'].plot(style='o',ms=3)






# 위에서 처리한 애들 결측치 확인
df_raw_pre10[['OUTPUT_COUNT_DAY_1', 'OUTPUT_COUNT_DAY_2', 
              'OUTPUT_COUNT_SUM','METAL_TEMP_CONTROL',
              'METAL_TEMP_CUT','MAIN_MOTOR_RPM', 'STATUS']].isna().sum()

# =============================================================================
# OUTPUT_COUNT_DAY_1        0
# OUTPUT_COUNT_DAY_2        0
# OUTPUT_COUNT_SUM         19
# METAL_TEMP_CONTROL      122
# METAL_TEMP_CUT           30
# MAIN_MOTOR_RPM        19424
# STATUS                    0
# =============================================================================

# 어디가 빠진 건지 그래프  
for col in ['OUTPUT_COUNT_DAY_1', 'OUTPUT_COUNT_DAY_2', 'OUTPUT_COUNT_SUM',
            'METAL_TEMP_CONTROL', 'METAL_TEMP_CUT','MAIN_MOTOR_RPM', 'STATUS']:
    _, ax= plt.subplots()
    ax.plot(df_raw_pre10.index, df_raw_pre10[col], 'o', ms=0.3)
    ax.set_title(col)
    nan_index=df_raw_pre10[col].loc[df_raw_pre10[col].isna()].index
    for idx in nan_index:  
        plt.axvline(x=idx, color='red', linestyle='-',linewidth=0.4,alpha=0.2)
    plt.show()
# 앞쪽 비는 애들 있음

# main motor rpm 앞쪽 빔
df_raw_pre10['MAIN_MOTOR_RPM'].loc[df_raw_pre10['MAIN_MOTOR_RPM']>12000]\
                                   .plot(style='o',ms=0.3)
# 13000으로 채우면 될 듯
df_raw_pre10['MAIN_MOTOR_RPM'].loc[df_raw_pre10['MAIN_MOTOR_RPM'].isna()]=13000
df_raw_pre10['MAIN_MOTOR_RPM'].notna().all() # True


# 나머지들 비는 index 확인
# 제일 긴 METAL_TEMP_CONTROL이 시작부터 2분일 뿐
df_raw_pre10['OUTPUT_COUNT_SUM'].\
    loc[df_raw_pre10['OUTPUT_COUNT_SUM'].isna()].index
df_raw_pre10['METAL_TEMP_CONTROL'].\
    loc[df_raw_pre10['METAL_TEMP_CONTROL'].isna()].index
df_raw_pre10['METAL_TEMP_CUT'].loc[df_raw_pre10['METAL_TEMP_CUT'].isna()].index

# =============================================================================
# DatetimeIndex(['2022-05-02 06:33:56', '2022-05-02 06:33:57',
#                '2022-05-02 06:33:58', '2022-05-02 06:33:59',
#                '2022-05-02 06:34:00', '2022-05-02 06:34:01',
#                '2022-05-02 06:34:02', '2022-05-02 06:34:03',
#                '2022-05-02 06:34:04', '2022-05-02 06:34:05',
#                '2022-05-02 06:34:06', '2022-05-02 06:34:07',
#                '2022-05-02 06:34:08', '2022-05-02 06:34:09',
#                '2022-05-02 06:34:10', '2022-05-02 06:34:11',
#                '2022-05-02 06:34:12', '2022-05-02 06:34:13',
#                '2022-05-02 06:34:14']
#               
# DatetimeIndex(['2022-05-02 06:33:56', '2022-05-02 06:33:57',
#                '2022-05-02 06:33:58', '2022-05-02 06:33:59',
#                '2022-05-02 06:34:00', '2022-05-02 06:34:01',
#                '2022-05-02 06:34:02', '2022-05-02 06:34:03',
#                '2022-05-02 06:34:04', '2022-05-02 06:34:05',
#                ...
#                '2022-05-02 06:35:48', '2022-05-02 06:35:49',
#                '2022-05-02 06:35:50', '2022-05-02 06:35:51',
#                '2022-05-02 06:35:52', '2022-05-02 06:35:53',
#                '2022-05-02 06:35:54', '2022-05-02 06:35:55',
#                '2022-05-02 06:35:56', '2022-05-02 06:35:57']
#               
# DatetimeIndex(['2022-05-02 06:33:56', '2022-05-02 06:33:57',
#                '2022-05-02 06:33:58', '2022-05-02 06:33:59',
#                '2022-05-02 06:34:00', '2022-05-02 06:34:01',
#                '2022-05-02 06:34:02', '2022-05-02 06:34:03',
#                '2022-05-02 06:34:04', '2022-05-02 06:34:05',
#                '2022-05-02 06:34:06', '2022-05-02 06:34:07',
#                '2022-05-02 06:34:08', '2022-05-02 06:34:09',
#                '2022-05-02 06:34:10', '2022-05-02 06:34:11',
#                '2022-05-02 06:34:12', '2022-05-02 06:34:13',
#                '2022-05-02 06:34:14', '2022-05-02 06:34:15',
#                '2022-05-02 06:34:16', '2022-05-02 06:34:17',
#                '2022-05-02 06:34:18', '2022-05-02 06:34:19',
#                '2022-05-02 06:34:20', '2022-05-02 06:34:21',
#                '2022-05-02 06:34:22', '2022-05-02 06:34:23',
#                '2022-05-02 06:34:24', '2022-05-02 06:34:25']              
# =============================================================================
# 제거해버림

df_raw_pre10.dropna(subset=['METAL_TEMP_CONTROL'],inplace=True)

df_raw_pre10[['OUTPUT_COUNT_DAY_1', 'OUTPUT_COUNT_DAY_2',
              'OUTPUT_COUNT_SUM','METAL_TEMP_CONTROL',
              'METAL_TEMP_CUT','MAIN_MOTOR_RPM', 'STATUS']].isna().sum()

# =============================================================================
# OUTPUT_COUNT_DAY_1    0
# OUTPUT_COUNT_DAY_2    0
# OUTPUT_COUNT_SUM      0
# METAL_TEMP_CONTROL    0
# METAL_TEMP_CUT        0
# MAIN_MOTOR_RPM        0
# STATUS                0
# =============================================================================

# 뒤에서, 결측치 아직 안 채운 컬럼들 결측치 채우기 전후 그래프 비교용. 전에 해당
df_raw_pre10_temp = df_raw_pre10.copy(deep=True)


df_raw_pre10.isna().sum().sort_values()
# =============================================================================
# OUTPUT_COUNT_DAY_1                   0
# OUTPUT_COUNT_SUM                     0
# OUTPUT_COUNT_DAY_2                   0
# METAL_TEMP_CUT                       0
# METAL_TEMP_CONTROL                   0
# STATUS                               0
# MAIN_MOTOR_RPM                       0
# MAIN_AIR_PRESS                     323
# OIL_SUPPLY_PRESS                  2558
# METAL_OIL_SUPPLY_PRESS_CONTR      3305
# METAL_OIL_SUPPLY_PRESS_CUT        3415
# MAIN_MOTOR_CURR                  26323
# TRANS_POS_RIGHT                  39999
# TRANS_POS_LEFT                   76350
# TRANS_POS_UP                     81520
# TRANS_POS_DOWN                   87832
# WORK_OIL_SUPPLY_PRESS           104247
# =============================================================================


# 남은 컬럼들
for col in df_raw_pre10.isna().sum().sort_values()\
    .loc[df_raw_pre10.isna().sum().sort_values()>0].index:
    plt.figure()
    df_raw_pre10[col].plot(title=col, style='o',ms=0.3)
    plt.show()


# 남은 컬럼별 결측치 백분율
df_raw_pre10[df_raw_pre10.isna().sum().sort_values()\
             .loc[df_raw_pre10.isna().sum().sort_values()>0].index]\
    .isna().sum().sort_values() / len(df_raw_pre10) *100
# 결측치 비율 1% 이하 컬럼이 4개. 4~7% 2개. 13~18% 4개
# 결측치 적은 컬럼부터 채워나가면, 13~18%인 컬럼들을 채울 때 더 많은 데이터를 이용할 수 있을 것

# =============================================================================
# MAIN_AIR_PRESS                   0.055778
# OIL_SUPPLY_PRESS                 0.441733
# METAL_OIL_SUPPLY_PRESS_CONTR     0.570730
# METAL_OIL_SUPPLY_PRESS_CUT       0.589725
# MAIN_MOTOR_CURR                  4.545635
# TRANS_POS_RIGHT                  6.907300
# TRANS_POS_LEFT                  13.184638
# TRANS_POS_UP                    14.077429
# TRANS_POS_DOWN                  15.167429
# WORK_OIL_SUPPLY_PRESS           18.002083
# =============================================================================

# 랜덤포레스트(n_estimators=10)를 이용해 결측치 채우고자 함

# =============================================================================
# # 로컬 환경이 느려 코랩에서 돌리고자 csv로
df_raw_pre10.to_csv("before_rdf.csv")
# =============================================================================

# 아래 실행하면 sklearn 버전 따라 Valueerror 뜰 수도 있음
# ValueError: Input X contains NaN. RandomForestRegressor does not accept missing values encoded as NaN natively.
# 해결책 : 오른쪽 콘솔창에 ! pip install --upgrade scikit-learn 입력하고 실행해서 버전 업그레이드

for col in df_raw_pre10.isna().sum().sort_values().loc\
    [df_raw_pre10.isna().sum().sort_values()>0].index: 
    train= df_raw_pre10.dropna(subset=col)
    test = df_raw_pre10.loc[df_raw_pre10[col].isna()]

    rf = RandomForestRegressor(n_estimators=10, random_state=1)
    rf.fit(train.drop(columns=[col]), train[col])

    df_raw_pre10.loc[df_raw_pre10[col].isna(),col] =\
        rf.predict(test.drop(columns=[col]))
    print(f'{col} 완료\n')
    
# =============================================================================
# MAIN_AIR_PRESS 완료
# 
# OIL_SUPPLY_PRESS 완료
# 
# METAL_OIL_SUPPLY_PRESS_CONTR 완료
# 
# METAL_OIL_SUPPLY_PRESS_CUT 완료
# 
# MAIN_MOTOR_CURR 완료
# 
# TRANS_POS_RIGHT 완료
# 
# TRANS_POS_LEFT 완료
# 
# TRANS_POS_UP 완료
# 
# TRANS_POS_DOWN 완료
# 
# WORK_OIL_SUPPLY_PRESS 완료    
# =============================================================================

# =============================================================================
# # 학습이 오래 걸려서 코랩에서 돌려서 '랜덤포레스트까지.csv'로 만들어서 가져옴
df_raw_pre10 = pd.read_csv('랜덤포레스트까지.csv')
df_raw_pre10['Timestamp'] = pd.to_datetime(df_raw_pre10['Timestamp'])
df_raw_pre10.set_index('Timestamp', inplace=True)
# =============================================================================

# 다 채워졌나 확인
df_raw_pre10.notna().all().all() # True

    
# 채운 컬럼들 결측치 채우기 전후 한 그래프에 그려봄
# 빨간 색이 결측치 새롭게 채운 부분. 검은색은 원래 결측치 아닌 부분.
df_raw_pre10_temp.set_index('Timestamp', inplace=True)
for col in df_raw_pre10_temp.isna().sum().sort_values()\
    .loc[df_raw_pre10_temp.isna().sum().sort_values()>0].index: 
    plt.figure()
    df_raw_pre10[col].plot(title=col, style='o',ms=0.3,color='red')
    df_raw_pre10_temp[col].plot(title=col, style='o',ms=0.3,color='black')
    plt.show()
# 추세에 맞게 적절히 채워졌다고 판단됨
# 다만, 'MAIN_MOTOR_CURR' 05-02, 'WORK_OIL_SUPPLY_PRESS' 05-02~05-03에 추세와 맞지 않는 부분이 보임

# MAIN_MOTOR_CURR
# 확대해서 보면,
plt.figure()
df_raw_pre10['MAIN_MOTOR_CURR']['2022-05-02 11:30:11':'2022-05-02 18:10:13']\
    .plot(style='o',ms=0.6,color='red')
df_raw_pre10_temp['MAIN_MOTOR_CURR']\
    ['2022-05-02 11:30:11':'2022-05-02 18:10:13']\
        .plot(style='o',ms=0.6,color='black')
plt.show()

# 2022-05-02 11:58:20부터 계속해서 결측치
df_raw_pre10['MAIN_MOTOR_CURR']['2022-05-02 11:30:11':
                                '2022-05-02 18:10:13'].loc\
    [df_raw_pre10_temp['MAIN_MOTOR_CURR']['2022-05-02 11:30:11':
                                          '2022-05-02 18:10:13']\
     .isna()].index[:20]
    
# =============================================================================
# DatetimeIndex(['2022-05-02 11:44:10', '2022-05-02 11:51:01',
#                '2022-05-02 11:51:12', '2022-05-02 11:58:00',
#                '2022-05-02 11:58:20', '2022-05-02 11:58:21',
#                '2022-05-02 11:58:22', '2022-05-02 11:58:23',
#                '2022-05-02 11:58:24', '2022-05-02 11:58:25',
#                '2022-05-02 11:58:26', '2022-05-02 11:58:27',
#                '2022-05-02 11:58:28', '2022-05-02 11:58:29',
#                '2022-05-02 11:58:30', '2022-05-02 11:58:31',
#                '2022-05-02 11:58:32', '2022-05-02 11:58:33',
#                '2022-05-02 11:58:34', '2022-05-02 11:58:35'],    
# =============================================================================

# 2022-05-02 16:44:23까지 연속된 결측치
df_raw_pre10['MAIN_MOTOR_CURR']['2022-05-02 11:30:11':
                                '2022-05-02 18:10:13'].loc\
    [df_raw_pre10_temp['MAIN_MOTOR_CURR']\
     ['2022-05-02 11:30:11':'2022-05-02 18:10:13'].isna()].index[-20:]

# =============================================================================
# DatetimeIndex(['2022-05-02 16:44:08', '2022-05-02 16:44:09',
#                '2022-05-02 16:44:10', '2022-05-02 16:44:11',
#                '2022-05-02 16:44:12', '2022-05-02 16:44:13',
#                '2022-05-02 16:44:14', '2022-05-02 16:44:15',
#                '2022-05-02 16:44:16', '2022-05-02 16:44:17',
#                '2022-05-02 16:44:18', '2022-05-02 16:44:19',
#                '2022-05-02 16:44:20', '2022-05-02 16:44:21',
#                '2022-05-02 16:44:22', '2022-05-02 16:44:23',
#                '2022-05-02 16:57:12', '2022-05-02 17:05:44',
#                '2022-05-02 17:08:11', '2022-05-02 17:21:02'],
#               dtype='datetime64[ns]', name='Timestamp', freq=None)
# =============================================================================

# 2022-05-02 11:58:20 ~ 16:44:23 제거
idx_drop = df_raw_pre10.index.intersection(pd.date_range\
                                           (start='2022-05-02 11:58:20',
                                            end='2022-05-02 16:44:23', freq='S'))
df_raw_pre10.drop(idx_drop, inplace=True)

# WORK_OIL_SUPPLY_PRESS
# 05-03 00 ~ 05 쯤이 경향이 다르고 결측치 채운 값들로만 이뤄져 있음
plt.figure()
df_raw_pre10['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-02 07:00:11':'2022-05-03 20:10:13']\
        .plot(style='o',ms=0.6,color='red')
df_raw_pre10_temp['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-02 07:00:11':'2022-05-03 20:10:13']\
        .plot(style='o',ms=0.6,color='black')
plt.show()

# 해당 부분 확대
plt.figure()
df_raw_pre10['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-02 23:38:20':'2022-05-03 05:44:23']\
        .plot(style='o',ms=0.6,color='red')
df_raw_pre10_temp['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-02 23:38:20':'2022-05-03 05:44:23']\
        .plot(style='o',ms=0.6,color='black')
plt.show()


# 원 데이터 끝나는 지점 보면
df_raw_pre10_temp['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-02 23:38:20':'2022-05-03 01:00:23']\
        .plot(style='o',ms=0.6,color='black')
# index 조회해보면 '2022-05-03 00:17:29' 에서 끝남
df_raw_pre10_temp['2022-05-03 00:15:20':'2022-05-03 00:20:23']\
    .loc[df_raw_pre10_temp['WORK_OIL_SUPPLY_PRESS']\
         ['2022-05-03 00:15:20':'2022-05-03 00:20:23'].notna()].index
# 결측치로 채운 곳 끝나는 지점 보면
df_raw_pre10['WORK_OIL_SUPPLY_PRESS']\
    ['2022-05-03 04:08:20':'2022-05-03 05:44:23']\
        .plot(style='o',ms=0.6,color='red')
        
# '2022-05-03 04:43:32'에서 끝남
df_raw_pre10['2022-05-03 04:40:20':'2022-05-03 05:44:23']\
    .loc[df_raw_pre10['WORK_OIL_SUPPLY_PRESS']\
         ['2022-05-03 04:40:20':'2022-05-03 05:44:23'].notna()].index

# 따라서 '2022-05-03 00:17:29' ~ '2022-05-03 04:43:32' 제거
idx_drop = df_raw_pre10.index.intersection(pd.date_range\
                                           (start='2022-05-03 00:17:29',
                                            end='2022-05-03 04:43:32', freq='S'))
df_raw_pre10.drop(idx_drop, inplace=True)

# 제거 후 두 컬럼 그래프
for col in ['MAIN_MOTOR_CURR','WORK_OIL_SUPPLY_PRESS']: 
    plt.figure()
    df_raw_pre10[col].plot(title=col, style='o',ms=0.3,color='red')
    df_raw_pre10_temp[col].plot(title=col, style='o',ms=0.3,color='black')
    plt.show()
    
# 두 컬럼 통해 수정하며 5% 정도 데이터 날렸음
1 - len(df_raw_pre10) / len(df_raw_pre10_temp)
# 545991 행 남음
len(df_raw_pre10)


# 정규화
scaler = MinMaxScaler()
scaler.fit(df_raw_pre10)
df_raw_pre11 = scaler.transform(df_raw_pre10)
df_raw_pre_fin = pd.DataFrame(data=df_raw_pre11, columns = df_raw_pre10.columns)

df_raw_pre_fin.shape # (545991, 17) 
df_raw_pre_fin.count().sort_values()

# =============================================================================
# OUTPUT_COUNT_DAY_1              545991
# OUTPUT_COUNT_SUM                545991
# OUTPUT_COUNT_DAY_2              545991
# METAL_TEMP_CUT                  545991
# MAIN_AIR_PRESS                  545991
# METAL_OIL_SUPPLY_PRESS_CUT      545991
# TRANS_POS_LEFT                  545991
# TRANS_POS_DOWN                  545991
# METAL_TEMP_CONTROL              545991
# TRANS_POS_UP                    545991
# OIL_SUPPLY_PRESS                545991
# MAIN_MOTOR_RPM                  545991
# MAIN_MOTOR_CURR                 545991
# METAL_OIL_SUPPLY_PRESS_CONTR    545991
# STATUS                          545991
# WORK_OIL_SUPPLY_PRESS           545991
# TRANS_POS_RIGHT                 545991
# =============================================================================

# 최종 그래프
for col in df_raw_pre_fin.columns.sort_values():
    plt.figure()
    df_raw_pre_fin[col].plot(title=col, style='o',ms=0.3)
    plt.show()


# =============================================================================
# # to csv
df_raw_pre_fin.to_csv("전처리 완료(index 없음).csv",index=False)
# =============================================================================

# 전처리 완료

# 알고리즘 시작

from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import time
from sklearn.neighbors import KNeighborsClassifier

# 0. 라벨링
df_raw_pre_fin = pd.read_csv("전처리 완료(index 없음).csv")
df_encoded = df_raw_pre_fin.copy()
df_encoded['STATUS'].value_counts()
# =============================================================================
# STATUS
# 1.00    457215
# 0.00     82406
# 0.50      4075
# 0.25      2295
# Name: count, dtype: int64
# =============================================================================

Y = df_encoded['STATUS']
X = df_encoded.drop(columns=['STATUS'])

le = LabelEncoder()
Y = le.fit_transform(Y)
unique_counts = np.unique(Y, return_counts = True)
# =============================================================================
# (array([0, 1, 2, 3], dtype=int64),
#  array([ 82406,   2295,   4075, 457215], dtype=int64))
# =============================================================================
# 0>0   0.25>1(비정상)   0.5>2   1>3(정상) 으로 라벨링됨


# 1. 샘플링
# 1.1. simple
X_train, X_test, Y_train, Y_test = train_test_split(X,  Y, test_size = 0.3,  
                                                    random_state = 42)

# 1.2. SMOTE
smote=SMOTE(random_state=0)
smote_X, smote_Y=smote.fit_resample(X, Y)
pd.Series(Y).value_counts()
# =============================================================================
# 3    457215
# 0     82406
# 2      4075
# 1      2295
# Name: count, dtype: int64
# =============================================================================
pd.Series(smote_Y).value_counts()
# =============================================================================
# 3    457215
# 0    457215
# 1    457215
# 2    457215
# Name: count, dtype: int64
# =============================================================================

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(smote_X,smote_Y,
                                                        test_size = 0.3, 
                                                        random_state = 42)


# 1.2.3. SMOTE + stratify=Y
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(smote_X,  smote_Y,
                                                        test_size = 0.3,  
                                                        random_state = 42, 
                                                        stratify=smote_Y)


# 2. 알고리즘
# 2.1. DecisionTreeClssifier
# 2.1.1. simple
# 2.1.1.1. default
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, Y_train)
Y_predict = dt.predict(X_test)
dt_report=classification_report(Y_test, Y_predict)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     24619
#            1       0.99      0.99      0.99       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      1.00      1.00    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================
accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average='weighted')
recall = recall_score(Y_test, Y_predict, average='weighted')
f1 = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy}')
print(f'Precision (weighted): {precision}')
print(f'Recall (weighted): {recall}')
print(f'F1 Score (weighted): {f1}')
# =============================================================================
# Accuracy: 0.9993345462093555
# Precision (weighted): 0.9993343713181775
# Recall (weighted): 0.9993345462093555
# F1 Score (weighted): 0.9993343902416277
# =============================================================================
# simple random sampling의 경우 데이터가 매우 불균형 하므로 average = 'weighted'를 선택함

# plot_tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=X.columns, class_names=[str(cls) for cls in np.unique(Y)])
# Y가 numpy배열 , 숫자를 문자로 반환
plt.show()

# 2.1.1.2. BayesianOptimization
def DT( max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    model = DecisionTreeClassifier( max_depth=int(round(max_depth)), 
        min_samples_split=int(round(min_samples_split)),  
        min_samples_leaf=int(round(min_samples_leaf)),  
        max_leaf_nodes=int(round(max_leaf_nodes)),
        class_weight='balanced' 
        )
        
    scoring = {'f1_score': make_scorer(f1_score, average='macro')}
    result = cross_validate(model, X_train, Y_train, cv=5, scoring=scoring)
    f1_score_mean = result["test_f1_score"].mean()
    return f1_score_mean

pbounds = {
           'max_depth': (2, 20),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 10),
           'max_leaf_nodes': (2, 30),
         
          }

LFBO = BayesianOptimization(f = DT, pbounds = pbounds, verbose = 2, random_state = 0)
LFBO.maximize(init_points=5, n_iter = 20)
LFBO.max
# =============================================================================
# {'target': 0.9855738656043359,
#  'params': {'max_depth': 11.878643070691846,
#   'max_leaf_nodes': 22.025302258427747,
#   'min_samples_leaf': 6.4248703846447945,
#   'min_samples_split': 11.807897293944144}}
# =============================================================================
# =============================================================================
# # 202402 업데이트: Bayesian Optimization 라이브러리에서는 maximize 메소드의 사용법이 변경되어
# 
# # maximize() 메서드에서 acq 와 xi 를 직접 설정시 에러가 납니다. 따라서 해당 코드를 삭제하였습니다.
# 
# # 기존 코드: optimizer.maximize(init_points=10, n_iter=100, acq='ei', xi=0.01)
# 
# # maximize 메소드 호출
# (https://www.inflearn.com/community/questions/1164233/bayesian-optimization%EC%97%90%EC%84%9C-optimizer-maximize-%ED%95%A8%EC%88%98%EB%A5%BC-%EB%8D%94%EC%9D%B4%EC%83%81-%EC%A7%80%EC%9B%90-%EC%95%88%ED%95%9C%EB%8B%A4%EA%B3%A0-%ED%95%A9%EB%8B%88%EB%8B%A4?srsltid=AfmBOorYUC_5B5TviI-KTqazBNJ5T9zoLkBEcd8Y81u-ism479ZZl9Q7)
# =============================================================================

dtc_bay = DecisionTreeClassifier(max_depth= 12, max_leaf_nodes=22, 
                                 min_samples_leaf= 6,min_samples_split=12, random_state= 0)
dtc_bay.fit(X_train, Y_train)
Y_predict = dtc_bay.predict(X_test)
dtc_bay_report=classification_report(Y_test, Y_predict)
accuracy_bay = accuracy_score(Y_test, Y_predict)
precision_bay = precision_score(Y_test, Y_predict, average='weighted')
recall_bay = recall_score(Y_test, Y_predict, average='weighted')
f1_bay = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy_bay}')
print(f'Precision (weighted): {precision_bay}')
print(f'Recall (weighted): {recall_bay}')
print(f'F1 Score (weighted): {f1_bay}')
# =============================================================================
# Accuracy: 0.997881537015104
# Precision (weighted): 0.9978817085248322
# Recall (weighted): 0.997881537015104
# F1 Score (weighted): 0.9978751334039059
# =============================================================================

# 2.1.1.3. BayesianOptimization + GridSearchCV
dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [10,11,12,13],
    'max_leaf_nodes': [20,21,22,23,24], 
    'min_samples_leaf': [4,5,6,7,8],  
    'min_samples_split': [10,11,12,13] 
}

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 10,
#  'max_leaf_nodes': 24,
#  'min_samples_leaf': 4,
#  'min_samples_split': 10}
# =============================================================================

dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [8,9,10,11],
    'max_leaf_nodes': [23,24,25,26], 
    'min_samples_leaf': [2,3,4,5],  
    'min_samples_split': [8,9,10,11] 
}

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 9,
#  'max_leaf_nodes': 26,
#  'min_samples_leaf': 2,
#  'min_samples_split': 8}
# =============================================================================

dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [8,9,10],
    'max_leaf_nodes': [25,26,27,28], 
    'min_samples_leaf': [1,2,3],  
    'min_samples_split': [6,7,8,9] 
}

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 9,
#  'max_leaf_nodes': 27,
#  'min_samples_leaf': 1,
#  'min_samples_split': 6}
# =============================================================================

dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [8,9,10],
    'max_leaf_nodes': [26,27,28], 
    'min_samples_leaf': [1,2,3],  
    'min_samples_split': [4,5,6,7] 
}

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 9,
#  'max_leaf_nodes': 27,
#  'min_samples_leaf': 1,
#  'min_samples_split': 4}
# =============================================================================

dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [8,9,10],
    'max_leaf_nodes': [26,27,28], 
    'min_samples_leaf': [1,2,3],  
    'min_samples_split': [1.0,2,3,4] 
}

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 9,
#  'max_leaf_nodes': 27,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2}
# =============================================================================
dtc_best = DecisionTreeClassifier(**best_params, random_state= 0)
dtc_best.fit(X_train, Y_train)

Y_predict = dtc_best.predict(X_test)

dtc_report=classification_report(Y_test, Y_predict)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      0.99      0.99     24619
#            1       1.00      0.97      0.98       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      0.99      0.99    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average='weighted')
recall = recall_score(Y_test, Y_predict, average='weighted')
f1 = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy}')
print(f'Precision (weighted): {precision}')
print(f'Recall (weighted): {recall}')
print(f'F1 Score (weighted): {f1}')
# =============================================================================
# Accuracy: 0.9980891097571399
# Precision (weighted): 0.9980916527710718
# Recall (weighted): 0.9980891097571399
# F1 Score (weighted): 0.9980840447927297
# =============================================================================
# 베이지안만 적용했을 때 보다는 결과가 향상됨
# 그러나 아무런 파라미터를 정의하지 않았을 때 보다는 결과가 좋지 않음
# 최적 파라미터가 default값으로 수렴하는 결과를 보임 
# 모델의 파라미터는 default를 이용하고 파라미터 증가에 따른 시간 변화를 확인해 봄

# 2.1.1.4 max_depth 증가에 따른 시간변화 
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X,  Y, test_size = 0.3,  random_state = 42, stratify=Y)
# 데이터의 불균형으로 인해 비정상상태 예측의 랜덤성이 커질 수 있음
# 각 파라미터 변화에 대한 시간을 측정하는데 오류가 있다고 판단
# random simple sampling 대신 stratified sampling를 적용 

max_depth_range = list(range(1,51)) # max_depth가 1-50까지 1씩 증가
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []
# 각 성능지표들을 리스트로 저장

for n in max_depth_range:
    dtc = DecisionTreeClassifier(max_depth=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train4, Y_train4)
    end_time = time.time()
    times.append(end_time - start_time) # 한번 fit하는데 걸린 시간을 리스트로 저장 
    
    Y_predict4 = dtc.predict(X_test4)

    accuracy_list.append(accuracy_score(Y_test4, Y_predict4))
    precision_list.append(precision_score(Y_test4, Y_predict4, average='macro'))
    recall_list.append(recall_score(Y_test4, Y_predict4, average='macro'))
    f1_list.append(f1_score(Y_test4, Y_predict4, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6)) #그래프

ax1.plot(max_depth_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_depth_range, precision_list, marker='x', label='precision')
ax1.plot(max_depth_range, recall_list, marker='s', label='recall')
ax1.plot(max_depth_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_depth_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

plt.title('DecisionTreeClassifier-max_depth(stratify)')
plt.show()
# 그래프 확인 결과 depth = 10, nodes = 30 이 최적값이라고 판단

# 2.1.1.5 max_leaf_nodes 증가에 따른 시간변화
max_leaf_nodes_range = list(range(2,501)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in max_leaf_nodes_range:
    dtc = DecisionTreeClassifier(max_leaf_nodes=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train4, Y_train4)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict4 = dtc.predict(X_test4)

    accuracy_list.append(accuracy_score(Y_test4, Y_predict4))
    precision_list.append(precision_score(Y_test4, Y_predict4, average='macro'))
    recall_list.append(recall_score(Y_test4, Y_predict4, average='macro'))
    f1_list.append(f1_score(Y_test4, Y_predict4, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(max_leaf_nodes_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_leaf_nodes_range, precision_list, marker='x', label='precision')
ax1.plot(max_leaf_nodes_range, recall_list, marker='s', label='recall')
ax1.plot(max_leaf_nodes_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_leaf_nodes')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_leaf_nodes_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('DecisionTreeClassifier-max_leaf_nodes(stratify)')
plt.show()
# 그래프 확인 결과 depth = 10, nodes = 30 이 최적값이라고 판단

# 2.1.2. SMOTE
# 2.1.2.1. default
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train2, Y_train2)
Y_predict2 = dt.predict(X_test2)
dt_report=classification_report(Y_test2, Y_predict2)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137167
#            1       1.00      1.00      1.00    137102
#            2       1.00      1.00      1.00    137453
#            3       1.00      1.00      1.00    136936
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================
accuracy = accuracy_score(Y_test2, Y_predict2)
precision = precision_score(Y_test2, Y_predict2, average='macro')
recall = recall_score(Y_test2, Y_predict2, average='macro')
f1 = f1_score(Y_test2, Y_predict2, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision (macro): {precision}')
print(f'Recall (macro): {recall}')
print(f'F1 Score (macro): {f1}')
# =============================================================================
# Accuracy: 0.9996482325966267
# Precision (macro): 0.9996482316021333
# Recall (macro): 0.9996482325966267
# F1 Score (macro): 0.9996482320794605
# =============================================================================
# SMOTE를 하지않은 것과 비교했을 때 결과가 약간 더 좋음 

# 2.1.2.2. BayesianOptimization
def DT( max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    model = DecisionTreeClassifier( max_depth=int(round(max_depth)), 
        min_samples_split=int(round(min_samples_split)),  
        min_samples_leaf=int(round(min_samples_leaf)),  
        max_leaf_nodes=int(round(max_leaf_nodes)),
        class_weight='balanced' 
        )
        
    scoring = {'f1_score': make_scorer(f1_score, average='macro')}
    result = cross_validate(model, X_train2, Y_train2, cv=5, scoring=scoring)
    f1_score_mean = result["test_f1_score"].mean()
    return f1_score_mean

pbounds = {
           'max_depth': (2, 20),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 10),
           'max_leaf_nodes': (2, 30),
         
          }

LFBO = BayesianOptimization(f = DT, pbounds = pbounds, verbose = 2, random_state = 0)
LFBO.maximize(init_points=5, n_iter = 20)
LFBO.max
# =============================================================================
# {'target': 0.9973611776455538,
#  'params': {'max_depth': 19.582475455291128,
#   'max_leaf_nodes': 29.817675386915255,
#   'min_samples_leaf': 8.186593973068774,
#   'min_samples_split': 8.34541221331382}}
# =============================================================================

dtc_smote_bay = DecisionTreeClassifier(max_depth= 20, max_leaf_nodes=30, min_samples_leaf= 8,min_samples_split=8, random_state= 0)
dtc_smote_bay.fit(X_train2, Y_train2)
Y_predict2 = dtc_smote_bay.predict(X_test2)
dtc_smote_bay_report=classification_report(Y_test2, Y_predict2)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137167
#            1       1.00      0.99      1.00    137102
#            2       1.00      1.00      1.00    137453
#            3       0.99      1.00      1.00    136936
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================

accuracy_smote_bay = accuracy_score(Y_test2, Y_predict2)
precision_smote_bay = precision_score(Y_test2, Y_predict2, average='macro')
recall_smote_bay = recall_score(Y_test2, Y_predict2, average='macro')
f1_smote_bay = f1_score(Y_test2, Y_predict2, average='macro')
print(f'Accuracy: {accuracy_smote_bay}')
print(f'Precision (macro): {precision_smote_bay}')
print(f'Recall (macro): {recall_smote_bay}')
print(f'F1 Score (macro): {f1_smote_bay}')
# =============================================================================
# Accuracy: 0.9973170900633911
# Precision (macro): 0.9973266705027357
# Recall (macro): 0.9973160184538188
# F1 Score (macro): 0.9973163567039328
# =============================================================================

# 2.1.2.3. BayesianOptimization + GridSearchCV
dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [19,20,21,22],
    'max_leaf_nodes': [28,29,30,31], 
    'min_samples_leaf': [7,8,9,10],  
    'min_samples_split': [7,8,9,10] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 19,
#  'max_leaf_nodes': 31,
#  'min_samples_leaf': 7,
#  'min_samples_split': 7}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [17,18,19,20],
    'max_leaf_nodes': [30,31,32,33], 
    'min_samples_leaf': [5,6,7,8],  
    'min_samples_split': [5,6,7,8] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 17,
#  'max_leaf_nodes': 33,
#  'min_samples_leaf': 5,
#  'min_samples_split': 5}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [12,13,14,15,16,17],
    'max_leaf_nodes': [32,33,34], 
    'min_samples_leaf': [5,6,7],  
    'min_samples_split': [4,5,6] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 12,
#  'max_leaf_nodes': 34,
#  'min_samples_leaf': 5,
#  'min_samples_split': 4}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [10,11,12],
    'max_leaf_nodes': [34,35,36], 
    'min_samples_leaf': [3,4,5],  
    'min_samples_split': [2,3,4] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 10,
#  'max_leaf_nodes': 36,
#  'min_samples_leaf': 3,
#  'min_samples_split': 2}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [7,8,9,10,11],
    'max_leaf_nodes': [36,37,38,39,40,41,42,43], 
    'min_samples_leaf': [1,2,3,4],  
    'min_samples_split': [2,4,8] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 11,
#  'max_leaf_nodes': 43,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [10,11,12,13,14],
    'max_leaf_nodes': [38,39,40,41,42,43,44,45,46,47], 
    'min_samples_leaf': [1,2],  
    'min_samples_split': [2,3] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 11,
#  'max_leaf_nodes': 47,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2}
# =============================================================================

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [10,11,12,13],
    'max_leaf_nodes': range(10,100,5), 
    'min_samples_leaf': [1,2],  
    'min_samples_split': [2,3] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 13,
#  'max_leaf_nodes': 95,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2}
# =============================================================================

pbounds = {
           'max_depth': (2, 30),
           'min_samples_split': (2, 15),
           'min_samples_leaf': (1, 15),
           'max_leaf_nodes': (2, 100),
         
          }

LFBO = BayesianOptimization(f = DT, pbounds = pbounds, verbose = 2, random_state = 0)
LFBO.maximize(init_points=5, n_iter = 20)
LFBO.max
# =============================================================================
# {'target': 0.9991084832068294,
#  'params': {'max_depth': 25.949781250099562,
#   'max_leaf_nodes': 99.82236808587494,
#   'min_samples_leaf': 1.6137477600598753,
#   'min_samples_split': 14.27740326316313}}
# =============================================================================
# 처음 베이지안으로 그리드서치를 돌렸더니 경향이 점점 이상해 짐
# 범위를 조정하여 베이지안 최적화를 한 번 더 시행 하고 다시 그리드 서치

dtc_smote = DecisionTreeClassifier(random_state=0)
param_grid = {
    'max_depth': [24,25,26,27,28],
    'max_leaf_nodes': [98,99,100,101], 
    'min_samples_leaf': [1,2,3],  
    'min_samples_split': [13,14,15] 
}

grid_search = GridSearchCV(estimator=dtc_smote, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train2, Y_train2)

best_params_smote = grid_search.best_params_
# =============================================================================
# {'max_depth': 24,
#  'max_leaf_nodes': 101,
#  'min_samples_leaf': 1,
#  'min_samples_split': 15}
# =============================================================================
# 이것 또한 파라미터 값이 default로 수렴하는 것을 알 수 있음 

dtc_best = DecisionTreeClassifier(**best_params_smote, random_state= 0)
dtc_best.fit(X_train2, Y_train2)

Y_predict2 = dtc_best.predict(X_test2)

dtc_smote_report=classification_report(Y_test2, Y_predict2)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137167
#            1       1.00      0.99      1.00    137102
#            2       1.00      1.00      1.00    137453
#            3       0.99      1.00      1.00    136936
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================

accuracy_smote_bay = accuracy_score(Y_test2, Y_predict2)
precision_smote_bay = precision_score(Y_test2, Y_predict2, average='macro')
recall_smote_bay = recall_score(Y_test2, Y_predict2, average='macro')
f1_smote_bay = f1_score(Y_test2, Y_predict2, average='macro')
print(f'Accuracy: {accuracy_smote_bay}')
print(f'Precision (macro): {precision_smote_bay}')
print(f'Recall (macro): {recall_smote_bay}')
print(f'F1 Score (macro): {f1_smote_bay}')
# =============================================================================
# Accuracy: 0.9991214928060832
# Precision (macro): 0.9991218188666763
# Recall (macro): 0.9991212142017036
# F1 Score (macro): 0.9991209165854962
# =============================================================================

# 2.3.2.4 max_depth 증가에 따른 시간 변화
max_depth_range = list(range(1,51)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in max_depth_range:
    dtc = DecisionTreeClassifier(max_depth=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train2, Y_train2)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict2 = dtc.predict(X_test2)

    accuracy_list.append(accuracy_score(Y_test2, Y_predict2))
    precision_list.append(precision_score(Y_test2, Y_predict2, average='macro'))
    recall_list.append(recall_score(Y_test2, Y_predict2, average='macro'))
    f1_list.append(f1_score(Y_test2, Y_predict2, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(max_depth_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_depth_range, precision_list, marker='x', label='precision')
ax1.plot(max_depth_range, recall_list, marker='s', label='recall')
ax1.plot(max_depth_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_depth_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('DecisionTreeClassifier-max_depth(SMOTE)')
plt.show()

# 2.3.2.5 max_leaf_nodes 증가에 따른 시간 변화
max_leaf_nodes_range = list(range(2,501)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in max_leaf_nodes_range:
    dtc = DecisionTreeClassifier(max_leaf_nodes=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train2, Y_train2)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict2 = dtc.predict(X_test2)

    accuracy_list.append(accuracy_score(Y_test2, Y_predict2))
    precision_list.append(precision_score(Y_test2, Y_predict2, average='macro'))
    recall_list.append(recall_score(Y_test2, Y_predict2, average='macro'))
    f1_list.append(f1_score(Y_test2, Y_predict2, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(max_leaf_nodes_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_leaf_nodes_range, precision_list, marker='x', label='precision')
ax1.plot(max_leaf_nodes_range, recall_list, marker='s', label='recall')
ax1.plot(max_leaf_nodes_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_leaf_nodes')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_leaf_nodes_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('DecisionTreeClassifier-max_leaf_nodes(SMOTE)')
plt.show()
# 그래프 확인 결과 depth = 10, nodes = 30 이 최적값이라고 판단

# 2.1.3. SMOTE + stratify=Y
# 2.1.3.1. default
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train3, Y_train3)
Y_predict3 = dt.predict(X_test3)
dt_report=classification_report(Y_test3, Y_predict3)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137165
#            1       1.00      1.00      1.00    137164
#            2       1.00      1.00      1.00    137165
#            3       1.00      1.00      1.00    137164
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================

accuracy = accuracy_score(Y_test3, Y_predict3)
precision = precision_score(Y_test3, Y_predict3, average='macro')
recall = recall_score(Y_test3, Y_predict3, average='macro')
f1 = f1_score(Y_test3, Y_predict3, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision (macro): {precision}')
print(f'Recall (macro): {recall}')
print(f'F1 Score (macro): {f1}')
# =============================================================================
# Accuracy: 0.9996956209514852
# Precision (macro): 0.9996956381240828
# Recall (macro): 0.9996956210245647
# F1 Score (macro): 0.9996956225423603
# =============================================================================

feature_importances = dt.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6)) # 중요도 그래프
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.show()

# 2.1.3.2 max_depth 증가에 다른 시간 변화
max_depth_range = list(range(1,51)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in max_depth_range:
    dtc = DecisionTreeClassifier(max_depth=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train3, Y_train3)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict3 = dtc.predict(X_test3)

    accuracy_list.append(accuracy_score(Y_test3, Y_predict3))
    precision_list.append(precision_score(Y_test3, Y_predict3, average='macro'))
    recall_list.append(recall_score(Y_test3, Y_predict3, average='macro'))
    f1_list.append(f1_score(Y_test3, Y_predict3, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(max_depth_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_depth_range, precision_list, marker='x', label='precision')
ax1.plot(max_depth_range, recall_list, marker='s', label='recall')
ax1.plot(max_depth_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_depth_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('DecisionTreeClassifier-max_depth(SMOTE+statify)')
plt.show()

# 2.1.3.3 max_leaf_nodes 증가에 따른 시간 변화
max_leaf_nodes_range = list(range(2,501)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in max_leaf_nodes_range:
    dtc = DecisionTreeClassifier(max_leaf_nodes=n, random_state=0)
    
    start_time = time.time()
    dtc.fit(X_train3, Y_train3)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict3 = dtc.predict(X_test3)

    accuracy_list.append(accuracy_score(Y_test3, Y_predict3))
    precision_list.append(precision_score(Y_test3, Y_predict3, average='macro'))
    recall_list.append(recall_score(Y_test3, Y_predict3, average='macro'))
    f1_list.append(f1_score(Y_test3, Y_predict3, average='macro'))

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(max_leaf_nodes_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(max_leaf_nodes_range, precision_list, marker='x', label='precision')
ax1.plot(max_leaf_nodes_range, recall_list, marker='s', label='recall')
ax1.plot(max_leaf_nodes_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('max_leaf_nodes')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(max_leaf_nodes_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('DecisionTreeClassifier-max_leaf_nodes(SMOTE+stratify)')
plt.show()
# 그래프 확인 결과 depth = 10, nodes = 30 이 최적이라고 판단


# 2.2. RandomForest
# 2.2.1. simple random 
# 2.2.1.1. default
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, Y_train)
Y_predict = rf.predict(X_test)
rf_report=classification_report(Y_test, Y_predict)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     24619
#            1       1.00      0.99      1.00       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      1.00      1.00    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================
accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average='weighted')
recall = recall_score(Y_test, Y_predict, average='weighted')
f1 = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy}')
print(f'Precision (weighted): {precision}')
print(f'Recall (weighted): {recall}')
print(f'F1 Score (weighted): {f1}')
# =============================================================================
# Accuracy: 0.9996581154837055
# Precision (weighted): 0.999658254922709
# Recall (weighted): 0.9996581154837055
# F1 Score (weighted): 0.9996579431251379
# =============================================================================

feature_importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6)) # 중요도 그래프
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.show()

# 2.2.1.2. BayesianOptimization 
def RF(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    model = RandomForestClassifier(
        n_estimators=int(round(n_estimators)), 
        max_depth=int(round(max_depth)), 
        min_samples_split=int(round(min_samples_split)),  
        min_samples_leaf=int(round(min_samples_leaf)),  
        max_leaf_nodes=int(round(max_leaf_nodes)), 
        class_weight='balanced', 
    )
        
    scoring = {'f1_score': make_scorer(f1_score, average='weighted')}
    result = cross_validate(model, X_train, Y_train, cv=5, scoring=scoring)
    f1_score_mean = result["test_f1_score"].mean()
    return f1_score_mean

pbounds = {
           'max_depth': (2, 25),
           'max_leaf_nodes': (2, 30),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 10),
           'n_estimators':(10,300)
         
          }

LFBO = BayesianOptimization(f = RF, pbounds = pbounds, verbose = 2, random_state = 0)
LFBO.maximize(init_points=5, n_iter = 20)
LFBO.max
# =============================================================================
# {'target': 0.9988972560192939,
#  'params': {'max_depth': 24.60138591746786,
#   'max_leaf_nodes': 29.99174582442941,
#   'min_samples_leaf': 7.253065928822271,
#   'min_samples_split': 17.68377125034208,
#   'n_estimators': 84.59825333094965}}
# =============================================================================

rfc_bay = RandomForestClassifier(max_depth= 25, max_leaf_nodes=30, 
                                 min_samples_leaf= 7,min_samples_split=18, 
                                 n_estimators=85, random_state= 0)
rfc_bay.fit(X_train, Y_train)
Y_predict = rfc_bay.predict(X_test)
rfc_bay_report=classification_report(Y_test, Y_predict)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      0.99      0.99     24619
#            1       1.00      0.88      0.94       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      0.97      0.98    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================

accuracy_bay = accuracy_score(Y_test, Y_predict)
precision_bay = precision_score(Y_test, Y_predict, average='weighted')
recall_bay = recall_score(Y_test, Y_predict, average='weighted')
f1_bay = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy_bay}')
print(f'Precision (weighted): {precision_bay}')
print(f'Recall (weighted): {recall_bay}')
print(f'F1 Score (weighted): {f1_bay}')
# =============================================================================
# Accuracy: 0.9981135300797324
# Precision (weighted): 0.9981132109885555
# Recall (weighted): 0.9981135300797324
# F1 Score (weighted): 0.9980955674490989
# =============================================================================

# 2.2.1.3. BayesianOptimization + GridSearchCV
rfc = RandomForestClassifier(random_state=0)
param_grid = {
    'max_depth': [23,24,25],
    'max_leaf_nodes': [29,30,31], 
    'min_samples_leaf': [6,7,8],  
    'min_samples_split': [16,17,18] ,
    'n_estimators' : [83,84,85]
}

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_
# =============================================================================
# {'max_depth': 23,
#  'max_leaf_nodes': 31,
#  'min_samples_leaf': 8,
#  'min_samples_split': 16,
#  'n_estimators': 83}
# =============================================================================


rfc = RandomForestClassifier(random_state=0)
param_grid = {
    'max_depth': [21,23,24],
    'max_leaf_nodes': [30,31,33], 
    'min_samples_leaf': [7,8,10],  
    'min_samples_split': [14,16,17] ,
    'n_estimators' : [81,83,84]
}

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, scoring='f1_macro',verbose=2)
grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_

# 2.2.1.4 n_estimators 증가에 따른 시간 변화
n_estimators_range = list(range(1,501,25)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in n_estimators_range:
    rfc = RandomForestClassifier(n_estimators=n, random_state=0)
    
    start_time = time.time()
    rfc.fit(X_train4, Y_train4)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict4 = rfc.predict(X_test4)

    accuracy_list.append(accuracy_score(Y_test4, Y_predict4))
    precision_list.append(precision_score(Y_test4, Y_predict4, average='macro'))
    recall_list.append(recall_score(Y_test4, Y_predict4, average='macro'))
    f1_list.append(f1_score(Y_test4, Y_predict4, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(n_estimators_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(n_estimators_range, precision_list, marker='x', label='precision')
ax1.plot(n_estimators_range, recall_list, marker='s', label='recall')
ax1.plot(n_estimators_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(n_estimators_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('RandomForestClassifier-n_estimators(stratify)')
plt.show()
# 그래프 확인 결과 n_estimators = 75이 최적값이라고 판단

# 2.2.2. SMOTE
# 2.2.2.1. default
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train2, Y_train2)
Y_predict2 = rf.predict(X_test2)
rf_report=classification_report(Y_test2, Y_predict2)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     24619
#            1       1.00      0.99      1.00       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      1.00      1.00    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================
accuracy = accuracy_score(Y_test2, Y_predict2)
precision = precision_score(Y_test2, Y_predict2, average='macro')
recall = recall_score(Y_test2, Y_predict2, average='macro')
f1 = f1_score(Y_test2, Y_predict2, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision (macro): {precision}')
print(f'Recall (macro): {recall}')
print(f'F1 Score (macro): {f1}')
# =============================================================================
# Accuracy: 0.9998505444192921
# Precision (macro): 0.9998503781567152
# Recall (macro): 0.9998505256212572
# F1 Score (macro): 0.9998504206671548
# =============================================================================

# 2.2.2.2. n_estimators 증가에 따른 시간 변화
n_estimators_range = list(range(1,301,50)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in n_estimators_range:
    rfc = RandomForestClassifier(n_estimators=n, random_state=0)
    
    start_time = time.time()
    rfc.fit(X_train2, Y_train2)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict2 = rfc.predict(X_test2)

    accuracy_list.append(accuracy_score(Y_test2, Y_predict2))
    precision_list.append(precision_score(Y_test2, Y_predict2, average='macro'))
    recall_list.append(recall_score(Y_test2, Y_predict2, average='macro'))
    f1_list.append(f1_score(Y_test2, Y_predict2, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(n_estimators_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(n_estimators_range, precision_list, marker='x', label='precision')
ax1.plot(n_estimators_range, recall_list, marker='s', label='recall')
ax1.plot(n_estimators_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(n_estimators_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('RandomForestClassifier-n_estimators(SMOTE)')
plt.show()
# 그래프 확인 결과 n_estimators = 200이 최적값이라고 판단

# 2.2.3. SMOTE + stratify=Y
# 2.2.3.1. default
rf = RandomForestClassifier(random_state=0)
start_time=time.time()
rf.fit(X_train3, Y_train3)
end_time=time.time()
times.append(end_time-start_time) # 796.6209037303925 = 13.27분
Y_predict3 = rf.predict(X_test3)
rf_report=classification_report(Y_test3, Y_predict3)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137165
#            1       1.00      1.00      1.00    137164
#            2       1.00      1.00      1.00    137165
#            3       1.00      1.00      1.00    137164
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================
accuracy = accuracy_score(Y_test3, Y_predict3)
precision = precision_score(Y_test3, Y_predict3, average='macro')
recall = recall_score(Y_test3, Y_predict3, average='macro')
f1 = f1_score(Y_test3, Y_predict3, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision (macro): {precision}')
print(f'Recall (macro): {recall}')
print(f'F1 Score (macro): {f1}')
# =============================================================================
# Accuracy: 0.9998541896773582
# Precision (macro): 0.9998542639558456
# Recall (macro): 0.9998541901158573
# F1 Score (macro): 0.9998541915678478
# =============================================================================

rf = RandomForestClassifier(n_estimators=150,random_state=0)
rf.fit(X_train3, Y_train3)
accuracy = accuracy_score(Y_test3, Y_predict3)
precision = precision_score(Y_test3, Y_predict3, average='macro')
recall = recall_score(Y_test3, Y_predict3, average='macro')
f1 = f1_score(Y_test3, Y_predict3, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision (macro): {precision}')
print(f'Recall (macro): {recall}')
print(f'F1 Score (macro): {f1}')
# =============================================================================
# Accuracy: 0.9996956209514852
# Precision (macro): 0.9996956381240828
# Recall (macro): 0.9996956210245647
# F1 Score (macro): 0.9996956225423603
# =============================================================================

# 2.2.3.2 n_estimators 증가에 따른 시간 변화 
n_estimators_range = list(range(1,301,50)) 
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
times = []

for n in n_estimators_range:
    rfc = RandomForestClassifier(n_estimators=n, random_state=0)
    
    start_time = time.time()
    rfc.fit(X_train3, Y_train3)
    end_time = time.time()
    times.append(end_time - start_time)
    
    Y_predict3 = rfc.predict(X_test3)

    accuracy_list.append(accuracy_score(Y_test3, Y_predict3))
    precision_list.append(precision_score(Y_test3, Y_predict3, average='macro'))
    recall_list.append(recall_score(Y_test3, Y_predict3, average='macro'))
    f1_list.append(f1_score(Y_test3, Y_predict3, average='macro'))


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(n_estimators_range, accuracy_list, marker='o', label='accuracy')
ax1.plot(n_estimators_range, precision_list, marker='x', label='precision')
ax1.plot(n_estimators_range, recall_list, marker='s', label='recall')
ax1.plot(n_estimators_range, f1_list, marker='d', label='f1')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(n_estimators_range, times, marker='.', label='time', color='black')
ax2.set_ylabel('Time(s)', color='black')
ax2.tick_params(axis='y', labelcolor='black')


plt.title('RandomForestClassifier-n_estimators(SMOTE+statify)')
plt.show()
# 그래프 확인 결과 n_estimators = 150이 최적값이라고 판단


# 2.3 knn
# 2.3.1. simple random
# 2.3.1.1. default
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
knn_report=classification_report(Y_test, Y_predict)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     24619
#            1       1.00      0.99      1.00       720
#            2       1.00      1.00      1.00      1211
#            3       1.00      1.00      1.00    137248
# 
#     accuracy                           1.00    163798
#    macro avg       1.00      1.00      1.00    163798
# weighted avg       1.00      1.00      1.00    163798
# =============================================================================

accuracy_bay = accuracy_score(Y_test, Y_predict)
precision_bay = precision_score(Y_test, Y_predict, average='weighted')
recall_bay = recall_score(Y_test, Y_predict, average='weighted')
f1_bay = f1_score(Y_test, Y_predict, average='weighted')
print(f'Accuracy: {accuracy_bay}')
print(f'Precision (weighted): {precision_bay}')
print(f'Recall (weighted): {recall_bay}')
print(f'F1 Score (weighted): {f1_bay}')
# =============================================================================
# Accuracy: 0.9996031697578726
# Precision (weighted): 0.9996032172788104
# Recall (weighted): 0.9996031697578726
# F1 Score (weighted): 0.9996030073373164
# =============================================================================

# 2.3.1.2. n_neighbors 증가에 따른 score 변화 (weights='distance') 
n_neighbors_range = list(range(1, 51, 4))  
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for n in n_neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    knn.fit(X_train, Y_train)
    Y_predict = knn.predict(X_test)

    accuracy_list.append(accuracy_score(Y_test, Y_predict))
    precision_list.append(precision_score(Y_test, Y_predict, average='macro'))
    recall_list.append(recall_score(Y_test, Y_predict, average='macro'))
    f1_list.append(f1_score(Y_test, Y_predict, average='macro'))


plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_range, accuracy_list,  marker='o',label = 'accuracy')
plt.plot(n_neighbors_range, precision_list, marker='x', label = 'precision')
plt.plot(n_neighbors_range, recall_list, marker='s', label = 'recall')
plt.plot(n_neighbors_range, f1_list, marker='d', label = 'f1')
plt.xlabel('n_neighbors')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
# 그래프 확인 결과 n_neighbors = 5이 최적값이라고 판단

# 2.3.2. SMOTE
# 2.3.2.1. default
knn = KNeighborsClassifier()
knn.fit(X_train2, Y_train2)
Y_predict2 = knn.predict(X_test2)
knn_report=classification_report(Y_test2, Y_predict2)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137167
#            1       1.00      1.00      1.00    137102
#            2       1.00      1.00      1.00    137453
#            3       1.00      1.00      1.00    136936
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================
accuracy_bay = accuracy_score(Y_test2, Y_predict2)
precision_bay = precision_score(Y_test2, Y_predict2, average='macro')
recall_bay = recall_score(Y_test2, Y_predict2, average='macro')
f1_bay = f1_score(Y_test2, Y_predict2, average='macro')
print(f'Accuracy: {accuracy_bay}')
print(f'Precision (macro): {precision_bay}')
print(f'Recall (macro): {recall_bay}')
print(f'F1 Score (macro: {f1_bay}')
# =============================================================================
# Accuracy: 0.9998013334354006
# Precision (macro): 0.9998011304614616
# Recall (macro): 0.9998012632665538
# F1 Score (macro: 0.9998011640211277
# =============================================================================

# 2.3.2.2. n_neighbors 증가에 따른 score 변화 (weights='distance')
n_neighbors_smote_range = list(range(1, 51, 4))  
accuracy_smote_list = []
precision_smote_list = []
recall_smote_list = []
f1_smote_list = []

for n in n_neighbors_smote_range:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    knn.fit(X_train2, Y_train2)
    Y_predict2 = knn.predict(X_test2)

    accuracy_smote_list.append(accuracy_score(Y_test2, Y_predict2))
    precision_smote_list.append(precision_score(Y_test2, Y_predict2, average='macro'))
    recall_smote_list.append(recall_score(Y_test2, Y_predict2, average='macro'))
    f1_smote_list.append(f1_score(Y_test2, Y_predict2, average='macro'))


plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_smote_range, accuracy_smote_list,  marker='o',label = 'accuracy')
plt.plot(n_neighbors_smote_range, precision_smote_list, marker='x', label = 'precision')
plt.plot(n_neighbors_smote_range, recall_smote_list, marker='s', label = 'recall')
plt.plot(n_neighbors_smote_range, f1_smote_list, marker='d', label = 'f1')
plt.xlabel('n_neighbors')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
# 그래프 확인 결과 n_neighbors = 17이 최적값이라고 판단

# 2.3.3. SMOTE + stratify=Y
# 2.3.3.1. default
knn = KNeighborsClassifier()
knn.fit(X_train3, Y_train3)
Y_predict3 = knn.predict(X_test3)
knn_report=classification_report(Y_test3, Y_predict3)
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    137165
#            1       1.00      1.00      1.00    137164
#            2       1.00      1.00      1.00    137165
#            3       1.00      1.00      1.00    137164
# 
#     accuracy                           1.00    548658
#    macro avg       1.00      1.00      1.00    548658
# weighted avg       1.00      1.00      1.00    548658
# =============================================================================
accuracy_bay = accuracy_score(Y_test3, Y_predict3)
precision_bay = precision_score(Y_test3, Y_predict3, average='macro')
recall_bay = recall_score(Y_test3, Y_predict3, average='macro')
f1_bay = f1_score(Y_test3, Y_predict3, average='macro')
print(f'Accuracy: {accuracy_bay}')
print(f'Precision (macro): {precision_bay}')
print(f'Recall (macro): {recall_bay}')
print(f'F1 Score (macro: {f1_bay}')
# =============================================================================
# Accuracy: 0.9998323181289619
# Precision (macro): 0.9998323483596729
# Recall (macro): 0.9998323183947179
# F1 Score (macro: 0.9998323165697844
# =============================================================================

# 2.3.3.2 n_neighbors 증가에 따른 score 변화 (weights = 'distance')
n_neighbors_stratify_range = list(range(1, 51, 4))  
accuracy_stratify_list = []
precision_stratify_list = []
recall_stratify_list = []
f1_stratify_list = []

for n in n_neighbors_stratify_range:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    knn.fit(X_train3, Y_train3)
    Y_predict3 = knn.predict(X_test3)

    accuracy_stratify_list.append(accuracy_score(Y_test3, Y_predict3))
    precision_stratify_list.append(precision_score(Y_test3, Y_predict3, average='macro'))
    recall_stratify_list.append(recall_score(Y_test3, Y_predict3, average='macro'))
    f1_stratify_list.append(f1_score(Y_test3, Y_predict3, average='macro'))


plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_stratify_range, accuracy_stratify_list,  marker='o', label = 'accuracy')
plt.plot(n_neighbors_stratify_range, precision_stratify_list, marker='x', label = 'precision')
plt.plot(n_neighbors_stratify_range, recall_stratify_list, marker='s', label = 'recall')
plt.plot(n_neighbors_stratify_range, f1_stratify_list, marker='d',  label = 'f1')
plt.xlabel('n_neighbors')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
# 그래프 확인 결과 n_neighbors = 25이 최적값이라고 판단