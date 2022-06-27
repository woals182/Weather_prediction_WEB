import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('data/2017.csv',encoding='euc-kr')
df2 = pd.read_csv('data/2018.csv',encoding='euc-kr')
df3 = pd.read_csv('data/2018_2019.csv',encoding='euc-kr')
df4 = pd.read_csv('data/2019_2020.csv',encoding='euc-kr')
df5 = pd.read_csv('data/2020_2021.csv',encoding='euc-kr')
df6 = pd.read_csv('data/2021_2022.csv',encoding='euc-kr')

df1_c = df1.copy()
df2_c = df2.copy()
df3_c = df3.copy()
df4_c = df4.copy()
df5_c = df5.copy()
df6_c = df6.copy()

df1_fillna = df1_c.fillna(0)
df2_fillna = df2_c.fillna(0)
df3_fillna = df3_c.fillna(0)
df4_fillna = df4_c.fillna(0)
df5_fillna = df5_c.fillna(0)
df6_fillna = df6_c.fillna(0)

seoul1 = df1_fillna[df1_fillna['지점']==108]
seoul2 = df2_fillna[df2_fillna['지점']==108]
seoul3 = df3_fillna[df3_fillna['지점']==108]
seoul4 = df4_fillna[df4_fillna['지점']==108]
seoul5 = df5_fillna[df5_fillna['지점']==108]
seoul6 = df6_fillna[df6_fillna['지점']==108]

Seoul = pd.concat([seoul1,seoul2,seoul3,seoul4,seoul5,seoul6])

Seoul_d=Seoul[['지점명','일시','기온(°C)','강수량(mm)','풍속(m/s)','풍향(16방위)','습도(%)','지면온도(°C)','현지기압(hPa)']]

Seoul_d=Seoul_d.reset_index()

Seoul_dr = Seoul_d.drop(labels = 'index', axis=1)

Seoul_dr['doy'] = None
Seoul_dr['hour'] = None
Seoul_dr['year'] = None

Seoul_dr['datetime']= pd.to_datetime(Seoul_dr['일시'])

for i in Seoul_dr.index:
    Seoul_dr['doy'][i]=pd.Period(Seoul_dr['datetime'][i],freq='D')
    Seoul_dr['doy'][i]=Seoul_dr['datetime'][i].day_of_year

for i in Seoul_dr.index:
    Seoul_dr['year'][i]=pd.Period(Seoul_dr['datetime'][i],freq='D')
    Seoul_dr['year'][i]=Seoul_dr['datetime'][i].year

for i in Seoul_dr.index:
    Seoul_dr['hour'][i]=pd.Period(Seoul_dr['datetime'][i],freq='D')
    Seoul_dr['hour'][i]=Seoul_dr['datetime'][i].hour

Seoul_df=Seoul_dr[['기온(°C)','강수량(mm)','풍속(m/s)','풍향(16방위)','습도(%)','현지기압(hPa)','doy','year','hour','datetime']]

Seoul_dfs=Seoul_df.set_index('datetime',drop=True,append=False,inplace=False)

target=['기온(°C)','강수량(mm)','풍속(m/s)','풍향(16방위)','습도(%)','현지기압(hPa)']
X = Seoul_dfs.drop(target,axis=1)
condition = (X['year'] > 2016) & (X['year'] <= 2021)
condition1 = (X['year'] > 2021)
y = Seoul_dfs[target]

X_train = X[condition]
X_test = X[condition1]
y_train = y[condition]
y_test = y[condition1]

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

def is_leap_year(year):
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0
    
def doy(M,D):
    if is_leap_year(2022):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N

def prediction(M,D,hour):
    N=doy(M,D)
    y_pred = clf.predict([[N,2022,hour]])
    
    pred = f"""2022년 {M}월{D}일의 {hour}시의 날씨 
    {round(float(y_pred[0][0]),2)} : 기온(°C),
    {round(float(y_pred[0][1]),2)} : 강수량(mm),
    {round(float(y_pred[0][2]),2)} : 풍속(m/s),
    {round(float(y_pred[0][3]),2)} : 풍향(16방위),
    {round(float(y_pred[0][4]),2)} : 습도(%),
    {round(float(y_pred[0][5]),2)} : 현지기압(hPa)"""

    return pred

    