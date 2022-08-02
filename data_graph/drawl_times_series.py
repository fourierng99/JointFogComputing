from cProfile import label
from cmath import sqrt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def column_chart1(strs):
    x=[i for i in range(1,101)]
    labels=["Combine fuzzy and deep q 1","deep q learning 1","fuzzy","random"]
    fig, ax = plt.subplots()
    files=pd.read_csv("result/DQN1/n_quality_tasks_DQN_s4.csv")[0:100]
    m=files["good"]+files["medium"]+files["bad"]
    print(m)
    print([np.mean(files["good"]/m)]*len(x))
    ax.plot(x,[np.mean(files["good"]/m)]*len(x), label='Mean good',color="violet", linestyle='--')
    ax.plot(x,[np.mean(files["medium"]/m)]*len(x), label='Mean medium',color="y", linestyle='--')
    ax.plot(x,[np.mean(files["bad"]/m)]*len(x), label='Mean bad',color="gray", linestyle='--')
    ax.plot(x,files["good"]/m,label="Good",color="violet",marker='*',markevery=5)
    ax.plot(x,files["medium"]/m,marker='^',label="Medium",color="y",markevery=5)
    #ax1.plot(x,files["bus2"],":",label="vehicular fog 2")
    ax.plot(x,files["bad"]/m,label="Bad",color="gray",markevery=5)
    ax.set_xlabel("Time slots",fontsize=15)
    ax.set_ylabel("Ratio",fontsize=15)
    ax.set_ylim(0,1)
    #ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=6)
    plt.grid(alpha=0.5)
    plt.show()

    #plt.savefig("fdqoqualitywithtimeslots.eps")

ts1_train = pd.read_csv("data/resource_manager_data/ts1_train.csv")
ts1_test_arima = pd.read_csv('data/resource_manager_data/ts1_test_arima.csv')
ts1_test_prophet = pd.read_csv('data/resource_manager_data/ts1_test_prophet.csv')

ts2_train = pd.read_csv('data/resource_manager_data/ts2_train.csv')
ts2_test_arima = pd.read_csv('data/resource_manager_data/ts1_test_prophet.csv')
ts2_test_prophet = pd.read_csv('data/resource_manager_data/ts2_test_prophet.csv')

#df = pd.read_csv("estimator/data_est/nasa_data_request_count_5min.csv", names =["datetime", "request_count"],index_col='datetime')
#df['datetime'] = pd.to_datetime(df['datetime'])
#df2 = pd.read_csv('estimator\data_est/nasa_data_request_count_filter.csv')
dfx = pd.read_csv('estimator\data_est/nasa_train_test.csv',index_col='datetime')
#mae = mean_absolute_error(test_val, predictions)
#mape = mean_absolute_percentage_error(y_true=test_val, y_pred=predictions)
#rmse = math.sqrt(mean_squared_error(test_val, predictions))
#print('MAE: %.3f' % mae)
#print('RMSE: %.3f' % rmse)
#print(mape)
#export data
#self.test_data['y_pred'] = predictions
#dpath = "estimator/ts{}_test_{}.csv".format(self.tdata, self.tmodel)
#self.test_data.to_csv(dpath,index=None)
#print(dpath)
#series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# dfx.plot(ylabel ='request count')
# plt.axvline(x = 401, color = 'violet', label = 'train-test bound',linestyle='dotted')
# #plt.plot(['7/5/1995 8:00']*20,[200]*1*20,color="violet", linestyle='--')
# plt.legend(loc = 'best')
# plt.show()   

df_clark = pd.read_csv('estimator\data_est/nasa_train_test.csv',index_col='datetime')
#df_clark.rename(columns = {'ds':'datetime', 'y':'request_count'}, inplace = True)
df_clark.plot(xlabel = 'datetime',ylabel ='request count', label='request count')
plt.subplots_adjust(wspace=5)
plt.axvline(x = 401, color = 'violet', label = 'train-test bound',linestyle='dotted')
#plt.plot(['7/5/1995 8:00']*20,[200]*1*20,color="violet", linestyle='--')
plt.legend(loc = 'best')
plt.show()   


# df_clark = pd.read_csv('estimator\data_est/clarknet_train_test.csv',index_col='ds')
# df_clark.rename(columns = {'ds':'datetime', 'y':'request_count'}, inplace = True)
# df_clark.plot(xlabel = 'datetime',ylabel ='request count', label='request count')
# plt.subplots_adjust(wspace=5)
# plt.axvline(x = 401, color = 'violet', label = 'train-test bound',linestyle='dotted')
# #plt.plot(['7/5/1995 8:00']*20,[200]*1*20,color="violet", linestyle='--')
# plt.legend(loc = 'best')
# plt.show()   
#plt.savefig("data_graph/fig/test.pdf")