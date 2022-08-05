from cProfile import label
from cmath import sqrt
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arima_clarknet = pd.read_csv('data/resource_manager_data/ts1_test_arima.csv')
prophet_clarknet = pd.read_csv('data/resource_manager_data/ts1_test_prophet.csv')
arima_nasa = pd.read_csv('estimator/ts2_test_arima.csv')
prophet_nasa = pd.read_csv('estimator/ts2_test_prophet.csv')
plt.plot(arima_nasa.request_count.values, label = 'true value')
#plt.plot(arima_clarknet.y.values, label = 'true value')
#plt.plot(predictions[0:], color='red')
#print(predictions)

# plt.plot(xlabel = 'datetime',ylabel ='request count', label='request count')
# plt.plot(arima_clarknet.y_pred.values[1:], color='red', label = 'arima',marker="o",markevery=5)
# plt.plot(prophet_clarknet.y_pred.values[1:], color='green', label = 'prophet',marker="*",markevery=5)

plt.plot(xlabel = 'datetime',ylabel ='request count', label='request count')
plt.plot(arima_nasa.y_pred.values[1:], color='red', label = 'arima',marker="o",markevery=5)
plt.plot(prophet_nasa.y_pred.values[1:], color='green', label = 'prophet',marker="*",markevery=5)

plt.subplots_adjust(wspace=5)
plt.legend(loc = 'best')
plt.xlabel("episode")
plt.ylabel('request_count')
plt.show()   