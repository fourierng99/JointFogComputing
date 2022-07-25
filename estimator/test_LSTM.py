###LSTM on Stock data

##Build a model that predicts the stock prices of a company based on the ##prices of the previous few days

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

df = pd.read_csv('estimator\clarknet_train_test.csv')

df.head()

length_split = int(len(df)* 0.8)
train_data = df[:length_split]
test_data = df[length_split:]

# creating an array with closing prices
trainingd = train_data.y.values


sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(trainingd.reshape(-1,1))

##x_train stores the values of closing prices of past 45(or as specified in ##timestamp) days

##y_train stores the values of closing prices of the present day

x_train = []
y_train = []
timestamp = 10
length = len(trainingd)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)



print (x_train[0])
print ('\n')
print (y_train[0])

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape



model = Sequential() #define the Keras model

model.add(LSTM(units = 120, return_sequences = True, input_shape = (x_train.shape[1], 1))) #120 neurons in the hidden layer
##return_sequences=True makes LSTM layer to return the full history including outputs at all times
model.add(Dropout(0.2))

model.add(LSTM(units = 120, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, return_sequences = False)) 
model.add(Dropout(0.2))

model.add(Dense(units = 1)) #output
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 25, batch_size = 32)

#test
test_set = test_data.y.values
test_set_scaled = sc.fit_transform(test_set.reshape(-1,1))

x_test = []
y_test = test_set[timestamp:]
length = len(test_set)
for i in range(timestamp, length):
    x_test.append(test_set_scaled[i-timestamp:i, 0])
    
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_pred = model.predict(x_test)
predictions = sc.inverse_transform(y_pred)

plt.plot(y_test, color = 'blue', label = "actual count")
plt.plot(predictions, color = 'red', label = "predicted count")
plt.title("Request Count")
plt.xlabel('time')
plt.ylabel("C")
plt.legend()
plt.show()
