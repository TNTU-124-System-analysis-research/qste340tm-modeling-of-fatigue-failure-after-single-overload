# Import python libraries required in this example:
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

fname = 'fig3.xlsx'
df = pd.read_excel(fname)
df_header = df.iloc[1:3]
print(df_header.head())
row_one = df_header.iloc[0]
print(row_one)
R_lst = []
Amp_lst = []
for column in row_one.items():
    print(column)
    R = column[1][2:5]
    R_lst.append(float(R))
    Amp = column[1][6:]
    Amp_lst.append(Amp)
print(R_lst)
print(Amp_lst)
#exit(0)
df_filtered = df.iloc[3:]
print(df_filtered.head())
print(df_filtered.shape)
print(df_filtered.describe())
print(df_filtered.info())
print(df_filtered.describe(include='all'))
df_train  = []
i = 1
df_test = []
for col_number in range(0,len(list(df_filtered.columns)),2):
    print(col_number)
    df_one  = df_filtered.iloc[:,col_number : col_number+2]
    print(df_one)
    df_one.dropna(axis=0,how='any', inplace=True)
    R =  R_lst[col_number]
    df_one['R'] = R
    if 'CA' in Amp_lst[col_number]:
        #continue
        df_one['Amp'] = 0
    else:
        #df_one['Amp'] = Amp_lst[col_number].split('=')[1]
        continue
    lst_cols = list(df_one.columns)
    print(str(lst_cols))
    df_one.rename(columns={lst_cols[0]:'cycles'}, inplace=True)
    df_one.rename(columns={lst_cols[1]: 'crack length'}, inplace=True)
    i += 1

    if R == 0.5:
        #df_test.append(df_one)
        df_test.append(df_one)
    else:
        df_train.append(df_one)
#print(df_train)
#for data_set in df_lst:
df_train = pd.concat(df_train)
df_test = pd.concat(df_test)
#all = df_train.values
#X = all[:,[0,2]]
#y = all[:,[1]]
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = .8)
X_train = df_train.iloc[:,0:2].values
y_train = df_train.iloc[:,1].values
X_test = df_test.iloc[:,0:2].values
y_test = df_test.iloc[:,1].values
#df_test = pd.concat(df_test)
#print('df_train', df_train)
#print('df_test', df_test)
#print(df_test)
#train = df_train.values
#test = df_test.values
#X_test = test[:,[0,2]]
#y_test = test[:,[1]]
#X_train = train[:,[0,2]]
#y_train = train[:,[1]]
print(X_test, y_test)
print(X_train, y_train)
X_test.reshape(-1, 2)
y_test.reshape(-1, 1)
X_train.reshape(-1, 2)
y_train.reshape(-1, 1)
print(X_test, y_test)
print(X_train, y_train)
print('Shape', y_test.shape, X_test.shape)
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
X_train_scaled = scaler.fit_transform(X_train)
#y_train_scaled = scaler.fit_transform(y_train)
X_test_scaled = scaler.transform(X_test)
#y_test_scaled = scaler.fit_transform(y_test)
'''
# Use numpy arrays to store inputs (x) and outputs (y):
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]]
'''
#Define the network model and its arguments.
#Set the number of neurons/nodes for each layer:
model = Sequential()
model.add(Dense(2, input_shape=(2,), activation = 'relu'))#, kernel_initializer='he_uniform'))
#model.add(Dense(10, activation = 'relu'))
model.add(Dense(15, activation = 'leaky_relu'))
model.add(Dense(10, activation = 'leaky_relu'))
#model.add(Dense(30, activation = 'relu'))
#model.add(Activation('relu'))
#model.add(Dense(1))
model.add(Dense(1, activation = 'relu'))
#model.add(Activation('linear'))
# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Print a summary of the Keras model:
model.summary()
#callback = keras.callbacks.EarlyStopping(monitor='loss',
#                                              patience=3)
callback = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    X_train,#_scaled,
    y_train,
    epochs=2000,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2, callbacks=[callback])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()
plot_loss(history)

print('X_test_scaled shape:', X_test_scaled.shape)
y_pred = model.predict(X_test_scaled)
print(y_pred)
print(y_pred.shape)
print(y_test.shape)
#test_pred = scaler.inverse_transform(test_predictions)
#print(test_pred)
#y_test = scaler.inverse_transform(y_test_scaled).reshape(-1,1)
print(y_test)
a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [mm]')
plt.ylabel('Predictions [mm]')
lims = [10, 20]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

# Fit the regressor with x and y data
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [mm]')
plt.ylabel('Predictions [mm]')
lims = [10, 20]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=12)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [mm]')
plt.ylabel('Predictions [mm]')
lims = [10, 20]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
from sklearn import svm
model = svm.SVR()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [mm]')
plt.ylabel('Predictions [mm]')
lims = [10, 20]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()



plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], y_pred)
plt.show()