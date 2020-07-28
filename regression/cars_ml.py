import pandas as pd

df = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
df = df.drop('dateCrawled', axis = 1)
df = df.drop('dateCreated', axis = 1)
df = df.drop('nrOfPictures', axis = 1)
df = df.drop('postalCode', axis = 1)
df = df.drop('lastSeen', axis = 1)

# Pouca variabilidade
df['name'].value_counts()
df = df.drop('name', axis = 1)
df['seller'].value_counts()
df = df.drop('seller', axis = 1)
df ['offerType'].value_counts()
df = df.drop('offerType', axis = 1)

# Remove registros de preço inválido
df = df.loc[df.price > 100]

df = df.loc[df.price < 500000]

# Valores faltantes
df['vehicleType'].value_counts()
#df = df.loc[pd.isnull(df['vehicleType'])]

df['gearbox'].value_counts()
#df = df.loc[pd.isnull(df['gearbox'])]

df['model'].value_counts()
#df = df.loc[pd.isnull(df['model'])]

df['fuelType'].value_counts()
#df = df.loc[pd.isnull(df['fuelType'])]

df['notRepairedDamage'].value_counts()
#df = df.loc[pd.isnull(df['notRepairedDamage'])]

fix = {'vehicleType': 'limousine', 'gearbox': 'manuell', 
       'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
df = df.fillna(value=fix)

X = df.iloc[:, 1:13].values
y = df.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#Encode de strings para int. Ex: 0 para 'a', 1 para 'b'...
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])

# Encode de string por quantidade de categorias sendo a,b,c (0,0,1)
onehotencoder = ColumnTransformer(transformers=[("OneHot", 
OneHotEncoder(), 
[0,1,3,5,8,9,10])],
remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

with tf.device('/device:GPU:0'):
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss='mean_absolute_error', optimizer = 'adam', 
                      metrics = ['mean_absolute_error'])
    regressor.fit(X, y, batch_size = 300, epochs = 100)
    
    previsoes = regressor.predict(X)
