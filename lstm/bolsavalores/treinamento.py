from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/petr4_treinamento.csv')
df = df.dropna()
df_treinamento = df.iloc[:, 1:2].values

normalizador = MinMaxScaler(feature_range=(0,1))
df_treinamento = normalizador.fit_transform(df_treinamento)

previsores = []
preco_real = []

for i in range(90, 1242):
    previsores.append(df_treinamento[i-90:i, 0])
    preco_real.append(df_treinamento[i, 0])

previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 100, return_sequences=True, input_shape = (previsores.shape[1], 1)))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50))
model.add(Dropout(0.3))

model.add(Dense(units=1, activation = 'linear'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
model.fit(previsores, preco_real, epochs = 100, batch_size=32, workers=6)

df_teste = pd.read_csv('data/petr4_teste.csv')
preco_real_teste = df_teste.iloc[:, 1:2].values
df_completa = pd.concat((df['Open'], df_teste['Open']), axis=0)
entradas = df_completa[len(df_completa) - len(df_teste) -90:].values
entradas = entradas.reshape(-1,1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = model.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()