from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

df = pd.read_csv('data/petr4_treinamento.csv')
df = df.dropna()
df_treinamento_nn = df.iloc[:, 1:7].values

normalizador = MinMaxScaler(feature_range=(0,1))
df_treinamento = normalizador.fit_transform(df_treinamento_nn)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(df_treinamento_nn[:,0:1])

previsores = []
preco_real = []

for i in range(90, 1242):
    previsores.append(df_treinamento[i-90:i, 0:6])
    preco_real.append(df_treinamento[i, 0])

previsores, preco_real = np.array(previsores), np.array(preco_real)

model = Sequential()
model.add(LSTM(units = 100, return_sequences=True, input_shape = (previsores.shape[1], 6)))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50))
model.add(Dropout(0.3))

model.add(Dense(units=1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor="loss", min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = "loss", factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath= 'data/pesos.h5', monitor = 'loss',
                      save_best_only = True)

model.fit(previsores, preco_real, epochs = 100, batch_size = 32,
          callbacks= [es, rlr, mcp])


df_teste = pd.read_csv('data/petr4_teste.csv')
preco_real_teste = df_teste.iloc[:, 1:2].values
frames = [df, df_teste]
df_completa = pd.concat(frames)
df_completa = df_completa.drop('Date', axis = 1)


entradas = df_completa[len(df_completa) - len(df_teste) - 90:].values
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = model.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()