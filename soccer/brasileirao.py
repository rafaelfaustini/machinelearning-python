import pandas as pd
import os
from math import abs

path = os.getcwd()
file = '{0}\datasets_269569_560136_Tabela_Clubes.csv'.format(path)
df = pd.read_csv(file, decimal=",")


df = df.drop(columns=df.loc[:,'Unnamed: 13':'Unnamed: 16'])
gols = df['GolsF/S'].str.split(':', n=1, expand = True)
df["GolsF"] = gols[0]
df["GolsS"] = gols[1]
df = df.drop(columns=["GolsF/S", "Clubes", "Ano"])

X = df.loc[:, 'Vitorias':'GolsS']
y = df.loc[:, 'Pos.']

X['GolsF'] = pd.to_numeric(X['GolsF'], errors='coerce')
X['GolsS'] = pd.to_numeric(X['GolsS'], errors='coerce')
y = abs((y - y.max()) / (y.min() - y.max())).to_frame().astype('float32')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def criarRede(optimizer, loss, kernel_initializer, activation, neurons, dropout):
    with tf.device('/device:GPU:0'):
        model = Sequential()
        model.add(Dense(units = neurons, kernel_initializer=kernel_initializer, activation=activation,
                        input_dim=11))
        model.add(Dropout(dropout))
        model.add(Dense(units = neurons, kernel_initializer=kernel_initializer, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(units = 1, activation = 'sigmoid'))
        model.compile(optimizer= optimizer, loss=loss, metrics=['mean_absolute_error'])
    return model


model = KerasRegressor(build_fn=criarRede)
parametros = {
    'batch_size': [10, 30],
    'epochs': [150],
    'optimizer': ['adam', 'sgd'],
    'loss': ['mean_absolute_error'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [6, 12, 24, 48],
    'dropout': [0.2, 0.4]
}

grid_search = GridSearchCV(estimator=model, param_grid=parametros, scoring='neg_mean_absolute_error', cv=4)
grid_search = grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

