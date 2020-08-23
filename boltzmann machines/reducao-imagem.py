import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
X = np.asarray(base.data, 'float32')
y = base.target

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

X, X_teste, y, y_teste = train_test_split(X, y,test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.n_iter = 25
rbm.n_components = 50
naive_rbm = GaussianNB()

model_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
model_rbm.fit(X, y)

plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r )
    plt.xticks(())
    plt.yticks(())
plt.show()

previsoes = model_rbm.predict(X_teste)
precisao = metrics.accuracy_score(previsoes, y_teste)

naive_simples = GaussianNB()
naive_simples.fit(X, y)
previsoes_naive = naive_simples.predict(X_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, y_teste)