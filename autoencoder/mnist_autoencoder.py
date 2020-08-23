import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense

(X_treinamento, _), (X_teste, _) = mnist.load_data()
X_treinamento = X_treinamento.astype('float32') / 255
X_teste = X_teste.astype('float32') / 255 

X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))


#Self supervised learning

autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim=784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(X_treinamento, X_treinamento,
                epochs=50,  batch_size = 256,
                validation_data=(X_teste, X_teste))

dimensao = Input(shape=(784,))
encode_layer = autoencoder.layers[0]
encoder = Model(dimensao, encode_layer(dimensao))
encoder.summary()

codificada = encoder.predict(X_teste)
decodificada = autoencoder.predict(X_teste)

numero_imagens = 10
imagens_teste = np.random.randint(X_teste.shape[0], size= numero_imagens)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    eixo = plt.subplot(10, 10, i+1)
    plt.imshow(X_teste[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
    
    eixo = plt.subplot(10, 10, i+1+ numero_imagens * 2)
    plt.imshow(codificada[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())

    eixo = plt.subplot(10, 10, i+1+numero_imagens * 2)
    plt.imshow(decodificada[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())    
    

