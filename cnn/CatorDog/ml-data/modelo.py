import time as ts
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 # Transferencia de aprendizado
from keras.callbacks import EarlyStopping
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.layers import Flatten, Dense


class Modelo:
    def __init__(self, batch_size=64, image_size=256):
        self.batch_size = batch_size
        self.image_size = image_size
        self.model = self.Modelo()
        self.h = None
        
    def criar(self):
        try:
            model = VGG16(include_top=False, input_shape=(self.image_size, self.image_size, 3))
            # Camadas preenchidas não podem ser treinadas
            for layer in model.layers:
                layer.trainable = False
                
            flat1 = Flatten()(model.layers[-1].output)
            class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
            output = Dense(1, activation='sigmoid')(class1)
        	# define new model
            model = Model(inputs=model.inputs, outputs=output)
        	# compile model
            opt = SGD(lr=0.001, momentum=0.9)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as erro:
            raise Exception('Houve um erro ao treinar a rede neural: %s'%(repr(erro)))
    
    def treinar(self, debug=True, verbose=1, epochs=50):
        try:
            model = self.model   
            gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                                         rotation_range = 7,
                                                         horizontal_flip = True,
                                                         shear_range = 0.2,
                                                         height_shift_range = 0.09,
                                                         zoom_range = 0.2)
            gerador_teste = ImageDataGenerator(rescale = 1./255)
                
            base_treinamento = gerador_treinamento.flow_from_directory('../dataset/training_set',
                                                                           target_size = (self.image_size, self.image_size),
                                                                           batch_size = self.batch_size,
                                                                           class_mode = 'binary')
            base_teste = gerador_teste.flow_from_directory('../dataset/test_set',
                                                               target_size = (self.image_size, self.image_size),
                                                               batch_size = self.batch_size,
                                                               class_mode = 'binary')
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=10)
            self.h =  model.fit(base_treinamento, steps_per_epoch= len(base_treinamento),
                                    epochs = epochs, validation_data = base_teste,
                                    validation_steps = len(base_teste), workers=4, callbacks=[es], verbose=verbose)
            
            _, acc = model.evaluate(base_teste, steps=len(base_teste), verbose=0)
            
            if debug == True:
                print('%.3f % de acurácia'%(acc * 100))
                
        except Exception as erro:
            raise Exception('Houve um erro ao treinar a rede neural: %s'%(repr(erro)))
            
    def carregar(self, caminho, nome='model_catdog'):
        try:
            with open('%s/%s.json'%(caminho,nome), 'r') as arquivo:
                estrutura_rede = arquivo.read()
            model = model_from_json(estrutura_rede)
            model.load_weights('%s/%s.h5'%(caminho,nome))
            self.model = model
        except Exception as erro:
            raise Exception('Houve um erro ao carregar a rede neural: %s'%(repr(erro)))            
            
    def salvar(self):
        try:
            model = self.model
            model_json = model.to_json()
            t = int(ts.time())
            with open('model_catdog-%d.json'%(t), 'w') as json_file:
                json_file.write(model_json)
            model.save_weights('model_catdog-%d.h5'%(t))
        except Exception as erro:
            raise Exception('Houve um erro ao salvar a rede neural: %s'%(repr(erro)))
                        
    def plotar(self, acuracia=True, perda=True):
        try:
            h = self.h
            if acuracia == True:
                plt.plot(h.history['accuracy'])
                plt.plot(h.history['val_accuracy'])
                plt.title('Acurácia do modelo')
                plt.ylabel('acurácia')
                plt.xlabel('época')
                plt.legend(['Treino', 'Teste'], loc='best')
                plt.show()
            if perda == True:    
                plt.plot(h.history['loss'])
                plt.plot(h.history['val_loss'])
                plt.title('Perda do modelo')
                plt.ylabel('perda')
                plt.xlabel('época')
                plt.legend(['train', 'test'], loc='best')
                plt.show()
        except Exception as erro:
            raise Exception('Houve um erro ao plotar o gráfico: %s'%(repr(erro)))