import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time

LOG_DIR = f"{int(time.time())}"

def criarRede(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int("input_units", min_value=32, max_value=256, step=32), (3,3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    for i in range(hp.Int("n_Convlayers", 1, 5)):
            model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32), (3,3), input_shape = (64, 64, 3), activation = 'relu'))
            model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    

    model.add(Flatten())

    
    for i in range(hp.Int("n_Denselayers", 1, 5)):
        model.add(
            Dense(units =  hp.Int(f'dense_{i}_units', min_value=128,max_value=512,step=32,default=128),
                        activation = hp.Choice("dense_activation", values=['relu', 'tanh', 'softmax'], default='relu') ))
        model.add(Dropout(hp.Float(f"Dropout_Dense_{i}", 0.1, 0.6)))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    
    model.compile(optimizer = hp.Choice("optimizer", values=['adam', 'sgd', 'rmsprop'], default='adam'), 
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                             rotation_range = 7,
                                             horizontal_flip = True,
                                             shear_range = 0.2,
                                             height_shift_range = 0.09,
                                             zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

    
base_treinamento = gerador_treinamento.flow_from_directory('../dataset/training_set',
                                                               target_size = (64, 64),
                                                               batch_size = 32,
                                                               class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('../dataset/test_set',
                                                   target_size = (64, 64),
                                                   batch_size = 32,
                                                   class_mode = 'binary')

tuner = RandomSearch(criarRede, objective="val_accuracy",
                     max_trials= 40,
                     executions_per_trial=2,
                     directory = LOG_DIR)
tuner.search(base_treinamento, steps_per_epoch= 4000/ 16,
                        epochs = 50, validation_data = base_teste,
                        validation_steps = 1000 / 16)

import pickle

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)

#tuner = pickle.load(open("tuner_1576720506.pkl", "rb"))

print(tuner.get_best_hyperparameters()[0].values)
        
                 

