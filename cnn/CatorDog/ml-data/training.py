import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
    
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(units =  352, activation = 'relu'))
model.add(Dropout(0.51))
model.add(Dense(units = 384, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 448, activation = 'relu'))
model.add(Dropout(0.49))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.43))
model.add(Dense(units = 352, activation = 'relu'))
model.add(Dropout(0.47))
model.add(Dense(units = 1, activation = 'sigmoid'))
    
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
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
h =  model.fit_generator(base_treinamento, steps_per_epoch= 12002/ 32,
                        epochs = 55, validation_data = base_teste,
                        validation_steps = 12002/ 32, workers=6)

y_real_cachorro, y_real_gato = np.split(base_teste.classes,2)

y_pred = model.predict(base_teste, 12002 // 33)
y_pred = (y_pred > 0.5).astype(int).reshape(12002,1)

y_pred_cachorro, y_pred_gato = np.split(y_pred, 2)

cm_cachorro = confusion_matrix(y_real_cachorro, y_pred_cachorro)  
cm_gato = confusion_matrix(y_real_gato, y_pred_gato)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm=cm_cachorro,classes=["Cachorro"], title="Matriz de confusão Cachorro") 
plot_confusion_matrix(cm=cm_gato, classes=["Gato"], title="Matriz de confusão Gato") 
           
           
    
    # Save Model Settings
model_json = model.to_json()
with open('model_catdog.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_catdog.h5')



plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Acurácia do modelo')
plt.ylabel('acurácia')
plt.xlabel('época')
plt.legend(['Treino', 'Teste'], loc='best')
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Perda do modelo')
plt.ylabel('perda')
plt.xlabel('época')
plt.legend(['train', 'test'], loc='best')
plt.show()