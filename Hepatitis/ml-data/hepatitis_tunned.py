
import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
labels = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites','Varices','Bilirubin','AlkPhosphate','Sgot', 'Albumin', 'Protime', 'Histology']
df = pd.read_csv(url, 
                 names=labels, header=None)

cols = df.columns[df.dtypes.eq('object')]
for col in cols:
  df[col] = pd.to_numeric(df[col], errors='coerce')
  if df[col].dtypes == 'int64':
    avg = round(df[col].mean())
    df[col] = df[col].fillna(avg)
  else:
    avg = df[col].mean()
    df[col] = df[col].fillna(avg)
    
df["Class"] = df["Class"].replace(to_replace=1, value=0)
df["Class"] = df["Class"].replace(to_replace=2, value=1)


x = df.loc[:, 'Age':]
y = df.loc[:,'Class']



import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def criarRede():
  with tf.device('/device:GPU:0'):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', kernel_initializer='random_uniform', input_dim = 19))
    model.add(Dropout(0.3))   
    model.add(Dense(units=32, activation='relu', kernel_initializer='random_uniform'))  
    model.add(Dropout(0.3))   
    model.add(Dense(units=1, activation= 'sigmoid'))
    model.compile(optimizer='adam', loss='poisson',
                        metrics = ['accuracy'])
  return model

model = criarRede()
h = model.fit(x, y, validation_split=0.25, batch_size=10, epochs=500)

# Save Model Settings
model_json = model.to_json()
with open('model_hepatitis.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_hepatitis.h5')

import matplotlib.pyplot as plt

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Acurácia do modelo')
plt.ylabel('acurácia')
plt.xlabel('época')
plt.legend(['Treino', 'Teste'], loc='lower right')
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Perda do modelo')
plt.ylabel('perda')
plt.xlabel('época')
plt.legend(['train', 'test'], loc='lower right')
plt.show()