import numpy as np
from keras.models import model_from_json

def loadModel():
    arquivo = open('ml-data/model_hepatitis.json', 'r')
    estrutura_rede = arquivo.read()
    arquivo.close()
    
    model = model_from_json(estrutura_rede)
    model.load_weights('ml-data/model_hepatitis.h5')
    
    return model

def runModel(obj=None):
    registro = np.array([[
        obj['Age'], obj['Sex'], obj['Steroid'], obj['Antivirals'], obj['Fatigue'],
         obj['Malaise'], obj['Anorexia'], obj['LiverBig'], obj['LiverFirm'], obj['SpleenPalpable'],
         obj['Spiders'], obj['Ascites'], obj['Varices'], obj['Bilirubin'], obj['AlkPhosphate'],
         obj['Sgot'], obj['Albumin'], obj['Protime'], obj['Histology'], 
         ]])
    model = loadModel()
    previsao = model.predict(registro)
    previsao = [previsao > 0.5]
    return int(previsao[0][0][0])
