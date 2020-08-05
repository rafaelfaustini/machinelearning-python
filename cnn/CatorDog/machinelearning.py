import numpy as np
from keras.models import model_from_json
import base64
import uuid 
from keras.preprocessing import image


def loadModel():
    arquivo = open('ml-data/model_catdog.json', 'r')
    estrutura_rede = arquivo.read()
    arquivo.close()
    
    model = model_from_json(estrutura_rede)
    model.load_weights('ml-data/model_catdog.h5')
    
    return model

def runModel(obj=None):
    imagem = base64.b64decode(obj['Image'])
    id = uuid.uuid1()
    filename = 'tests/'+str(id)+'.png'
    try:
        reader = open(filename, 'wb')
        reader.write(imagem)
        reader.close()
        
        imagem = image.load_img(filename, target_size = (64,64))
        imagem = image.img_to_array(imagem)
        imagem /= 255
        imagem = np.expand_dims(imagem, axis=0)
    except:
        print("Erro no processamento de imagem")
    
    model = loadModel()
    previsao = model.predict(imagem)
    resultado = {
        True: "Gato",
        False: "Cachorro"
    }
    previsao = [previsao > 0.5]
    return resultado[previsao[0][0][0]]
