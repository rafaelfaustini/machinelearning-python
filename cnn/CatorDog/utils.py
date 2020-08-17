import os
import random
import base64
import shutil


def getRandomTest():
    arquivos = os.listdir("tests\\")
    if not arquivos:
        return []
    nomearquivo = random.choice(arquivos)
    if not nomearquivo:
        return []
    filedir = "tests\{}".format(nomearquivo)
    try:
        with open(filedir, "rb") as image:
            image = str(base64.b64encode(image.read()))
        result = {
            "filename": nomearquivo,
            "imagem": image
        }
        return result
    except:
        print("RandomTest Error")
        return []
    
def runAction(d, a):
    if a == -1:
        os.remove("tests\{}".format(d))
    elif a == 0:
        l = len(os.listdir("dataset\\training_set\\cachorro"))
        shutil.move("tests\{}".format(d), "dataset\\training_set\\cachorro\\dog.{}.jpg".format(l))  
    elif a == 1:
        l = len(os.listdir("dataset\\training_set\\gato"))
        shutil.move("tests\{}".format(d), "dataset\\training_set\\gato\\cat.{}.jpg".format(l))
    
def validateRandomTest(obj=None):
    diretorio = "tests\{}".format(obj["name"])
    if os.path.isfile(diretorio):
        runAction(obj["name"],obj["action"])
        return 1
    return 0