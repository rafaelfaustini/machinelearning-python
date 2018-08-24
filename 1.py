#Tem escamas ?
#Tem dentes alinhados ?
#É acinzentado


jacare1 = [1, 0, 1]
jacare2 = [1, 0, 0]
jacare3 = [1, 0, 0]

crocodilo1 = [1, 1, 1]
crocodilo2 = [1, 1, 0]
crocodilo3 = [1, 1, 1]

dados= [jacare1,jacare2,jacare3,crocodilo1,crocodilo2,crocodilo3]
marcacoes = [1,1,1,-1,-1,-1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados,marcacoes)

descobrir1 = [1,1,1] #Crocodilo (-1)
descobrir2 = [1,0,0] #Jacare (1)
descobrir3 = [1,0,1] #Jacare (1)

marcacoes_teste = [-1,1,1]

descobrir = [descobrir1,descobrir2,descobrir3]



res = modelo.predict(descobrir)
for item in res:
    if(item == -1):
        print("Jacaré")
    if(item == 1):
        print("Crocodilo")


diffs = res - marcacoes_teste
acertos = [d for d in diffs if d==0]
print(str((len(descobrir)/len(acertos))*100.)+" %")

