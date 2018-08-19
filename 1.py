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

descobrir1 = [1,1,1]
descobrir2 = [1,0,0]
descobrir3 = [1,0,1]

descobrir = [descobrir1,descobrir2,descobrir3]

res = modelo.predict(descobrir)
print(res)
