from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)

#Recomendação de filmes
# Terror Terror Terror Comédia Comédia Comédia

base = np.array([[1,1,1,0,0,0],
                [1,0,1,0,0,0],
                [1,1,1,0,0,0],
                [0,0,1,1,1,1],
                [0,0,1,1,0,1],
                [0,0,1,1,0,1]])
rbm.train(base, max_epochs=5000)

filmes = ["A bruxa", "Invocação do mal", "O chamado",
          "Se beber não case", "Gente grande", "American pie"]


rbm.weights

usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])


rbm.run_visible(usuario1)
rbm.run_visible(usuario2)

hidden = np.array([[1,0]])
recomendacao = rbm.run_hidden(hidden)

for i in range(len(usuario1[0])):
    if usuario1[0, i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])