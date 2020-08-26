from modelo import Modelo

Rede = Modelo(batch_size=64, image_size=256)
Rede.treinar()
Rede.salvar()
Rede.plotar()



