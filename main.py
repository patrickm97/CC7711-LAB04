import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

lista = []

for n in range(10):
    regr = MLPRegressor(hidden_layer_sizes=(70),
                        max_iter=30000,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=1000)
    print('Treinando RNA')
    regr = regr.fit(x,y)

    print('Preditor')
    y_est = regr.predict(x)

    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)
    lista.append(regr.loss_curve_[-1])
    
    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)

    plt.show()

for idx, item in enumerate(lista):
    if idx == len(lista)-1:
        print("%.3f" % float(item))
    else:
        print("%.3f" % float(item), end=" + ")

#for item in lista:
#    print("%.3f" % float(item), end=" +")

print()
print("media: %.3f" % np.average(lista))
print("desvio padrao: %.3f" % np.std(lista))
