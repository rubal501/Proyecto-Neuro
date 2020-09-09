import numpy as np
import pandas as pd
import gudhi as gd  
from sklearn import manifold
import matplotlib.pyplot as plt
from pylab import *
#from skbio.stats.distance import DissimilarityMatrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
import csv
from gudhi.representations import DiagramSelector, Clamping, Landscape, Silhouette, BettiCurve, ComplexPolynomial,\
  TopologicalVector, DiagramScaler, BirthPersistenceTransform,\
  PersistenceImage, PersistenceWeightedGaussianKernel, Entropy, \
  PersistenceScaleSpaceKernel, SlicedWassersteinDistance,\
  SlicedWassersteinKernel, PersistenceFisherKernel

#esta funcion se encarga de generar las matrices de correlaciones entre neuronas de 
#red previamente entrenadada 

# TODO limpiar este codigo, tiene mucho hardcodeo
#TODO vectorizar 




def get_points(id_exp,epoch, metodo,digito, muestras):
    #Este metodo obtiene los vectores generados con vectores.ipynb 
    route = "data/experimental/experimento_"+str(id_exp)+"/activaciones/vectores_"+str(metodo)+"_epoch"+str(epoch)+".csv"
    b = [] #Lista de puntos
    with open(route, newline='') as csvfile:
        datos = list(csv.reader(csvfile))
        for element in datos[(digito*muestras):muestras + (digito*muestras)]:
            b.append(np.array(element,dtype='float64'))
    return b 

def generate_tree(id_exp, epoch,metodo,digito, muestras):
    #Este metodo permite crear un simplex tree usando 
    print("leido")
    #TODO Corregir este hardcodeo 
    data = get_points(id_exp,epoch,metodo,digito, muestras)
    skeleton  = gd.RipsComplex(points=data)
    print("esqueleto listo")

    alpha_simplex_tree = skeleton.create_simplex_tree(max_dimension = 2)
    print("listo")
    return alpha_simplex_tree


def generate_barcode(per,id_exp,metodo,epoch):
    nombre = "images/barr" +str(metodo) + "_exp_" + str(id_exp)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_barcode(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_barcode(per).figure.clf()
    

def generate_persistence_diagram(per,id_exp,metodo,epoch):
    nombre = "images/diagram" +str(metodo) + "_exp_" + str(id_exp)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_diagram(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_diagram(per).figure.clf()

def average_persistence(per_diagram, per_dim):
    persistencias = []
    for feature in per_diagram:
        dimension = feature[0]
        if dimension == per_dim:
            persistencias.append(feature)

    per_array = np.array(persistencias)

    return per_array.mean()
 

tamanos = [[784,30,10],[784, 50, 50, 50, 10], [784, 300, 10], [784, 12, 12, 12, 12, 12, 10]] 
epochs = [7,7,7,57]
nombres = [ "experimento_" + str(i) for i in [1,2,3,4] ]

"""
for i in range(0, len(tamanos)):
    for e in range(1,epochs[i]+1):
        tree = generate_tree(nombres[i], e , "betta" )
        pre = tree.persistence()


        # IDEA Puedo agilizar este proceso seleccionando un campo diferente ? 
        generate_barcode(pre, nombres[i], "betta",e)
        generate_persistence_diagram(pre, nombres[i], "betta",e)
"""
epocas_finales = [[7,7,7,7,7],[16,16,32,51,44,19,17,34,20,24]]


tamano_muestra = 10
for digito in range(0,10):
    for familia in range(1,5):
        promedios = []
        for miembro in range(1,11):
            name = "familia_" + str(familia)+"experimento_"+str(miembro)
            tree = generate_tree(name, epocas_finales[familia-1][miembro-1] , "betta", digito, tamano_muestra )
            per = tree.persistence()
            nombre_salvado = "data/persistencias/per_fam"+ str(familia)+"miembro"+str(miembro) + ".pers"
            tree.write_persistence_diagram(nombre_salvado)
            promedios.append(average_persistence(per, 0))

        data = np.array(promedios)
        fig1, ax1 = plt.subplots()
        titulo = "Persistencias promedio de dimension 0 arquitectura "+ str(familia) + "digito" + str(digito)
        ax1.set_title(titulo)
        ax1.boxplot(data)
        ruta_guardado = "images/box_pers_avg_dim0_fam" + str(familia)  + "_digito_" + str(digito) + ".png"
        fig1.savefig(ruta_guardado)


        





