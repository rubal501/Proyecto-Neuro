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




def get_points(id_exp,epoch, metodo):
    #Este metodo obtiene los puntos de nuestro espacio a estudiar 
    route = "data/experimental/experimento_"+str(id_exp)+"/activaciones/vectores_"+str(metodo)+"_epoch"+str(epoch)+".csv"
    b = [] #Lista de puntos
    with open(route, newline='') as csvfile:
        datos = list(csv.reader(csvfile))
        for element in datos:
            b.append(np.array(element,dtype='float64'))
    return b 

def generate_tree(id_exp, epoch,metodo):
    print("leido")
    #TODO Corregir este hardcodeo 
    points = get_points(id_exp,epoch,metodo)
    skeleton  = gd.AlphaComplex(points)
    print("esqueletolisto")

    cech_simplex_tree = skeleton.create_simplex_tree()
    print("listo")
    return cech_simplex_tree


def generate_barcode(per,id_exp,metodo,epoch):
    nombre = "images/barr" +str(metodo) + "_exp_" + str(id_exp)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_barcode(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_barcode(per).figure.clf()
    

def generate_persistence_diagram(per,id_exp,metodo,epoch):
    nombre = "images/diagram" +str(metodo) + "_exp_" + str(id_exp)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_diagram(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_diagram(per).figure.clf()

 
tamanos = [[784,30,10],[784, 50, 50, 50, 10], [784, 300, 10], [784, 12, 12, 12, 12, 12, 10]] 
epochs = [7,7,7,57]
nombres = [ "experimento_" + str(i) for i in [1,2,3,4] ]


for i in range(0, len(tamanos)):
    for e in range(1,epochs[i]+1):
        tree = generate_tree(nombres[i], e , "betta" )
        tree.compute_persistence() # IDEA Puedo agilizar este proceso seleccionando un campo diferente ? 
        generate_barcode(tree.persistence_intervals_in_dimension(1), nombres[i], "betta",e)
        generate_persistence_diagram(tree.persistence_intervals_in_dimension(1), nombres[i], "betta",e)


