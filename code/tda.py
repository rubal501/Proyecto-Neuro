import numpy as np
import pandas as pd
import gudhi as gd  
from sklearn import manifold
import matplotlib.pyplot as plt
from pylab import *
from skbio.stats.distance import DissimilarityMatrix
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

def get_points(route, dim_vectores):
    b = np.zeros((dim_vectores,dim_vectores), dtype = float)
    with open(route, newline='') as csvfile:
        datos = list(csv.reader(csvfile))
        for i in range(0,dim_vectores):
            for j in range(0,dim_vectores):
                b[i][j] = np.linalg.norm(np.array(datos[i], dtype='float64') - np.array(datos[j], dtype='float64') )

    return b 

def generate_tree(epoch,route, dim_vectores):
    print("leido")
    #TODO Corregir este hardcodeo 
    points = get_points(route,dim_vectores)
    skeleton  = gd.AlphaComplex(points)
    print("esqueletolisto")

    cech_simplex_tree = skeleton.create_simplex_tree(max_dimension=1)
    print("listo")
    return cech_simplex_tree


def generate_barcode_diagram(per,route,epoch):
    nombre = "/content/gdrive/My Drive/MatApli2020-2/Proyectos/TDA/images/exp"+str(experiment)+"/bar_exp" + str(experiment)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_barcode(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_barcode(per).figure.clf()
    

def generate_persistence_diagram(per,rou,epoch):
    nombre = "/content/gdrive/My Drive/MatApli2020-2/Proyectos/TDA/images/exp"+str(experiment)+"/diagram_exp" + str(experiment)+ "epoch"+str(epoch)+ ".png"
    gd.plot_persistence_diagram(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_diagram(per).figure.clf()