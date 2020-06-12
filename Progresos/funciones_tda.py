import numpy as np
import pandas as pd
import gudhi as gd  
from sklearn import manifold
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
import csv
import plotly.graph_objects as go
from gudhi.representations import  BettiCurve

#esta funcion se encarga de generar las matrices de correlaciones entre neuronas de 
#red previamente entrenadada 

def generate_matrix(route, n_neuronas):
    b = np.zeros((n_neuronas,n_neuronas), dtype = float)
    with open(route, newline='') as csvfile:
        datos = list(csv.reader(csvfile))
        for i in range(0,n_neuronas):
            for j in range(0,n_neuronas):
                b[i][j] = np.linalg.norm(np.array(datos[i], dtype='float64') - np.array(datos[j], dtype='float64') )
    return b 

def generate_tree(epoch):
    route = "dataexp/epoch"+str(epoch)+ ".csv"
    #print("leido")
    #TODO Corregir este hardcodeo 
    matrix = generate_matrix(route,108)
    skeleton  = gd.RipsComplex(distance_matrix = matrix )
    #print("esqueletolisto")

    Rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    #print("listo")
    return Rips_simplex_tree


def generate_persistence_diagram(per,epoch):
    nombre = "diagram_epoch"+str(epoch)+ ".png"
    gd.plot_persistence_diagram(per).figure.savefig(nombre, dpi=170)
    gd.plot_persistence_diagram(per).figure.clf()

def generate_betticurve(pers,dim):
    
    for i in range(0,len(pers)):
        arr = pers[i].persistence_intervals_in_dimension(dim)
        diags = [arr]        
        #diags = DiagramSelector(use=True, point_type="finite").fit_transform(diags)
        #diags = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diags)
        #diags = DiagramScaler(use=True, scalers=[([1], Clamping(maximum=.9))]).fit_transform(diags)

        BC = BettiCurve(resolution=1000)
        bc = BC.fit_transform(diags)
        etiqueta = 'epoch ' +str(i+1)
        plt.plot(bc[0],label = etiqueta )

    #plt.xlim(left=100) 
    plt.title("Curva de betti de dimension " + str(dim) )
    plt.legend()
    nombre =  "curvabetti.png"
    #plt.xscale('log')
    #plt.yscale('log')
    plt.savefig(nombre, dpi=300)
    plt.clf()
    #print("curva obtenida")

def get_betti_number(PI_0):
    diags = [PI_0]        
    BC = BettiCurve(resolution=1000)
    bc = BC.fit_transform(diags)
    res = bc[0]
    return res 

def clear_interval(PI_0):
    res = []
    for element in PI_0:
        if np.inf != element[1]:
            res.append(element)
    return res 

def get_persistences(PI_0):
    #La siguiente funcion recibe los inervalos de persistencia de dimesion n y calcula
    #la distancia entre el nacimiento y la muerte de la carateristica topologica
    res = []
    for interval in PI_0:
        res.append(np.abs(interval[0] - interval[1]))
    return np.array(res)

def analyse_persistence(dim):
    lista_avgs = []
    lista_desviaciones = []
    lista_avgs_betti = []
    lista_desviaciones_betti = []
    cont = []
    cotree = []
    for epoch in range(0,18):
        tree = generate_tree(epoch)
        tree.persistence()
        PI_0 = tree.persistence_intervals_in_dimension(dim)
        PI_0_refined = clear_interval(PI_0)
        pers = get_persistences(PI_0_refined)
        lista_avgs.append(np.average(pers))
        lista_desviaciones.append(np.std(pers))
        res = get_betti_number(PI_0)
        lista_avgs_betti.append(np.average(res))
        lista_desviaciones_betti.append(np.std(res))


    final = [ np.array(lista_avgs), np.array(lista_desviaciones), np.array(lista_avgs_betti), np.array(lista_desviaciones_betti)]
    return final

def graph_persistences(persistencias, epoch):
    x = [i for i in range(0,epoch)]

    upper_bound = go.Scatter(
    name='Upper Bound',
    x=x,
    y= persistencias[0] + persistencias[1] ,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty')

    trace = go.Scatter(
    name='Measurement',
    x=x,
    y=persistencias[0],
    mode='lines',
    line=dict(color='rgb(31, 119, 180)'),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty')

    lower_bound = go.Scatter(
    name='Lower Bound',
    x=x,
    y=persistencias[0]- persistencias[1] ,
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines')

    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
    yaxis=dict(title='persistencia'),
    xaxis=dict(title='epoch'),
    title='Persistencia a traves de las epochs',
    showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    fig.show()


def graph_betti(bettis, epoch):
    x = [i for i in range(0,epoch)]

    upper_bound = go.Scatter(
    name='Upper Bound',
    x=x,
    y= bettis[0] + bettis[1] ,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty')

    trace = go.Scatter(
    name='Measurement',
    x=x,
    y=bettis[0],
    mode='lines',
    line=dict(color='rgb(31, 119, 180)'),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty')

    lower_bound = go.Scatter(
    name='Lower Bound',
    x=x,
    y= bettis[0]- bettis[1] ,
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines')

    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
    yaxis=dict(title='betti-1'),
    xaxis=dict(title='epoch'),
    title='valor del numero de betti-1 a traves de las epochs',
    showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    fig.show()


