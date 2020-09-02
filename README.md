# Proyecto-Neuro bitacora
## Cosas por hacer:
- ~~Implementar entropia cruzada como funcion de perdida~~ FALTA CHECAR QUE ESTO ESTE BIEN
- ~~Implementar una funcion que permita que el valor de eta tenga una decaimiento exponencial al lo largo del entrenamoiento ~~
- Implementar el metodo que menciono raziel para construir los nuevos vectores 

## Nuevas funciones del codigo
### Funcion de decaimeniento exponencial
La funcion utilizada en la ultima version del codigo `code/func_experimenta` esta dada de la siguiente forma:
$$f(p) = 10 \exp(-(0.085)p) $$
donde p es el desempeno de la red en alguna epoca del entrenamiento.

### Funcion de finalizado automatico.
Para evitar que un sobre entrenamiento se creo un buffer $B$ que almacena los desempenos de la red en las ultimas 7 epocas de
entrenamiento. En caso de que se cumpla la siguiente desigualdad en alguna epoca:
$$ \| \max(B) - \min(B) \| \leq 1  $$
El entrenamiento es terminado inmediatamente y se borran la informacion de las ultimas 6 epocas donde la red estuvo estancada.

## Experimentos:
### Entrenamientos
Se entrenaron 4 redes de diferentes arquitecturas con el objetivo de llegar a un 95% de desempeno, el entrenamiento fue 
limitado a un maximo de 60 epocas.

| id                        | arquitectura            | desempeno logrado (%) | epocas de entrenamiento |
|---------------------------|-------------------------|-----------------------|-------------------------|
| experimento_experimento_1 | [784,30,10]             | 92.19                 | 7                       |
| experimento_experimento_2 | [784,50,50,50,10]       | 91.72                 | 7                       |
| experimento_experimento_3 | [784,300,10]            | 93.89                 | 7                       |
| experimento_experimento_4 | [784,12,12,12,12,12,10] | 69.0                  | 57                      |

Los pesos y sesgos de cada una de las redes a lo largo del experimento fueron guardados en archivos csv almacenados en la carpeta de `Proyecto-Neuro/data/experimental/experimento_experimento_x/activaciones` de cada experimento.
Tambien a lo largo del entrenamiento cada experimento guardo informacion como el desempeno en cada una de las epocas en cada
uno de los archivos `log.txt`.
### Generacion de vectores de activacion

Los  alpha vectores fueron generados con el metodo usado en el proyecto final en el curso de matematicas aplicadas.
Los betta vectores fueron generados con el nuevo metodo, los cuales tienen como entradas la activacion de cada una de 
las neuronas dado un elemento de la muestra.
### Analisis de datos
Hasta este momento solo se ha probado con una muestra de 20 elementos, es decir 2 elementos de cada clase. El calculo de la 
homologia persistente se ha logrado en tiempo aceptable en los dos primeros experimentos, sin embargo en los ultimos tarda mucho tiempo. Los diagramas de persistencia y de barras generados hasta este punto pueden ser consultados aqui: [link a la graficas](https://photos.app.goo.gl/kzDA4EHaDGGHY5cAA)




 
