cd(dirname(@__FILE__))


include("funciones_red.jl")
include("training_data.jl")

# Vector de tamanos de cada capa
tamanos = [784,30,10]
# Crea la red
red = crea_red(tamanos)

## Entrenamiento

epochs = 1 #Numero de épocas
mini_batch_size = 20
η = 1.00 #training rate
test_data = true


# Con test_data
red = SGD(red, training_data, epochs, mini_batch_size, η, test_data, validation_data)

## Despues de ejecutar te daras cuenta de que Sí consigue entrenar


