## Aqui se crea bonito la red
cd(dirname(@__FILE__))

include("funciones_red.jl")

# Vector de tamanos de cada capa
tamanos = [3,5,2]
# Crea la red
red = crea_red(tamanos)

# ejemplo de feedforward
vect= [2,4,5]
resultado = feedforward(red,vect)
