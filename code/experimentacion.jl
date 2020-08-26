# experimento
using Blink

include("funciones_red.jl")
include("func_DistMat.jl")
include("func_experimenta.jl")
include("training_data.jl")

# Quiero crear un objeto experimento que contenga los 4
# puntos considerados abajo

tamanos = [[784,30,10],[784, 50, 50, 50, 10], [784, 300, 10], [784, 12, 12, 12, 12, 12, 10]] 

nombres = [ "experimento_" * string(i) for i in 1:length(tamanos) ]
#Quiero ir calculando los pesos que voy a analizar despues
for element in 1:length(tamanos)
    exp = experimento_simple(tamanos[element], 95, nombres[element], max_ejec = 60)

end

#exp_plotbettivsepochs(e, 1)
#exp_plotpersisvsepoch(experimento, 1)
