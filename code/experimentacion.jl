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
#experimento = experimento_simple(tamanos[1], 95, nombres[1], max_ejec = 60)
#red_salvada = recuperar_red(tamanos[1], nombres[1],60)
#experimento = experimento_preentrenado(red_salvada, "alpha", norm, 20, tamanos[1], 95)
#Quiero ir calculando los pesos que voy a analizar despues

for element in 1:length(tamanos)
    exp = experimento_simple(tamanos[element], 95, nombres[element], max_ejec = 80)

end

#exp_plotbettivsepochs(e, 1)
#exp_plotpersisvsepoch(experimento, 1)
