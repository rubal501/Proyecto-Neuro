

module MNIST
using CSV
using DelimitedFiles
 
function vectorizar(valor)
    vector =  zeros(Int64, 10)
    vector[valor+1] = valor
    return vector 
end

function obtener_datos()
    datos_entrenamiento = readdlm("data/mnist_train.csv", ',',Int64)
    datos_prueba = readdlm("data/mnist_test.csv", ',',Int64)
    arreglo_entrenamiento =  Array{Tuple{Array{Int64,1},Array{Int64,1}},1}()
    #println("llegue")
    arreglo_prueba = Array{Tuple{Array{Int64,1},Int64},1}()
    #println("llegue2")
    for i in 1:60000
        push!(arreglo_entrenamiento, ([datos_entrenamiento[i,n] for n in 2:785], vectorizar(datos_entrenamiento[i,1])))
        #Estas lineas son para debuggear
        #println("-----------------------------------")
        #println(arreglo_entrenamiento)
    end
    
    for j in 1:10000
        push!(arreglo_prueba, ([datos_prueba[j,n] for n in 2:785], datos_prueba[j,1]))
        #Estas lineas son para debuggear
        #println("-----------------------------------")
        #println(arreglo_prueba)
    end
    return (arreglo_entrenamiento, arreglo_prueba)
end

end # module
