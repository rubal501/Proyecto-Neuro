using CSV
using DelimitedFiles
using StatsBase
using DataFrames


function obtener_muestras()
    datos_prueba = readdlm("data/fashion-mnist_test.csv", ',',Int16)
    println(typeof(datos_prueba))
    println(datos_prueba[1])
    for e in 0:9
        println(e)
        entradas = []
        contador = 1
        for indice in 1:10000
            if datos_prueba[indice][1] == e
                #println(datos_prueba[indice])
                push!(entradas, indice)
            end
            
        end
        println("el tamano del vector es ", length(entradas) )
        println(datos_prueba[1,2:785] )
        
        #println(entradas)

        indices_muestras = sample(1:length(entradas), 10, replace = false)

        for i in indices_muestras
            io = open("muestras.csv", "a")
            println("meti al csv este dato  ", i)   
            tmp = datos_prueba[i,2:785]
            writedlm( io, transpose(tmp), ',')
            close(io)  
        end
        

    end
end

obtener_muestras()


 # module
