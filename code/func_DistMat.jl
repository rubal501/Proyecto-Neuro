
# Saca una muestra de training data
# come el tamaño de  la muestra deseada

function find_index(n)
    return findall(x -> findmax(x)[2] == n, [training_data[i][2] for i in 1:50000] )
end


# size debe dividir a 50,000 y ser mayor a 10
function muestrea( size )
    # hace un shuffle a los datos para que
    # las muestras sean aleatorias
    shuffle!(training_data)
    # encuentra los indices de mi muestra
    indices = []
    for i ∈ 1:10
        append!(indices, find_index(i)[1:convert(Int,size/10)])
    end

    muestra = []
    # llena la muestra usando los indices
    for i in indices
        push!(muestra, training_data[i] )
    end
    return muestra
end

# Calcula los vectores que representan neuronas
function α_neuronas(Γ::Red_Neu, muestra, output::Bool=false)
    neuronas = 0
    tabla = []
    # Hace feedforward a cada uno de los datos
    for dato in muestra
        x, y = dato
        # calcula el vector de activaciones
        zs, acts = Γ(x, true)
        if output
            neuronas = sum(Γ.tamanos) - Γ.tamanos[1]
            acts = acts[2:end]
        else
            neuronas = sum(Γ.tamanos) - Γ.tamanos[1] - Γ.tamanos[end]
            acts = acts[2:end-1]
        end
        Acts = zeros(Float64, neuronas)
         i = 1
         for k in eachindex(acts)
             for j in eachindex( acts[k] )
                 Acts[i] = acts[k][j]
                 i += 1
             end
         end
         # Acts es el vector de activaciones
         # lo agregamos a la tabla
         push!(tabla,  Acts)
     end
     # Construimos los alphas
     A = []
     n = length(muestra)
     for i in 1:neuronas
         αᵢ = zeros(n)
         for j in 1:n
             αᵢ[j] = tabla[j][i]
         end
         push!(A, αᵢ)
     end
     return A
end

# calcula la matriz de distancia
function DistMatrix( Å,  F::Function )
    neuronas = length( Å )
    𝕄 = zeros(Float64, neuronas,  neuronas)

    for i in 1:neuronas
        for j in 1:neuronas
            𝕄[i,j] = F( Å[i] - Å[j] )
        end
    end
    return 𝕄
end


# Come el resultado de experimenta y calcula
# las matrices de distancia para todas las epocas

function exp_DistMat( epocas_red, norma ,size )
    # saca una muestra
    muestra = muestrea(size)
    matrices = []
    # Calcula matriz de distancia para cada epoca
    for Γ in epocas_red
        A = α_neuronas(Γ, muestra)
        M = DistMatrix( A, norma )
        push!(matrices, M)
    end
    return matrices
end
