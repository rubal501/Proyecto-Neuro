## Crea la estructura de la red

using Random
using LinearAlgebra
using DelimitedFiles

struct Red_Neu
    num_layers
    tamanos
    biases
    weights 
end
# Extiende las suma y resta de Julia para nuestro tipo de dato :D
for op = (:+, :-)
    eval(quote
        Base.$op(a::Red_Neu, b::Red_Neu) = Red_Neu(a.num_layers, a.tamanos, $op(a.biases, b.biases), $op(a.weights, b.weights) )
    end)
end
# Extiende el producto y division por escalares a nuestro tipo de dato :)
for op = (:*, :/)
    eval(quote
        Base.$op(red::Red_Neu, k) = Red_Neu(red.num_layers, red.tamanos,$op(red.biases,k), $op(red.weights, k) )
    end)
end

# Crea la red, si nula=true los pesos son cero
function crea_red(tamanos, nula::Bool = false)
    num=length(tamanos)

    #se forma un sub array de num menos el ultimo
    a = view(tamanos, 1:(num-1))
    #se forma un sub array de num menos el primero
    b = view( tamanos,  2:num)
    if nula
        biases=[zeros( Float64,  (i,1) ) for i in b ]
        weights=[zeros( Float64, (i,j) ) for (i,j) in zip( b, a ) ]
    else
        biases=[randn( Float64,  (i,1) ) for i in b ]
        weights=[randn( Float64, (i,j) ) for (i,j) in zip( b, a ) ]
    end
    gamma = Red_Neu(num,tamanos,biases,weights)
    return gamma
end

function sigmoid(z, derivative = false)
     val = 1.0/(1.0+exp(-z))
    if derivative
        return val * (1-val)
    else
        return val
    end
end

# feedforward as functor
function (red::Red_Neu)(a, partes::Bool = false)
    if partes
        zs = []
        activations = []
        a = reshape(a ,red.tamanos[1] , 1)
        push!(activations, a)
        #push!(zs, reshape([0.0,0.0],2,1 ) )
        push!(zs, "ola k ase")
    end
    for i in 1:(red.num_layers-1)
       # println(size(red.weights[i]))
        #println(size(a))
        #println(size(red.weights[i]*a))
        #println(typeof(red.biases[i]))
        #println(size(red.biases[i]))
        z = red.weights[i]*a + red.biases[i]
        a = sigmoid.(z)
        if partes
            push!(zs, z)
            push!(activations, a)
        end
    end
    if partes
        return zs, activations
    else
        return a
    end
end

function cost_derivative(output_activations, y)
    return output_activations - y   
end

# Backprop as functor (simply call your Red_Neu object with your
# data = x, y, i.e. if red ∈ Red_Neu, simply call red(x, y)   )
function (red::Red_Neu)(x , y)
    # Crea nubla a la forma de red
    nabla = crea_red(red.tamanos, true)
    zs, activaciones = red( x, true)

    #Todos los indices mas uno en activacion y zs
    delta = cost_derivative.( activaciones[red.num_layers], y ) 
    
    # Primera capa de errores
    nabla.biases[nabla.num_layers-1] = delta
    nabla.weights[nabla.num_layers-1] = delta*transpose(activaciones[nabla.num_layers-1])

    # Retro-propaga
    for l in 2:(red.num_layers-1)
        z = zs[red.num_layers-l+1]
        sp = sigmoid.(z,true)
        delta = sp .* (transpose( red.weights[red.num_layers-l+1]) * delta)
        nabla.biases[red.num_layers-l] = delta
        nabla.weights[red.num_layers-l] = delta*transpose( activaciones[red.num_layers-l] )
    end
    return nabla
end

# Esta es la función que actualiza los pesos de la red
function update_mini_batch(net::Red_Neu, mini_batch, eta::Float64)
    # Guardara la suma de las nublas individuales, inicia nula
    grad = crea_red(net.tamanos, true)

    #Calcula cada nubla individual(error)
    for dato in mini_batch
        x,y = dato
        gradx = net(x, y)
        grad += gradx
    end
    #Actualizacion de pesos de la red
    grad = grad*(-eta/length(mini_batch))

    return grad
end

# Stochastic Gradient Descent
function SGD( gamma::Red_Neu, training_data, epochs, mini_batch_size, eta, test_data = false, validation_data = nothing)
        n = length(training_data)
        m = mini_batch_size
        experimento(gamma,0)

        for j in 1:epochs
                shuffle!(training_data)

                maxx = div(n, mini_batch_size)
                mini_batches = [ training_data[1+k*m:(k+1)*m] for k in 0:maxx-1]

                for mini_batch in mini_batches
                        grad = update_mini_batch(gamma, mini_batch, eta)
                        gamma += grad # Se suma la grad calculada en update_mini_batch
                end
                if test_data
                    aa = validation_data
                    correctos = evaluation(gamma, validation_data)
                    n_data = length(validation_data)
                    print("Epoca ", j," : " ,correctos,"/", n_data, "\n" )
                    pres = correctos/n_data * 100
                    save_pres(pres)
                    experimento(gamma, j)
                else
                    print("Epoca ", j, " completada :D \n")
                end
        end
        salva_red(gamma)
        println("Termine :3")
        return gamma
end

# Prueba el descempeño de la red
function evaluation(red::Red_Neu, data)
    correctos = 0
    for dato in data
        x, y = dato
        # Este desmadre es por que es una matriz y
        # regresa coordenadas cartesianas
        a = findmax( red(x) )[2][1]
        if( a==(y+1) )
            correctos += 1
        end
    end
    return correctos
end
# usada para dar el formato adecuado a los datos
function vectorizar(valor)
    vector =  zeros(Float32, 10)
    vector[valor+1] = 1
    return vector
end

function cost(x, y)
    norm(y - x)
end

# backprop numerico para ver si backprop funciona
function numeric_gradient(red, x, y, ξ)
    nubla = crea_red(red.tamanos,  true)
      for i in eachindex(red.weights)
            for j in eachindex( red.weights[i] )
                  temp1 = deepcopy(red)
                  temp2 = deepcopy(red)

                  temp1.weights[i][j] = red.weights[i][j] + ξ
                  temp2.weights[i][j] = red.weights[i][j] - ξ

                  nubla.weights[i][j] = ( cost(temp1(x), y)-cost(temp2(x), y) )/(2*ξ)
            end
      end

      for i in eachindex(red.biases)
            for j in eachindex( red.biases[i] )
                  temp1 = deepcopy(red)
                  temp2 = deepcopy(red)

                  temp1.biases[i][j] = red.biases[i][j] + ξ
                  temp2.biases[i][j] = red.biases[i][j] - ξ

                  nubla.biases[i][j] = ( cost(temp1(x), y)-cost(temp2(x), y) )/(2*ξ)
            end
      end
      #print(ξ, " ")
      return nubla
end

#=esta aplica el feedforward con las 100 imagenes pre seleccionadas 
y guarda los valores de activacion en cada una de las iteracionesjj
function experimento(red::Red_Neu, epoch)
    datos_experimento = readdlm("muestras.csv", ',',Int16)
    cont = [ [] for i in 1:red.num_layers ]
    
    for index in 1:100
        z,activation = red(datos_experimento[index,1:784], true)
        for lay in 1:red.num_layers
            push!(cont[lay], activation[lay])
        end
    end

    for lay in 1:red.num_layers
        #println("estoy en la layer:"*string(lay))
        for neu in 1:red.tamanos[lay]
            prev =[]
            #println("Estoy en la neurona : " * string(neu))
            for e in 1:100
                #println("el tipo conflictiov es " * string(typeof(cont[lay-1])))
                #println(e)
                #println("estoy en el elemento de la muestra: " * string(e))
                #println("espero esto jale "* string( typeof(cont[lay-1][e])))
                push!(prev, cont[lay][e][neu])
            end
            name = "dataexp/epoch" * string(epoch) *".csv"
            #println(name)
            io = open(name, "a")
            writedlm( io,transpose(prev) , ',')
            close(io)

        end
    end


end

=#
# Stochastic Gradient Descent
function SGD_ϐ( gamma::Red_Neu, training_data, mini_batch_size, eta)
    n = length(training_data)
    m = mini_batch_size
    # No mas epocas
    shuffle!(training_data)
    maxx = div(n, mini_batch_size)
    mini_batches = [ training_data[1+k*m:(k+1)*m] for k in 0:maxx-1]
    for mini_batch in mini_batches
            grad = update_mini_batch(gamma, mini_batch, eta)
            gamma += grad # Se suma la grad calculada en update_mini_batch
    end
    # Aqui terminaba la epoca
    return gamma
end


function salva_red(red::Red_Neu, epoch, ruta)
    #Esta funcion recibe como argumentos a una red neuronal, la epoca de entrenamiento
    # la que se encuentra y la ruta en la cual se deben de guardar los archivos de las activaciones
    name = ruta* "/epoch_"* string(epoch)* ".csv"
    #println(length(red.num_layers))
    for i in 1:(red.num_layers-1)
        io = open(name, "a")
        #Concatenamos la matriz de pesos con el vector columna de los biases 
        bias = red.biases[i]
        #Se forma un vector de pesos modificando la forma de la matriz de pesos
        weights = reshape(red.weights[i], :,1)
        #Se concatena el vector formado con la matriz de pesos de la capa con el vector de pesos 
        data = transpose(vcat(bias, weights))
        #Transponemos el vector para poder almacenarlo como una sola fila en nuestro archivo csv
        writedlm( io, data, ',')
        close(io)
    end
end

function prueba_pres(red::Red_Neu, validation_data)
    correctos = evaluation(red, validation_data)
    n_data = length(validation_data)
    print(correctos,"/", n_data, "\n" )
    prom = (correctos/n_data * 100)
    return prom
end

function save_pres(pres)
    io = open("pres.csv", "a")
    writedlm( io, pres, ',')
end