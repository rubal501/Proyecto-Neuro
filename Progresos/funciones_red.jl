## Crea la estructura de la red

using Random

struct Red_Neu
    num_layers
    tamanos
    biases
    weights
end

function ++(x::Red_Neu,y::Red_Neu)
    for i in 1:(x.num_layers-1)
        x.biases[i] = x.biases[i] + y.biases[i]
        x.weights[i] = x.weights[i] + y.weights[i]
    end
end

# Multiplica todas las matrices de la red temporal
# temporal nubla por el escalar dado
function escala(x::Red_Neu, n)
    for i in 1:(x.num_layers-1)
        x.biases[i] = x.biases[i]*n
        x.weights[i] = x.weights[i]*n
    end
end

function crea_red(tamanos, nula = false)

    num=length(tamanos)

    #menos el ultimo
    a = view(tamanos, 1:(num-1))
    #menos el primero
    b = view( tamanos,  2:num)
    if nula
        biases=[zeros( Float64,  (i,1) ) for i in b ]
        weights=[zeros( Float64, (i,j) ) for (i,j) in zip( b, a ) ]
    else
        biases=[randn( Float64,  (i,1) ) for i in b ]
        weights=[randn( Float64, (i,j) ) for (i,j) in zip( b, a ) ]
    end
    return Red_Neu(num,tamanos,biases,weights)
end

function sigmoid(z, derivative = false)
    val = 1.0/(1.0+exp(-z))
    if derivative
        return val * (1-val)
    else
        return val
    end
end

function feedforward(red::Red_Neu, a, partes = false)
    if partes
        zs = []
        activations = []
        a = reshape(a ,red.tamanos[1] , 1)
        push!(activations, a)
        push!(zs, reshape([0.0,0.0],2,1 ) )
    end
    for i in 1:(red.num_layers-1)
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

function backprop(x,y,red::Red_Neu)
    # Crea nubla a la forma de red
    nubla = crea_red(tamanos, true)
    zs, activaciones = feedforward(red, x, true)

    #Todos los indices mas uno en activacion y zs
    delta = cost_derivative.( activaciones[red.num_layers], y ) .* sigmoid.(zs[red.num_layers], true)

    # Primera capa de errores
    nubla.biases[nubla.num_layers-1] = delta
    nubla.weights[nubla.num_layers-1] = delta*transpose(activaciones[nubla.num_layers-1])

    # Retro-propaga
    for l in 2:(red.num_layers-1)
        z = zs[red.num_layers-l+1]
        sp = sigmoid.(z,true)
        delta = transpose( red.weights[red.num_layers-l+1]) * delta .* sp
        nubla.biases[red.num_layers-l] = delta
        nubla.weights[red.num_layers-l] = delta*transpose( activaciones[red.num_layers-l] )
    end
    return nubla
end

function update_mini_batch(red::Red_Neu, mini_batch, eta)
    # Guardara la suma de las nublas individuales, inicia nula
    nubla = crea_red(tamanos, true)

    #Calcula cada nubla individual(error)
    for dato in mini_batch
        x,y = dato
        nubla_x = backprop(x, y, red)
        ++(nubla,nubla_x)
    end
    #Actualizacion de pesos de la red

    # Multiplica todas las matrices de la red temporal
    # temporal nubla por el escalar dado
    escala(nubla, eta*(-1)/length(mini_batch))
    ++(red, nubla)
end

function SGD(red,training_data, epochs, mini_batch_size,  eta)
        n = length(training_data)
        m = mini_batch_size

        for j in 1:epochs
                shuffle!(training_data)

                maxx = div(n, mini_batch_size)
                mini_batches = [ training_data[1+k*m:(k+1)*m] for k in 0:maxx-1]

                for mini_batch in mini_batches
                        update_mini_batch(red,mini_batch, eta)
                end
                # if test_data
        end
end
