## Crea la estructura de la red
struct Red_Neu
    num_layers
    tamanos
    biases
    weights
end

tamanos = [3,7,2]

function crea_red(tamanos)

    num=length(tamanos)

    #menos el ultimo
    a = view(tamanos, 1:(num-1))
    #menos el primero
    b = view( tamanos,  2:num)
    biases=[randn( Float64,  (i,1) ) for i in b ]

    weights=[randn( Float64, (i,j) ) for (i,j) in zip( b, a ) ]

    return Red_Neu(num,tamanos,biases,weights)
end

function sigmoid(z)
    return 1.0/(1.0+exp(-z))
end

function feedforward(red::Red_Neu,vect)
    for i in 1:(red.num_layers-1)
        vect = sigmoid.(red.weights[i]*vect + red.biases[i])
    end
    return vect
end
