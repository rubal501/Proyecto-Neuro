## Aqui se procesan los datos desde la libreria

# hace falta descargar los datos y eso se puede hacer sencillamente
# desde la consola en la primera ejecucion

using MLDatasets

train_x , train_y = MNIST.traindata()

training_data = []
for i in 1:50000
    push!(training_data, [ reshape(train_x[:,:,i], 784), vectorizar( train_y[i] )] )
end
validation_data = []
for i in 50001:60000
    push!(validation_data, [reshape(train_x[:,:,i], 784 ), train_y[i] ] )
end
