using Eirene
using PlotlyJS
using Statistics
using Dates
using Logging

# Objeto Experimento
struct Exp_Red_Neu
    tamanos 
    performance
    epocas 
    dist_mat
    eirene_objs
end

# Hace un objeto experimento, una red entrenada, todas sus capturas
# y sus correspindientes objetos de Eirene
function haz_experimento( arquitectura, perf_deseado, norma, id, metodo; eta = 1.0, max_ejec = 20, 
    tamano_muestra = 100, dimensiones = 1 )

    #Se forman los archivos y carpetas referentes al experimento
    fecha = string(Dates.now())
    nombre_carpeta = "data/experimental/experimento_"*string(id)
    try
        mkdir(nombre_carpeta)
        mkdir(nombre_carpeta * "/activaciones")
        mkdir(nombre_carpeta * "/pesos")
    catch
    end

    #Se crea un archivo donde se guarde la informacion del experimento
    io = open("log.txt", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    @info("Experimento incializado")
    @info("Identificador de experimento =  "* string(id) )
    @info("Fecha de inicializacion =  "* fecha)
    @info("Arquitectura  =  " * string(arquitectura))
    @info("performance deseado = " * string( perf_deseado))
    flush(io)
    close(io)
    #Fin de la creacion de archivos


    performance, epocas = entrena( arquitectura, perf_deseado, max_ejec,id , eta = eta )
    dist_mat = exp_DistMat( epocas, norma, tamano_muestra, metodo)

    eirene_objs = []
    for matriz in dist_mat
        c = eirene(matriz, maxdim = dimensiones)
        push!(eirene_objs, c)
    end

    return Exp_Red_Neu(arquitectura, performance, epocas, dist_mat, eirene_objs)
end

function experimento_simple( arquitectura, perf_deseado, id,; max_ejec = 20, tamano_muestra = 100)

    #Se forman los archivos y carpetas referentes al experimento
    fecha = string(Dates.now())
    nombre_carpeta = "data/experimental/experimento_"*string(id)
    try
        mkdir(nombre_carpeta)
        mkdir(nombre_carpeta * "/activaciones")
        mkdir(nombre_carpeta * "/pesos")
    catch
    end

    #Se crea un archivo donde se guarde la informacion del experimento
    io = open("data/experimental/experimento_"*string(id)*"/log.txt", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    @info("Experimento incializado")
    @info("Identificador de experimento =  "* string(id) )
    @info("Fecha de inicializacion =  "* fecha)
    @info("Arquitectura  =  " * string(arquitectura))
    @info("performance deseado = " * string( perf_deseado))
    flush(io)
    #Fin de la creacion de archivos

    performance, epocas = entrenamiento_caida( arquitectura, perf_deseado, max_ejec,id,io )
    flush(io)
    close(io)
end

function recuperar_red(tamanos, id,num_epochs)
    redes = Red_Neu[]
    for epoch in 0:(num_epochs-1)
        ruta = "data/experimental/experimento_"*string(id)* "/pesos/epoch_"* string(epoch)* ".csv"
        
        data = readdlm(ruta, ',')
        #println("archivo abierto")

        array_bias = []
        array_weigths = []
        for lay in 1:(length(tamanos)-1)
            prev = Float64[]
            for element in data[lay,:]
                if typeof(element) == Float64
                    push!(prev, float(element))
                end
            end
            prev = reshape(prev,:,1) #Lo convertimos en un vector columna
            bias = prev[1:tamanos[lay+1]]
            #println("el tipo del vias es:  " * string(typeof(bias)))
            weight = reshape(prev[tamanos[lay+1]+1:length(prev)], tamanos[lay+1], tamanos[lay])
            push!(array_bias, bias)
            push!(array_weigths, weight)
        end
        nueva_red = Red_Neu(length(tamanos), tamanos, array_bias, array_weigths)
        push!(redes, nueva_red)
    end
    return redes
    
end


function experimento_preentrenado(redes, tamano_muestra,id )
    #=
    Esta funcion recibe como argumento un array con los tamanos de cada uno de las capas,
    un array de matrices de pesos y un arrray de vectores de biases
    =#
    alpha_vectores = []
    betta_vectores = []
    muestra  = muestrea(tamano_muestra)

    #Se crean los nuevos vectores de activaccion los diferentes metodos disponibles
    for red in redes
        push!(alpha_vectores, alpha_neuronas(red, muestra,false))
        push!(betta_vectores,  betta_neuronas(red, muestra, false))
    end
    
    for epoch in 1:length(redes)
        save_vectors(alpha_vectores[epoch], "alpha", id, string(epoch))
        save_vectors(betta_vectores[epoch], "betta", id, string(epoch))
    end

    println("Experimento terminado.")
    
end


# Entrena una red y guarda todas sus capturas y datos de performance
function entrena(tamanos, wanted_performance, max_ejecutions, id; eta = 1.0  )
        #Constantes de la ejecución
        ruta_pesos = "data/experimental/experimento_"*string(id)* "/pesos"
        mini_batch_size = 20
        # Crea una red con los tamanos dados
        gamma = crea_red(tamanos)
        # Y documenta toda su ejecución
        # Contadores
        epoca = 0
        performance = evalua(gamma, validation_data, epoca)
        salva_red(gamma, 0, ruta_pesos)

        capturas_red = [] # Capturas de la red
        capturas_perf = []
        push!(capturas_red, gamma) # Guarda la red recien creada
        push!(capturas_perf,performance )

        while( performance <= wanted_performance && epoca < max_ejecutions)
                # Haz un entrenamiento
                gamma = SGD_ϐ(gamma, training_data, mini_batch_size, eta)
                salva_red(gamma, epoca, ruta_pesos)
                println("red salvada ")
                # Cuenta las épocas
                epoca += 1

                # Evalua el performance
                performance = evalua(gamma, validation_data, epoca)
                # Guarda una captura de la red
                push!(capturas_red, gamma)
                append!(capturas_perf, performance)
                #println( performance,  epoca )
        end
        #println("Terminé :3")
        return capturas_perf, capturas_red
end

function caida(perf)
    #Esta funcion nos perimte hacer una caida exponencial de la eta
    return 10* exp(-1*(0.085)*perf)
end

function check_buffer(buffer)
    #Recibe como argumento un array y valor escalar del mismo tipo,
    #revisas si todas las entradas del buffer son iguales al elemento
    criterio = abs(maximum(buffer) - minimum(buffer) ) <= .7
    return criterio
end

function update_buffer(buffer, new_element)
    popfirst!(buffer)
    new_buffer = push!(buffer, new_element)
    return new_buffer
end

function entrenamiento_caida(tamanos, wanted_performance, max_ejecutions, id, logger )
    #Constantes de la ejecución
    ruta_pesos = "data/experimental/experimento_"*string(id)* "/pesos"
    mini_batch_size = 20
    # Crea una red con los tamanos dados
    gamma = crea_red(tamanos)
    # Y documenta toda su ejecución
    # Contadores
    # TODO hacer que el tamano del buffer sea dinamico
    tamano_buffer = 5
    buffer = ones(tamano_buffer)
    epoca = 0
    performance = evalua(gamma, validation_data, epoca)
    salva_red(gamma, 0, ruta_pesos)
    @info("Entrenamiento Iniciado en la fecha =  " *string(now())  )
    capturas_red = [] # Capturas de la red
    capturas_perf = []
    push!(capturas_red, gamma) # Guarda la red recien creada
    push!(capturas_perf,performance )
    


    while( performance <= wanted_performance && epoca < max_ejecutions )
            if check_buffer(buffer) == true && epoca > (tamano_buffer + 1)
                println("Fin del entrenamiento")
                @info("El aprendizaje se estanco en la epoca:  " * string(epoca) *"con el desempeno: " *string(performance))
                #Se borran las epocas donde SGD se quedo estancado 
                deleteat!(capturas_perf, (epoca-(tamano_buffer-2)):epoca)
                break
            end
            println(check_buffer(buffer))
            # Haz un entrenamiento
            eta = caida(performance)
            gamma = SGD_ϐ(gamma, training_data, mini_batch_size, eta)
            salva_red(gamma, epoca, ruta_pesos)
            println("red salvada ")
            # Cuenta las épocas
            epoca += 1
            # Evalua el performance
            performance = evalua(gamma, validation_data, epoca)
            # Guarda una captura de la red
            push!(capturas_red, gamma)
            append!(capturas_perf, performance)
            @info("Epoca terminada en la fecha =  " *string(now())  )
            @info("performance logrado en epoca:  " *string(epoca) * " = " * string( performance) *" por ciento"  )
            @info("valor de la eta:  " *string(eta))
            #println( performance,  epoca )
            update_buffer(buffer, performance)
    end
    @info("Entrenamiento terminado en la fecha =  " *string(now())  )
    flush(logger)
    #println("Terminé :3")
    return capturas_perf, capturas_red
end

    

function evalua(gamma, validation_data, epoca)
        correctos = evaluation(gamma, validation_data)
        n_data = length(validation_data)
		percent = convert(Int,floor((correctos/n_data)*100))
		println( "Epoca ", epoca," ", percent,"% :: "  )
        #print("Epoca ", epoca," : " ,correctos,"/", n_data,":::" )
        return (correctos/n_data)*100
end

# Grafica todas las curvas de betti para todas las epocas del experimento
function exp_plotbetticurves(epsilon::Exp_Red_Neu)
	traces = Vector{GenericTrace{Dict{Symbol,Any}}}(undef,length(epsilon.eirene_objs))
	for i in eachindex(epsilon.eirene_objs)
		D = epsilon.eirene_objs[i]
		bcu = Eirene.betticurve(D)
		trace = PlotlyJS.scatter(x=bcu[:,1], y=bcu[:,2],mode="lines", line_width=1, name=string("Epoca ",i-1) )
		traces[i]= trace
	end
	PlotlyJS.plot(traces)
end

# Grafica la curva de entrenamiento del experimento
function exp_plottrainingcurve(epsilon::Exp_Red_Neu)
    trace = PlotlyJS.scatter(;x=0:length(epsilon.performance)-1,y=epsilon.performance,mode="lines", name="Training")
    PlotlyJS.plot(trace)
end

# Grafica los numeros de betti promedio vs las epochs
function exp_plotbettivsepochs(exp::Exp_Red_Neu, dimension=1)
    avg_betti = zeros(length(exp.eirene_objs))
    std_betti = zeros(length(exp.eirene_objs))

    for i in eachindex(exp.eirene_objs)
        betti = betticurve(exp.eirene_objs[i], dim = dimension)
        prev = [betti[p,2] for p in 1:Int(length(betti)/2)]
        avg_betti[i] = mean(prev)
        std_betti[i] = std(prev)
    end

    upper_bound = PlotlyJS.scatter(
        name="Upper Bound",
        x=1:length(exp.eirene_objs),
        y= avg_betti + std_betti ,
        mode="lines",
        marker=attr(color="#444"),
        line=attr(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty")

    trace = PlotlyJS.scatter(
        name="Measurement",
        x=1:length(exp.eirene_objs),
        y= avg_betti ,
        mode="lines",
        line=attr(color="rgb(31, 119, 180)"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty")

    lower_bound = PlotlyJS.scatter(
        name="Lower Bound",
        x=1:length(exp.eirene_objs),
        y= avg_betti - std_betti ,
        marker=attr(color="#444"),
        line=attr(width=0),
        mode="lines")

    data = [lower_bound, trace, upper_bound]

    layout = PlotlyJS.Layout(
    yaxis=attr(title="betti-1"),
    xaxis=attr(title="epoch"),
    title="valor del numero de betti-1 a traves de las epochs",
    showlegend = false)

    pl = PlotlyJS.plot(data, layout)
    PlotlyJS.savefig(pl, "prueba2.html")
    PlotlyJS.savefig(pl, "prueba2.png")
    println("Graficas de betti logradas")
    return nothing
end


#gradica las persistencias promedio vs las epochs
function exp_plotpersisvsepoch(exp::Exp_Red_Neu, dimension=1 )
    num_epochs = length(exp.epocas)
    average_persistence = zeros(num_epochs)
    std_persistence = zeros(num_epochs)
    for i in eachindex(exp.eirene_objs)
        pers = barcode(exp.eirene_objs[i], dim= dimension)
        sum = zeros(Float64 , Int(length(pers)/2))
        for j in 1:Int(length(pers)/2)
            sum[j] = abs(pers[j,1] - pers[j,2])
        end
        prom = mean(sum)
        std = std(sum)
        average_persistence[i] = prom
        std_persistence[i] = std
    end
    upper_bound = PlotlyJS.scatter(
        name="Upper Bound",
        x=0:num_epochs,
        y=average_persistence + std_persistence ,
        mode="lines",
        marker=attr(color="#444"),
        line=attr(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty")
    
    trace = PlotlyJS.scatter(
        name="Measurement",
        x=0:num_epochs,
        y=average_persistence  ,
        mode="lines",
        line=attr(color="rgb(31, 119, 180)"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty")
    
    lower_bound = PlotlyJS.scatter(
        name="Lower Bound",
        x=0:num_epochs,
        y=average_persistence  - std_persistence ,
        marker=attr(color="#444"),
        line=attr(width=0),
        mode="lines")


    data = [upper_bound, trace,  lower_bound]
    
    layout = PlotlyJS.Layout(
        yaxis=attr(title="persistencias promedio"),
        xaxis=attr(title="epoch"),
        title="Persistencias promedio a lo largo de las epocas",
        showlegend = false)
    pl = PlotlyJS.plot(data, layout)
    PlotlyJS.savefig(pl, "prueba1.html")
    return nothing
end
