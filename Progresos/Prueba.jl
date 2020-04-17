include("MNIST.jl")
using .MNIST 

println(isequal(Tuple{Array{Tuple{Array{Int64,1},Array{Int64,1}},1},Array{Tuple{Array{Int64,1},Int64},1}}, typeof(MNIST.obtener_datos())))