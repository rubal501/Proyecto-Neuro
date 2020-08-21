# experimento
using Blink

include("funciones_red.jl")
include("func_DistMat.jl")
include("func_experimenta.jl")
include("training_data.jl")

# Quiero crear un objeto experimento que contenga los 4
# puntos considerados abajo

tamanos = [784,10,10]
e = haz_experimento( tamanos, 30, norm )

exp_plotbettivsepochs(e, 1)
#exp_plotpersisvsepoch(experimento, 1)
#
#win = Window()
#win1 = Window()
#win2 = Window()

#g = plotbarcode_pjs( eirene_objs[16] )
#body!(win, g )


#h_0 = plotbetticurve_pjs( experimento.eirene_objs[1] )
#body!( win1, h_0 )

#h_1 = plotbetticurve_pjs( experimento.eirene_objs[5] )
#body!( win2, h_1 )

#h = plotbetticurve_pjs( experimento.eirene_objs[end] )
#body!( win, h )
#

#plotpersistencediagram_pjs(C)
#plotclassrep_pjs(eirene_objs[16], coords = "mds")
#plotbetticurve_pjs(C)
