from matplotlib import pyplot

neu_num_arr = []
hitrate_arr = []

with open("mnist_var_inner_layer_res", "r") as f:

    for line in f:
        tokens = line.strip().split(";")
        neu_num, hitrate = tokens

        neu_num_arr.append(neu_num)
        hitrate_arr.append(hitrate)

pyplot.plot(neu_num_arr, hitrate_arr)
pyplot.savefig("mnist_var_inner_layer_res_plot.png")
pyplot.close()
