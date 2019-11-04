from matplotlib import pyplot

alphas = [10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
trains = [50.59, 52.11, 54.51, 61.66, 68.08, 94.71, 99.22, 99.25, 99.25, 99.18]
devs   = [47.02, 54.01, 54.24, 61.47, 62.61, 76.72, 77.75, 77.06, 77.18, 76.61]

alphas.reverse()
trains.reverse()
devs.reverse()

for i in range(len(alphas)):
    print("%1.0e  &   %.2f  &   %.2f \\\\" % (alphas[i], trains[i], devs[i]))

pyplot.title("L2 Normalization - Accuracy vs. Alpha")
pyplot.xlabel("Alpha")
pyplot.ylabel("Accuracy (%)")
pyplot.plot(alphas, trains, "+-", label="Training")
pyplot.plot(alphas, devs, "+-",label="Validation")
pyplot.legend()
pyplot.show()
