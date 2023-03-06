import numpy as np
from GPPY.monte_carlo_tests_setup import mc_results
from GPPY.monte_carlo_test_functions import support_integrands

# Setup
nb_point_list= np.arange(50, 1550, 50)
print("lenght list of N", len(nb_point_list))

nb_sample=100
nb_function=5
estimators = ["MC",
              #"MCR",
              "MCP",
              "MCDPP",
              "RQMC"]
core_number=30

# d=2
d=2
print("for d= ", d)
support_window = support_integrands(d)
if __name__ == "__main__":
    mc_results_2d, nb_point_2d = mc_results(d, nb_point_list, nb_sample,
                                            nb_function, support_window,
                                            estimators=estimators, core_number=core_number,
                                            )
##save results

# d=3
d=3
support_window = support_integrands(d)
print("for d= ", d)
if __name__ == "__main__":
    mc_results_3d, _ = mc_results(d, nb_point_list, nb_sample,
                                    nb_function, support_window,
                                    estimators=estimators,
                                    core_number=core_number)

##save results

#d=4
d=4
support_window = support_integrands(d)
print("for d= ", d)
if __name__ == "__main__":
    mc_results_4d, _ = mc_results(d, nb_point_list, nb_sample,
                                    nb_function, support_window,
                                    estimators=estimators,
                                    core_number=core_number)

##save results
