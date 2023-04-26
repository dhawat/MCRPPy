import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))
from mcrppy.monte_carlo_methods import mc_results
from mcrppy.integrand_test_functions import (support_integrands,
                                             f_1, f_2, f_3,
                                             exact_integral_f_1, exact_integral_f_2,
                                             exact_integral_f_3)

# Setup
np.random.seed(123)
nb_samples=100
nb_cores = 20
nb_points_list=np.arange(50, 1050, 50).tolist()
fct_list = [f_1, f_2, f_3]
fct_names = ["f_1", "f_2", "f_3"]
estimators = ["MC",
              "MCRB",
              "MCCV",
              "RQMC"]
print("Number of tests: ", len(nb_points_list))
print("Number of points to be used:", nb_points_list)
print("Methods to be used:", estimators)

# For d=5
d=5
nb_points_list=np.arange(50, 1050, 50).tolist()
exact_integrals= [exact_integral_f_1(d),
                  exact_integral_f_2(d),
                  exact_integral_f_3(d)]
support_window = support_integrands(d)
if __name__ == "__main__":
    results, nb_points = mc_results(d,
                                          nb_points_list=nb_points_list,
                                          nb_samples=nb_samples,
                                          support_window=support_window,
                                          fct_list=fct_list,
                                          fct_names=fct_names,
                                          exact_integrals=exact_integrals,
                                          estimators=estimators,
                                          nb_cores=nb_cores,
                                          file_name="mc_results_5d_final.pickle"
                                            )

print("Done with d=", d)
#------------------------------------------------
