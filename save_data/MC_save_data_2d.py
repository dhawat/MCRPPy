import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))
from rpppy.monte_carlo_tests_setup import mc_results
from rpppy.monte_carlo_test_functions import (f_1, f_2, f_3, f_4, f_5,
                                             exact_integral_f_1, exact_integral_f_2,
                                             exact_integral_f_3, exact_integral_f_4,
                                             exact_integral_f_5,
                                             support_integrands)
import pickle


# Setup
np.random.seed(123)
nb_sample=100
nb_core = 30
nb_point_list=np.arange(50, 1025, 25).tolist()
estimators = ["MC",
              "MCP",
              "RQMC",
              "MCCV",
              "MCDPP"]
fct_list = [f_1, f_2, f_3, f_4, f_5]
fct_names = ["f_1", "f_2", "f_3", "f_4", "f_5"]

# For d=2
d=2
print("for d= ", d)
exact_integrals= [exact_integral_f_1(d), exact_integral_f_2(d),
                  exact_integral_f_3(d), exact_integral_f_4(d),
                  exact_integral_f_5(d)]
support_window = support_integrands(d)
if __name__ == "__main__":
    mc_results_2d, nb_point_list_used = mc_results(d, nb_point_list,
                                  nb_sample=nb_sample,support_window=support_window,
                                  fct_list=fct_list,
                                  fct_names=fct_names, exact_integrals=exact_integrals,
                                  estimators=estimators,
                                  nb_core=nb_core
                                  )

dict_to_save = {"d":d,
                "nb_point_list": nb_point_list_used,
                "mc_result":mc_results_2d
                }
with open('mc_results_2d.pickle', 'wb') as handle:
    pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done with d=", d)
#------------------------------------------------
