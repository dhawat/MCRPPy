import math
import pickle
import statistics as stat
import time
from multiprocessing import freeze_support
from multiprocessing.pool import Pool

import numpy as np
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from scipy import stats

from mcrppy.monte_carlo_base import (bandwidth_0_delyon_portier,
                                       control_variate_mc,
                                       delyon_portier_mc,
                                       estimate_control_variate_parameter,
                                       estimate_control_variate_proposal,
                                       monte_carlo_method)
from mcrppy.point_pattern import PointPattern
from mcrppy.point_processes import BinomialPointProcess, ScrambleSobolPointProcess
from mcrppy.utils import regression_line, error, mse

def mc_results(d, nb_points_list,
               nb_samples, support_window,
               fct_list, fct_names,
               exact_integrals=None,
               estimators=None,
               nb_point_cv=500,
               file_name=None,
               nb_cores=7,
               pool_dpp=False,
               **repelled_params):
    if estimators is None:
        estimators = ["MC", "MCRB", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
    else :
        estimators_all = ["MC", "MCRB", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
        if sum([k not in estimators_all for k in estimators])!=0:
            raise ValueError("The allowed estimators are {}".format(estimators_all))

    print("d=", d, ", nb samples=", nb_samples, ", nb points=", nb_points_list)
    print("------------------------------------------------")
    results = {}
    nb_points_used=[]
    time_1 = time.time()
    MC, MCRB = None, None
    RQMC, MCCV, MCDPP =  None, None, None
    MCKS_h0, MCKSc_h0 = None, None
    for n in nb_points_list :
        time_mc = {k:0 for k in estimators}
        # Push Binomial
        ## Push Binomial pp
        time_start = time.time()
        repelled_binomial_pp = _repelled_binomial_samples(nb_samples=nb_samples,
                                                          nb_points=n, window=support_window,nb_cores=nb_cores, **repelled_params)
        time_end = time.time() - time_start
        time_mc["MCRB"]=[int(time_end/60), (time_end%60)]
        mean_nb_points_rbpp= int(stat.mean([p.points.shape[0] for p in repelled_binomial_pp]))
        nb_points_used.append(mean_nb_points_rbpp)
        ## MCRB
        MCRB = _mc_result(pp_list=repelled_binomial_pp, type_mc="MCRB", mc_f_n=MCRB, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MC" in estimators:
            # Binomial
            ## Binomial pp
            time_start = time.time()
            binomial_pp = _binomial_samples(nb_samples=nb_samples,
                                            nb_points=mean_nb_points_rbpp,
                                            window=support_window)
            time_end = time.time() - time_start
            time_mc["MC"]=[int(time_end/60), time_end%60]
            ## MC classic
            MC = _mc_result(pp_list=binomial_pp, type_mc="MC", mc_f_n=MC, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MCCV" in estimators:
            #MC Control variate
            time_start = time.time()
            binomial_pp = _binomial_samples(nb_samples=nb_samples, nb_points=mean_nb_points_rbpp, window=support_window)
            MCCV = _mc_result(pp_list=binomial_pp,
                              type_mc="MCCV",
                              mc_f_n=MCCV,
                              fct_list=fct_list,fct_names=fct_names, exact_integrals=exact_integrals,
                              nb_point_cv=nb_point_cv, support_window_cv=support_window)
            time_end = time.time() - time_start
            time_mc["MCCV"]=[int(time_end/60), time_end%60]


        if "MCDPP" in estimators:
            # DPP Bardenet Hardy
            ## DPP pp
            time_start = time.time()
            # dpp samples in [-1, 1]^d
            dpp_samples = _dpp_samples(nb_points=mean_nb_points_rbpp, d=d, nb_samples=nb_samples, nb_cores=nb_cores, pool_dpp=pool_dpp)
            #rescale points to be in support_window
            dpp_pp_scaled = [PointPattern(p/2, window=support_window) for p in dpp_samples]
            ##MCDPP
            #scaled weight
            weights_dpp = [_mcdpp_weights(p, eval_pointwise=True)
                        for p in dpp_samples]
            time_end = time.time() - time_start
            time_mc["MCDPP"]=[int(time_end/60), time_end%60]
            MCDPP = _mc_result(pp_list=dpp_pp_scaled, type_mc="MCDPP", mc_f_n=MCDPP,
                                fct_list=fct_list,
                                fct_names=fct_names,exact_integrals=exact_integrals,
                                weights=weights_dpp)

        if "RQMC" in estimators:
            #RQMC
            ## Scrambeled Sobol pp
            time_start = time.time()
            sobol_pp = _scramble_sobol_samples(nb_samples=nb_samples, nb_points=mean_nb_points_rbpp, window=support_window)
            time_end = time.time() - time_start
            time_mc["RQMC"]=[int(time_end/60), time_end%60]
            ## RQMC
            RQMC = _mc_result(pp_list=sobol_pp, type_mc="RQMC", mc_f_n=RQMC, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MCKS_h0" in estimators:
            time_start = time.time()
            MCKS_h0= _mc_result(pp_list=binomial_pp, type_mc="MCKS_h0",
                                mc_f_n=MCKS_h0,
                                fct_list=fct_list,
                                fct_names=fct_names,
                                exact_integrals=exact_integrals,
                                correction=False)
            time_end = time.time() - time_start
            time_mc["MCKS_h0"]=[int(time_end/60), time_end%60]


        if "MCKSc_h0" in estimators:
            time_start = time.time()
            MCKSc_h0= _mc_result(pp_list=binomial_pp,
                                    type_mc="MCKSc_h0",
                                    mc_f_n=MCKSc_h0,
                                    fct_list=fct_list,
                                    fct_names=fct_names,
                                    exact_integrals=exact_integrals,
                                    correction=True)
            time_end = time.time() - time_start
            time_mc["MCKSc_h0"]=[int(time_end/60), time_end%60]

        print("----------------------------------------------")
        print("N expected=", n, ", N obtained", mean_nb_points_rbpp)
        print("Time =", time_mc)
        print("----------------------------------------------")

    time_2 = time.time() - time_1
    print("Time all", int(time_2/60), "min", time_2%60, "s")
    for k in estimators:
        results[k] = locals()[k]
    if file_name is not None:
        dict_to_save = {"d":d,
                "nb_point_list": nb_points_used,
                "mc_result":results
                }
        with open(file_name, 'wb') as handle:
            pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results, nb_points_used

# data frame of the MSE of MC methods for N = nb_points_list[idx_nb_points]
def dataframe_mse_results(mc_results, fct_names, exact_integrals, nb_samples, idx_nb_point=-1):
    type_mc = mc_results.keys()
    mse_dict={}
    for name_f, integ_f in zip(fct_names, exact_integrals):
        mse_dict["MSE("+ name_f + ")"]={}
        mse_dict["std(MSE(" + name_f + "))"]={}
        for t in type_mc:
            m_f = mc_results[t]["mc_results_" + name_f]["m_"+ t]
            std_f = mc_results[t]["mc_results_"+ name_f]["std_"+ t]
            mse_f = mse(m_f, std_f, integ_f, verbose=False)
            std_mse = np.array(std_f/np.sqrt(nb_samples))
            mse_dict["MSE(" + name_f + ")"][t] = mse_f[idx_nb_point]
            mse_dict["std(MSE("+ name_f +"))"][t] = std_mse[idx_nb_point]
    return mse_dict

# Kolmogorov Smirniv test for residual of the liear regression to test if the residual is Gaussian
# q-q plot pf the residual of the linear regresssion of log(std) w.r.t. log(N)
def dataframe_residual_test(mc_list, nb_points_list, fct_names, test_type="SW", **kwargs):
    result_test_dict = {}
    type_mc = mc_list.keys()
    for name in fct_names :
        result_test_f_dict = {}
        for t in type_mc:
            std_f = mc_list[t]["mc_results_"+ name]["std_"+ t]
            _, _, _, _, result_test = regression_line(nb_points_list, std_f, residual=True,residual_normality_test=test_type, **kwargs)
            result_test_f_dict[t] =  ("stat={0:.3f}".format(result_test[0]), "p={0:.3}".format(result_test[1]))
        result_test_dict[name] = result_test_f_dict
    return result_test_dict

# Mann-Whitney test for the (square) errors of the method of type 'type_mc_test' with the others methods
def dataframe_error_test(mc_list, nb_points_list, fct_name, type_mc_test="MCRB"):
    mw_test_dict = {}
    type_mc = list(mc_list.keys())
    nb_nb_points = len(nb_points_list)
    #MC methods to be tested with type_mc_to_test
    type_mc.remove(type_mc_test)
    error_test = mc_list[type_mc_test]["mc_results_"+fct_name]["error_"+ type_mc_test]
    for t in type_mc:
        error_tested_with = mc_list[t]["mc_results_"+fct_name]["error_"+ t]
        mw_test_N_dict = {}
        for n in range(nb_nb_points):
            mw_test = stats.mannwhitneyu(error_test[n], error_tested_with[n])
            mw_test_N_dict["N={}".format(nb_points_list[n])] =  ("stat={0:.3f}".format(mw_test[0]), "p={0:.3}".format(mw_test[1]))
        mw_test_dict[type_mc_test+ " and "+ t] = mw_test_N_dict
    return mw_test_dict

def _repelled_binomial_samples(nb_samples, nb_points, window, nb_cores=4, **repelled_params):
    binomial = BinomialPointProcess()
    repelled_pp = []
    for _ in range(nb_samples):
        _, rpp = binomial.generate_repelled_point_pattern(nb_points=nb_points, window=window, nb_cores=nb_cores, **repelled_params)
        repelled_pp.append(rpp)
    return repelled_pp

def _binomial_samples(nb_samples, nb_points, window):
    binomial = BinomialPointProcess()
    binomial_pp = [binomial.generate_point_pattern(nb_points=nb_points, window=window) for _ in range(nb_samples)]
    return binomial_pp

def _scramble_sobol_samples(nb_samples, nb_points, window):
    sobol = ScrambleSobolPointProcess()
    sobol_pp = [sobol.generate_point_pattern(nb_points=nb_points, window=window) for _ in range(nb_samples)]
    return sobol_pp

def _multivariate_jacobi_samples(nb_points, d, nb_samples=None):
    jac_params = np.array([[0, 0]]*d) #jaccobi measure=1
    dpp = MultivariateJacobiOPE(nb_points, jac_params)
    if nb_samples is not None:
        dpp_samples = [dpp.sample() for _ in range(nb_samples)]
    else:
        dpp_samples = dpp.sample()
    return dpp_samples

def _dpp_samples(nb_points, d, nb_samples, nb_cores, pool_dpp):
    if pool_dpp:
        freeze_support()
        with Pool(processes=nb_cores) as pool:
            #print("Number of processes in the DPP pool ",pool._processes)
            dpp_samples = pool.starmap(_multivariate_jacobi_samples, [(nb_points, d)]*nb_samples)
        pool.close()
        pool.join()

    else:
        dpp_samples = _multivariate_jacobi_samples(nb_points=nb_points, d=d, nb_samples=nb_samples)
    return dpp_samples

def _mcdpp_weights(points, eval_pointwise=True, scale=None, jacobi_params=None):
    nb_points, d = points.shape
    if scale is None:
        scale = 1/2**d #for support equal [-1/2,1/2]^d
    if jacobi_params is None:
        jacobi_params = np.array([[0, 0]]*d) #jaccobi measure=1
    dpp = MultivariateJacobiOPE(nb_points, jacobi_params)
    weights_dpp = scale/dpp.K(points, eval_pointwise=eval_pointwise)
    return weights_dpp

def _mc_result(pp_list, type_mc, fct_list,
                 fct_names,
                 exact_integrals=None,
                 mc_f_n=None,
                 weights=None,
                 correction=True,
                 verbose=True,
                 nb_point_cv=None,
                 support_window_cv=None):
    print("For", type_mc)
    print("---------------")
    if mc_f_n is None:
        mc_f_n = {}
        for name in fct_names:
            mc_f_n["mc_results_" +name] = _mc_f_dict(type_mc=type_mc)
    i=0
    if type_mc=="MCCV":
        points_cv_proposal= support_window_cv.rand(n=nb_point_cv, seed=0)
        points_cv_param_estimate= support_window_cv.rand(n=nb_point_cv, seed=1)
    for f,name in zip(fct_list, fct_names):
        if type_mc in ["MC", "MCRB", "RQMC"]:
            mc_values = [monte_carlo_method(points=p.points, f=f) for p in pp_list]
        elif type_mc=="MCCV":
            proposal, m_proposal = estimate_control_variate_proposal(points=points_cv_proposal, f=f, poly_degree=2)
            c = estimate_control_variate_parameter(points=points_cv_param_estimate, f=f, proposal=proposal)
            mc_values=[control_variate_mc(points=p.points,
                                          f=f,
                                          proposal=proposal,
                                          mean_proposal= m_proposal,
                                          c=c)
                      for p in pp_list]
        elif type_mc=="MCDPP":
            mc_values = [monte_carlo_method(points=p.points, f=f, weights=w)
                         for (p,w) in zip(pp_list, weights)]
        elif type_mc in ["MCKS", "MCKSc"]:
            mc_values = [delyon_portier_mc(point_pattern=p,
                                           f=f,
                                           correction=correction)
                        for p in pp_list]
        elif type_mc in ["MCKS_h0", "MCKSc_h0"]:
            mc_values = [delyon_portier_mc(point_pattern=p,
                                           f=f,
                                           bandwidth=bandwidth_0_delyon_portier(p.points),
                                           correction=correction)
                        for p in pp_list]
        else:
            raise ValueError("The possible Monte Carlo methods are : MC, MCRB, RQMC, MCDPP, MCCV, MCKS_h0, MCKSc_h0, MCKS, MCKSc.")
        # mean MC method of type 'type_mc'
        mean_mc = stat.mean(mc_values)
        mc_f_n["mc_results_" + name]["m_"+ type_mc].append(mean_mc)
        # std MC method of type 'type_mc'
        # didn't use stat.stdev since it uses different estimator of the variance
        std_mc = math.sqrt(stat.mean((mc_values - mean_mc)**2))
        mc_f_n["mc_results_" + name]["std_"+ type_mc].append(std_mc)
        # square erreur MC method of type 'type_mc'
        if exact_integrals is not None:
            integ_f= exact_integrals[i]
            mc_f_n["mc_results_" + name]["error_"+ type_mc].append(error(mc_values, integ_f))
        # print MSE
        if verbose:
            print("FOR " + name)
            m_list = mc_f_n["mc_results_" + name]["m_"+ type_mc]
            std_list = mc_f_n["mc_results_" + name]["std_"+ type_mc]
            print( "std=", std_list)
            if exact_integrals is not None:
                mse_mc = mse(mean=m_list, std=std_list, exact=integ_f)
                print("MSE=", mse_mc)
        i+=1
    return mc_f_n

def _mc_f_dict(type_mc, se=True):
    d = {}
    d["m_"+type_mc]=[]
    d["std_"+type_mc]=[]
    if se:
        d["error_"+type_mc]=[]
    return d
