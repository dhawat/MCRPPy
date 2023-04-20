import math
import pickle
import statistics as stat
import time
from multiprocessing import freeze_support
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import statsmodels.api as sm
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from scipy import stats

from mcrppy.monte_carlo_methods import (bandwidth_0_delyon_portier,
                                       control_variate_mc,
                                       delyon_portier_mc,
                                       estimate_control_variate_parameter,
                                       estimate_control_variate_proposal,
                                       monte_carlo_method, sobol_point_pattern,
                                       sobol_sequence)
from mcrppy.point_pattern import PointPattern
from mcrppy.point_processes import generate_scramble_sobol_sample
from mcrppy.repelled_point_process import RepelledPointProcess
from mcrppy.spatial_windows import BallWindow, BoxWindow
from mcrppy.utils import regression_line, error, mse

def mc_f_dict(type_mc, se=True):
    d = {}
    d["m_"+type_mc]=[]
    d["std_"+type_mc]=[]
    if se:
        d["error_"+type_mc]=[]
    return d

def mc_results(d, nb_points_list,
               nb_samples, support_window,
               fct_list, fct_names,
               exact_integrals=None,
               estimators=None,
               nb_point_cv=500,
               file_name=None,
               epsilon_push=None,
               nb_core=7,
               pool_dpp=True,
               add_r_push=None,
               **kwargs):
    if estimators is None:
        estimators = ["MC", "MCR", "MCP", "MCPS", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
    else :
        estimators_all = ["MC", "MCR", "MCP", "MCPS", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
        if sum([k not in estimators_all for k in estimators])!=0:
            raise ValueError("The allowed estimators are {}".format(estimators_all))

    print("d=", d, ", nb samples=", nb_samples, ", nb points=", nb_points_list)
    print("------------------------------------------------")
    results = {}
    nb_points_used=[]
    time_1 = time.time()
    MCP, MC, RQMC, MCCV, MCDPP= None, None, None, None, None
    MCR, MCPS, MCKS_h0, MCKSc_h0 =None, None, None, None
    for n in nb_points_list :
        time_mc = {k:0 for k in estimators}
        # Push Binomial
        ## Push Binomial pp
        time_start1 = time.time()
        push_pp = generate_repelled_binomial(d, support_window=support_window, nb_point=n, nb_sample=nb_samples, add_r=add_r_push, epsilon=epsilon_push, core_number=nb_core, **kwargs)
        time_end = time.time() - time_start1
        time_mc["MCP"]=[int(time_end/60), (time_end%60)]
        nb_point_output= int(stat.mean([p.points.shape[0] for p in push_pp]))
        nb_points_used.append(nb_point_output)
        ## MCP
        MCP = mc_results_n(pp_list=push_pp, type_mc="MCP", mc_f_n=MCP, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MCPS" in estimators:
            # Push Sobol
            ## Push Sobol pp
            time_start1 = time.time()
            push_sobol_pp = generate_repelled_binomial(d, support_window=support_window, nb_point=n, nb_sample=nb_samples, father_type="Sobol", add_r=add_r_push, **kwargs)
            nb_point_output_sobol= int(stat.mean([p.points.shape[0] for p in push_sobol_pp]))
            time_end = time.time() - time_start1
            time_mc["MCPS"]=[int(time_end/60), (time_end%60)]
            ## MCPS
            MCPS = mc_results_n(pp_list=push_sobol_pp, type_mc="MCPS", mc_f_n=MCPS, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MC" in estimators:
            # Binomial
            ## Binomial pp
            time_start2 = time.time()
            binomial_pp = [PointPattern(points=support_window.rand(n=nb_point_output), window=support_window)
                            for _ in range(nb_samples)]
            time_end = time.time() - time_start2
            time_mc["MC"]=[int(time_end/60), time_end%60]
            ## MC classic
            MC = mc_results_n(pp_list=binomial_pp, type_mc="MC", mc_f_n=MC, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MCR" in estimators:
            # Binomial ranodom
            ## Binomial pp big
            time_start2_ = time.time()
            binomial_pp_big = _binomial_pp_ball(d, window=support_window, nb_point=n, nb_sample=nb_samples)
            binomial_pp_res = [p.restrict_to_window(support_window) for p in binomial_pp_big]
            time_end = time.time() - time_start2_
            time_mc["MCR"]=[int(time_end/60), time_end%60]
            ### MC classique random
            MCR = mc_results_n(pp_list=binomial_pp_res, type_mc="MCR", mc_f_n=MCR, fct_list=fct_list, fct_names=fct_names,exact_integrals=exact_integrals)

        if "MCDPP" in estimators:
            # DPP Bardenet Hardy
            ## DPP pp
            time_start3 = time.time()
            if pool_dpp:
                freeze_support()
                with Pool(processes=nb_core) as pool:
                    #print("Number of processes in the DPP pool ",pool._processes)
                    dpp_points = pool.starmap(sample_dpp, [(d, nb_point_output)]*nb_samples)
                    pool.close()
                    pool.join()


            else:
                dpp_points = [sample_dpp(d, nb_point_output) for _ in range(nb_samples)]
            #rescale points to be in support_window
            dpp_pp_scaled = [PointPattern(p/2, window=support_window) for p in dpp_points]
            ##MCDPP
            #scaled weight
            weights_dpp = [_mcdpp_weights(p, eval_pointwise=True)
                        for p in dpp_points]
            time_end = time.time() - time_start3
            time_mc["MCDPP"]=[int(time_end/60), time_end%60]
            MCDPP = mc_results_n(pp_list=dpp_pp_scaled, type_mc="MCDPP", mc_f_n=MCDPP,
                                fct_list=fct_list,
                                fct_names=fct_names,exact_integrals=exact_integrals,
                                weights=weights_dpp)

        if "RQMC" in estimators:
            #RQMC
            ## Scrambeled Sobol pp
            time_start4 = time.time()
            sobol_points_list = [generate_scramble_sobol_sample(window=support_window, nb_points=nb_point_output)
                                for _ in range(nb_samples)]
            sobol_pp = [PointPattern(p, window=support_window) for p in sobol_points_list]
            time_end = time.time() - time_start4
            time_mc["RQMC"]=[int(time_end/60), time_end%60]
            ## RQMC
            RQMC = mc_results_n(pp_list=sobol_pp, type_mc="RQMC", mc_f_n=RQMC, fct_list=fct_list, fct_names=fct_names, exact_integrals=exact_integrals)

        if "MCKS_h0" in estimators:
            time_start9 = time.time()
            MCKS_h0= mc_results_n(pp_list=binomial_pp, type_mc="MCKS_h0",
                                mc_f_n=MCKS_h0,
                                fct_list=fct_list,
                                fct_names=fct_names,
                                exact_integrals=exact_integrals,
                                correction=False)
            time_end = time.time() - time_start9
            time_mc["MCKS_h0"]=[int(time_end/60), time_end%60]


        if "MCKSc_h0" in estimators:
            time_start10 = time.time()
            MCKSc_h0= mc_results_n(pp_list=binomial_pp,
                                    type_mc="MCKSc_h0",
                                    mc_f_n=MCKSc_h0,
                                    fct_list=fct_list,
                                    fct_names=fct_names,
                                    exact_integrals=exact_integrals,
                                    correction=True)
            time_end = time.time() - time_start10
            time_mc["MCKSc_h0"]=[int(time_end/60), time_end%60]

        if "MCCV" in estimators:
            #MC Control variate
            time_start11 = time.time()
            #support_window_cv = support_integrands_ball(d)
            binomial_pp = [PointPattern(points=support_window.rand(n=nb_point_output), window=support_window)
                            for _ in range(nb_samples)]
            MCCV = mc_results_n(pp_list=binomial_pp, type_mc="MCCV",
                                       mc_f_n=MCCV, fct_list=fct_list,fct_names=fct_names, exact_integrals=exact_integrals, nb_point_cv=nb_point_cv, support_window_cv=support_window)
            time_end = time.time() - time_start11
            time_mc["MCCV"]=[int(time_end/60), time_end%60]

        print("----------------------------------------------")
        if "MCPS" in estimators:
            print("N expected=", n, ", N obtained", nb_point_output, ", N sobol obtained =", nb_point_output_sobol)
        else :
            print("N expected=", n, ", N obtained", nb_point_output)
        print("Time =", time_mc)
        print("----------------------------------------------")

    time_2 = time.time() - time_1
    print("Time all", int(time_2/60), "min", time_2%60, "s")
    for k in estimators:
        results[k] = locals()[k]
    if file_name is not None:
        with open(file_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results, nb_points_used

def mc_results_n( pp_list, type_mc, fct_list,
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
            mc_f_n["mc_results_" +name] = mc_f_dict(type_mc=type_mc)
    i=0
    if type_mc=="MCCV":
        points_cv_proposal= support_window_cv.rand(n=nb_point_cv, seed=0)
        points_cv_param_estimate= support_window_cv.rand(n=nb_point_cv, seed=1)
    for f,name in zip(fct_list, fct_names):
        if type_mc=="MCCV":
            proposal, m_proposal = estimate_control_variate_proposal(points=points_cv_proposal, f=f)
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
            mc_values = [delyon_portier_mc(point_pattern=p, f=f,
                                                    bandwidth=bandwidth_0_delyon_portier(p.points),
                                                   correction=correction)
                        for p in pp_list]
        elif type_mc in ["MC", "MCR", "MCP", "MCPS", "RQMC"]:
            mc_values = [monte_carlo_method(points=p.points, f=f) for p in pp_list]
        else:
            raise ValueError("The possible Monte Carlo methods are : MC, MCR, MCP, MCPS, RQMC, MCDPP, MCCV, MCKS_h0, MCKSc_h0, MCKS, MCKSc.")
        #print(mc_f_n["mc_results_f_{}".format(i)].keys(), type_mc)
        # mean MC method of type 'type_mc'
        mean_mc = stat.mean(mc_values)
        mc_f_n["mc_results_" + name]["m_"+ type_mc].append(stat.mean(mc_values))
        # var MC method of type 'type_mc'
        var_mc = stat.mean((mc_values - mean_mc)**2)
        mc_f_n["mc_results_" + name]["std_"+ type_mc].append(math.sqrt(var_mc))
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
                print("MSE=", mse(m_list, std_list, integ_f))
        i+=1
    return mc_f_n

# data frame of the MSE of MC methods for N = nb_point_list[idx_nb_points]
def dataframe_mse_results(mc_results, fct_names, exact_integrals, nb_sample, idx_nb_point=-1):
    type_mc = mc_results.keys()
    mse_dict={}
    for name_f, integ_f in zip(fct_names, exact_integrals):
        mse_dict["MSE("+ name_f + ")"]={}
        mse_dict["std(MSE(" + name_f + "))"]={}
        for t in type_mc:
            m_f = mc_results[t]["mc_results_" + name_f]["m_"+ t]
            std_f = mc_results[t]["mc_results_"+ name_f]["std_"+ t]
            mse_f = mse(m_f, std_f, integ_f, verbose=False)
            std_mse = np.array(std_f/np.sqrt(nb_sample))
            mse_dict["MSE(" + name_f + ")"][t] = mse_f[idx_nb_point]
            mse_dict["std(MSE("+ name_f +"))"][t] = std_mse[idx_nb_point]
    return pd.DataFrame(mse_dict)

# Kolmogorov Smirniv test for residual of the liear regression to test if the residual is Gaussian
# q-q plot pf the residual of the linear regresssion of log(std) w.r.t. log(N)
def dataframe_residual_test(mc_list, nb_point_list, fct_names, test_type="SW", **kwargs):
    result_test_dict = {}
    type_mc = mc_list.keys()
    for name in fct_names :
        result_test_f_dict = {}
        for t in type_mc:
            std_f = mc_list[t]["mc_results_"+ name]["std_"+ t]
            _, _, _, _, result_test = regression_line(nb_point_list, std_f, residual=True,residual_normality_test=test_type, **kwargs)
            result_test_f_dict[t] =  ("stat={0:.3f}".format(result_test[0]), "p={0:.3}".format(result_test[1]))
        result_test_dict[name] = result_test_f_dict
    return pd.DataFrame(result_test_dict)


# Mann-Whitney test for the (square) errors of the method of type 'type_mc_test' with the others methods
def dataframe_error_test(mc_list, nb_point_list, fct_name, type_mc_test="MCP"):
    mw_test_dict = {}
    type_mc = list(mc_list.keys())
    nb_nb_points = len(nb_point_list)
    #MC methods to be tested with type_mc_to_test
    type_mc.remove(type_mc_test)
    error_test = mc_list[type_mc_test]["mc_results_"+fct_name]["error_"+ type_mc_test]
    for t in type_mc:
        error_tested_with = mc_list[t]["mc_results_"+fct_name]["error_"+ t]
        mw_test_N_dict = {}
        for n in range(nb_nb_points):
            mw_test = stats.mannwhitneyu(error_test[n], error_tested_with[n])
            mw_test_N_dict["N={}".format(nb_point_list[n])] =  ("stat={0:.3f}".format(mw_test[0]), "p={0:.3}".format(mw_test[1]))
        mw_test_dict[type_mc_test+ " and "+ t] = mw_test_N_dict
    return pd.DataFrame(mw_test_dict)

def generate_repelled_binomial(d, support_window, nb_point, nb_sample, father_type="Binomial", add_r=None, epsilon=None, **kwargs):
    time_start = time.time()
    if father_type =="Binomial":
        father_pp_list = _binomial_pp_ball(d, window=support_window, nb_point=nb_point, nb_sample=nb_sample, add_r=add_r)
    elif father_type =="Sobol":
        father_pp_list = _sobol_pp_ball(d, window=support_window, nb_point=nb_point, nb_sample=nb_sample, add_r=add_r)
        #print("herere", len(father_pp_list))
    else:
        raise ValueError("type are Binomial and Sobol")
    gpp_pp = [RepelledPointProcess(p) for p in father_pp_list]
    if epsilon is None:
        epsilon = gpp_pp[0].epsilon_critical
    print("N Big=", father_pp_list[0].points.shape[0],
          ", N expected =", nb_point, ", Epsilon=", epsilon)
    #time_start = time.time()
    push_pp_big = [g.repelled_point_pattern(epsilon=epsilon,
                                            **kwargs)
                    for g in gpp_pp]
    push_pp = [g.restrict_to_window(support_window) for g in push_pp_big]
    time_end = time.time() - time_start
    print("Time Push=", int(time_end/60), "min", time_end%60, "s")
    return push_pp

def sample_dpp(d, nb_point):
    jac_params = np.array([[0, 0]]*d) #jaccobi measure=1
    dpp = MultivariateJacobiOPE(nb_point, jac_params)
    return dpp.sample()

def _mcdpp_weights(points, eval_pointwise=True, scale=None, jacobi_params=None):
    nb_points, d = points.shape
    if scale is None:
        scale = 1/2**d #for support equal [-1/2,1/2]^d
    if jacobi_params is None:
        jacobi_params = np.array([[0, 0]]*d) #jaccobi measure=1
    dpp = MultivariateJacobiOPE(nb_points, jacobi_params)
    weights_dpp = scale/dpp.K(points, eval_pointwise=eval_pointwise)
    return weights_dpp

#! done
def _binomial_pp_ball(d, window, nb_point, nb_sample, add_r=None):
    r"""Binomial in a ball of radius r containing window, used to obtaine a pushed binomial process.
    r = smallest r possible + epsilon
    """
    if isinstance(window, BoxWindow):
        l = min(np.diff(window.bounds, axis=1)) #length side window
        r=math.sqrt(d*(l/2)**2) #radius ball window containing box_window
    elif isinstance(window, BallWindow):
        r = window.radius
    else:
        print("Restart kernel")
    if add_r is None:
        add_r=0
    r += add_r
    rho = nb_point/window.volume
    simu_window = BallWindow(center=[0]*d, radius=r) #simulation window
    simu_nb_point = int(rho*simu_window.volume) #simulation nb points
    binomial_pp = [PointPattern(points=simu_window.rand(n=simu_nb_point),
                                    window=simu_window)
                       for _ in range(nb_sample)]
    return binomial_pp

#! done
def _sobol_pp_ball(d, window, nb_point, nb_sample, epsilon=0):
    r"""Sobol in a ball of radius r containing window, used to obtaine a pushed sobol process.
    r = smallest r possible + epsilon
    """
    if isinstance(window, BoxWindow):
        l = min(np.diff(window.bounds, axis=1))[0] #length side window
        r=math.sqrt(d)*(l/2) + epsilon #radius ball window containing box_window
    elif isinstance(window, BallWindow):
        r = window.radius + epsilon
    rho = nb_point/window.volume
    simu_window = BallWindow(center=[0]*d, radius=r) #required ball window
    simu_nb_point = int(rho*simu_window.volume) #simulation nb points
    sobol_pp = [sobol_point_pattern(window =simu_window,
                                    nb_points=simu_nb_point)
                       for _ in range(nb_sample)]
    return sobol_pp


# def _residual_log_lin_reg_std_mc(mc_list, nb_point_list, mc_type, fct_name, test_type="KS"):
#     std = mc_list[mc_type]["mc_results_" + fct_name]["std_"+ mc_type]
#     _,_, _, residual, test_result = regression_line(nb_point_list, std, log=True, residual=True, residual_normality_test=test_type)
#     return residual, test_result
