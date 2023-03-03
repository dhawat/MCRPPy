from GPPY.gravity_point_process import GravityPointProcess
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
import math
from GPPY.numerical_integration import (monte_carlo_integration,
                                        sobol_sequence,
                                        sobol_point_pattern,
                                        control_variate_integration,
                                        delyon_portier_integration,
                                       bandwidth_0_delyon_portier)
import statistics as stat
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from GPPY.spatial_windows import BallWindow, BoxWindow
from GPPY.point_pattern import PointPattern
from GPPY.monte_carlo_test_functions import (f_1, f_2, f_3, f_4,
                                             f_5,
                                             exact_integral_f_1,
                                             exact_integral_f_2,
                                             exact_integral_f_3,
                                             exact_integral_f_4,
                                             exact_integral_f_5
                                             )


def mc_results(d, nb_point_list, nb_sample, nb_function, support_window, estimators=None, add_r_push=None, **kwargs):

    if estimators is None:
        estimators = ["MC", "MCR", "MCP", "MCPS", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
    else :
        estimators_all = ["MC", "MCR", "MCP", "MCPS", "MCDPP",
                      "MCKS_h0", "MCKSc_h0", "RQMC", "MCCV"]
        if sum([k not in estimators_all for k in estimators])!=0:
            raise ValueError("the allowed estimators are {}".format(estimators_all))

    print("d=", d, ", nb points simu=", nb_point_list, ", nb samples=", nb_sample)
    print("------------------------------------------------")
    results = {}
    N_output_list=[]
    time_1 = time.time()
    MCP, MC, RQMC, MCCV, MCDPP= None, None, None, None, None
    MCR = None
    MCPS = None
    MCKS_h0, MCKSc_h0 =None,None
    #mcks, mcksc =None,None
    for n in nb_point_list :
        time_mc = {k:0 for k in estimators}
        # Push Binomial
        ## Push Binomial pp
        time_start1 = time.time()
        push_pp = samples_push(d, support_window=support_window, nb_point=n, nb_sample=nb_sample, add_r=add_r_push, **kwargs)
        time_end = time.time() - time_start1
        time_mc["MCP"]=[int(time_end/60), (time_end%60)]
        nb_point_output= int(stat.mean([p.points.shape[0] for p in push_pp]))
        N_output_list.append(nb_point_output)
        ## MCP
        MCP = mc_n_samples(pp_list=push_pp, type_mc="MCP", mc_f_n=MCP, nb_function=nb_function)

        if "MCPS" in estimators:
            # Push Sobol
            ## Push Sobol pp
            time_start1 = time.time()
            push_sobol_pp = samples_push(d, support_window=support_window, nb_point=n, nb_sample=nb_sample, father_type="Sobol", add_r=add_r_push, **kwargs)
            nb_point_output_sobol= int(stat.mean([p.points.shape[0] for p in push_sobol_pp]))
            time_end = time.time() - time_start1
            time_mc["MCPS"]=[int(time_end/60), (time_end%60)]
            ## MCPS
            MCPS = mc_n_samples(pp_list=push_sobol_pp, type_mc="MCPS", mc_f_n=MCPS, nb_function=nb_function)

        if "MC" in estimators:
            # Binomial
            ## Binomial pp
            time_start2 = time.time()
            binomial_pp = [PointPattern(points=support_window.rand(n=nb_point_output), window=support_window)
                            for _ in range(nb_sample)]
            time_end = time.time() - time_start2
            time_mc["MC"]=[int(time_end/60), time_end%60]
            ## MC classic
            MC = mc_n_samples(pp_list=binomial_pp, type_mc="MC", mc_f_n=MC, nb_function=nb_function)

        if "MCR" in estimators:
            # Binomial ranodom
            ## Binomial pp big
            time_start2_ = time.time()
            binomial_pp_big = _binomial_pp_ball(d, window=support_window, nb_point=n, nb_sample=nb_sample)
            binomial_pp_res = [p.restrict_to_window(support_window) for p in binomial_pp_big]
            time_end = time.time() - time_start2_
            time_mc["MCR"]=[int(time_end/60), time_end%60]
            ### MC classique random
            MCR = mc_n_samples(pp_list=binomial_pp_res, type_mc="MCR", mc_f_n=MCR, nb_function=nb_function)

        if "MCDPP" in estimators:
            # DPP Bardenet Hardy
            ## DPP pp
            time_start3 = time.time()
            #jac_params = -0.5 + np.random.rand(d, 2)
            jac_params = np.array([[0, 0]]*d) #jaccobi measure=1
            dpp = MultivariateJacobiOPE(nb_point_output, jac_params)
            dpp_points = [dpp.sample() for _ in range(nb_sample)]
            #rescale points to be in support_window
            dpp_pp_scaled = [PointPattern(p/2, window=support_window) for p in dpp_points]
            #scaled weight
            ##MCDPP
            weights_dpp = [1/(2**d*dpp.K(p, eval_pointwise=True))
                        for p in dpp_points]
            time_end = time.time() - time_start3
            time_mc["MCDPP"]=[int(time_end/60), time_end%60]
            MCDPP = mc_n_samples(pp_list=dpp_pp_scaled, type_mc="MCDPP", mc_f_n=MCDPP,
                                nb_function=nb_function,
                                weights=weights_dpp)

        if "RQMC" in estimators:
            #RQMC
            ## Scrambeled Sobol pp
            time_start4 = time.time()
            sobol_points_list = [sobol_sequence(window=support_window, nb_points=nb_point_output)
                                for _ in range(nb_sample)]
            sobol_pp = [PointPattern(p, window=support_window) for p in sobol_points_list]
            time_end = time.time() - time_start4
            time_mc["RQMC"]=[int(time_end/60), time_end%60]
            ## RQMC
            RQMC = mc_n_samples(pp_list=sobol_pp, type_mc="RQMC", mc_f_n=RQMC, nb_function=nb_function)

        if "MCKS_h0" in estimators:
            time_start9 = time.time()
            MCKS_h0= mc_n_samples(pp_list=binomial_pp, type_mc="MCKS_h0",
                                mc_f_n=MCKS_h0,
                                nb_function=nb_function,
                                correction=False)
            time_end = time.time() - time_start9
            time_mc["MCKS_h0"]=[int(time_end/60), time_end%60]


        if "MCKSc_h0" in estimators:
            time_start10 = time.time()
            MCKSc_h0= mc_n_samples(pp_list=binomial_pp,
                                    type_mc="MCKSc_h0",
                                    mc_f_n=MCKSc_h0,
                                    nb_function=nb_function,
                                    correction=True)
            time_end = time.time() - time_start10
            time_mc["MCKSc_h0"]=[int(time_end/60), time_end%60]

        if "MCCV" in estimators:
            #MC Control variate
            time_start11 = time.time()
            MCCV = mc_n_samples(pp_list=binomial_pp, type_mc="MCCV", mc_f_n=MCCV, nb_function=5)
            time_end = time.time() - time_start11
            time_mc["MCCV"]=[int(time_end/60), time_end%60]

        # MC kernel smoothing corrected (Delyon Portier)
        #! commented because of its computational complexity
        ##MC kernel smoothing corrected
        # time_start10 = time.time()
        # mcksc= mc_n_samples(pp_list=binomial_pp, type_mc="MCKSc", mc_f_n=mcksc,
        #                     nb_function=nb_function,
        #                    correction=True)
        # time_end = time.time() - time_start10
        # time_mc["MCKSc"]=[int(time_end/60), time_end%60]
        ##MC kernel smoothing corrected using h0
         # MC kernel smoothing (Delyon Portier)
        # time_start9 = time.time()
        # mcks= mc_n_samples(pp_list=binomial_pp, type_mc="MCKS", mc_f_n=mcks,
        #                    nb_function=nb_function,
        #                    correction=False)
        # time_end = time.time() - time_start9
        # time_mc["MCKS"]=[int(time_end/60), time_end%60]
        ##MC kernel smoothing (Delyon Portier) using h0
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
    return results, N_output_list


# data frame of the MSE of MC methods for N = nb_point_list[idx_nb_points]
def dataframe_mse_results(d, mc_results, nb_function, nb_sample, idx_nb_point=-1):
    type_mc = mc_results.keys()
    mse_dict={}
    for j in range(1, nb_function+1):
        mse_dict["MSE(f_{})".format(j)]={}
        mse_dict["std(MSE(f_{}))".format(j)]={}
        for t in type_mc:
            integ_f = globals()["exact_integral_f_{}".format(j)](d)
            if t!="MCCV" or j<6:
                m_f = mc_results[t]["mc_results_f_{}".format(j)]["m_"+ t]
                std_f = mc_results[t]["mc_results_f_{}".format(j)]["std_"+ t]
                mse_f = mse(m_f, std_f, integ_f, verbose=False)
                std_mse = np.array(std_f/np.sqrt(nb_sample))
                mse_dict["MSE(f_{})".format(j)][t] = mse_f[idx_nb_point]
                mse_dict["std(MSE(f_{}))".format(j)][t] = std_mse[idx_nb_point]
    return pd.DataFrame(mse_dict)

# Kolmogorov Smirniv test for residual of the liear regression to test if the residual is Gaussian
def dataframe_residual_test(mc_list, nb_point_list, **kwargs):
    result_test_dict = {}
    nb_function = len(mc_list["MC"])
    type_mc = mc_list.keys()
    for j in range(1, nb_function+1) :
        result_test_f_dict = {}
        for t in type_mc:
            std_f = mc_list[t]["mc_results_f_{}".format(j)]["std_"+ t]
            _, _, _, _, result_test = regression_line(nb_point_list, std_f, test_stat=True, **kwargs)
            result_test_f_dict[t] =  ("stat={0:.3f}".format(result_test[0]), "p={0:.3}".format(result_test[1]))
        result_test_dict["f_{}".format(j)] = result_test_f_dict
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

def mc_n_samples( pp_list, type_mc, mc_f_n=None, nb_function=5,
                 weights=None, correction=True, verbose=True):
    d= pp_list[0].window.dimension
    print("For", type_mc)
    print("---------------")
    if mc_f_n is None:
        mc_f_n = {}
        for i in range(1, nb_function+1):
            mc_f_n["mc_results_f_{}".format(i)] = mc_f_dict(type_mc=type_mc)
    for i in range(1,nb_function+1):
        f = globals()["f_{}".format(i)]
        integ_f = globals()["exact_integral_f_{}".format(i)](d)
        if type_mc=="MCCV":
            proposal, m_proposal = globals()["cv_proposal_f_{}".format(i)](d)
            mc_values=[control_variate_integration(points=p.points,
                                                   f=f,
                                                   proposal=proposal,
                                                   mean_proposal= m_proposal)
                      for p in pp_list]
        elif type_mc=="MCDPP":
            mc_values = [monte_carlo_integration(points=p.points, f=f, weights=w)
                         for (p,w) in zip(pp_list, weights)]
        elif type_mc in ["MCKS", "MCKSc"]:
            mc_values = [delyon_portier_integration(point_pattern=p,
                                                    f=f,
                                                   correction=correction)
                        for p in pp_list]
        elif type_mc in ["MCKS_h0", "MCKSc_h0"]:
            mc_values = [delyon_portier_integration(point_pattern=p, f=f,
                                                    bandwidth=bandwidth_0_delyon_portier(p.points),
                                                   correction=correction)
                        for p in pp_list]
        elif type_mc in ["MC", "MCR", "MCP", "MCPS", "RQMC"]:
            mc_values = [monte_carlo_integration(points=p.points, f=f) for p in pp_list]
        else:
            raise ValueError("Wrong MC type.")
        #print(mc_f_n["mc_results_f_{}".format(i)].keys(), type_mc)
        # mean MC method of type 'type_mc'
        mean_mc = stat.mean(mc_values)
        mc_f_n["mc_results_f_{}".format(i)]["m_"+ type_mc].append(stat.mean(mc_values))
        # var MC method of type 'type_mc'
        var_mc = stat.mean((mc_values - mean_mc)**2)
        mc_f_n["mc_results_f_{}".format(i)]["std_"+ type_mc].append(math.sqrt(var_mc))
        # square erreur MC method of type 'type_mc'
        mc_f_n["mc_results_f_{}".format(i)]["error_"+ type_mc].append(error(mc_values, integ_f))
        # print MSE
        if verbose:
            print("FOR f%s"%i)
            m_list = mc_f_n["mc_results_f_{}".format(i)]["m_"+ type_mc]
            std_list = mc_f_n["mc_results_f_{}".format(i)]["std_"+ type_mc]
            print("MSE=", mse(m_list, std_list, integ_f))
    return mc_f_n

def samples_push(d, support_window, nb_point, nb_sample, father_type="Binomial", add_r=None, **kwargs):
    time_start = time.time()
    if father_type =="Binomial":
        father_pp_list = _binomial_pp_ball(d, window=support_window, nb_point=nb_point, nb_sample=nb_sample, add_r=add_r)
    elif father_type =="Sobol":
        father_pp_list = _sobol_pp_ball(d, window=support_window, nb_point=nb_point, nb_sample=nb_sample, add_r=add_r)
        #print("herere", len(father_pp_list))
    else:
        raise ValueError("type are Binomial and Sobol")
    gpp_pp = [GravityPointProcess(p) for p in father_pp_list]
    epsilon_0 = gpp_pp[0].epsilon_critical
    print("N Big=", father_pp_list[0].points.shape[0],
          ", N expected =", nb_point, ", Epsilon=", epsilon_0)
    #time_start = time.time()
    push_pp_big = [g.pushed_point_pattern(epsilon=epsilon_0,
                                            **kwargs)
                    for g in gpp_pp]
    push_pp = [g.restrict_to_window(support_window) for g in push_pp_big]
    time_end = time.time() - time_start
    print("Time Push=", int(time_end/60), "min", time_end%60, "s")
    return push_pp

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
        add_r=r
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

def jaccobi_measure(x, jac_params):
    d = x.shape[1]
    support_window = BoxWindow([[-1,1]]*d)
    alpha, betta = jac_params[:, 0], jac_params[:, 1]
    a = np.zeros_like(x)
    for i in range(d):
        a[:,i]= (1-x[:,i])**alpha[i]*(1+x[:,i])**betta[i]
    result = np.prod(a, axis=1)
    support = support_window.indicator_function(x)*1
    return result*support


def error(approx, exact):
    return np.array(approx) - exact

def mse(mean, std, exact, verbose=True):
    #print(mean)
    var = np.square(std)
    bias_square = np.square(np.array(mean)- exact)
    if verbose:
        print( "Bias=", bias_square, "Var=", var)
    return var + bias_square

# def pd_square_error(N, approx, exact):
#     se = approx - exact
#     return (pd.DataFrame({"{}".format(N): se}))

def regression_line(x, y, log=True, test_stat=False, test_type="KS"):
    if log:
        x = np.log(x)
        y = np.log(y)
    else:
        x = np.array(x)
        y = np.array(y)
    #reg = LinearRegression().fit(x, y)
    #slope = reg.coef_
    reg_fit = stats.linregress(x,y)
    slope = reg_fit.slope
    std_slope = reg_fit.stderr
    reg_line = x*slope + reg_fit.intercept
    if test_stat:
        #Kolmogorov-Smirnov or Shapiro-Wilk test on the resedual of linear regressian to determine if it came from a normal distribution
        residual = y - reg_line # residual r = y - estimated_y
        if test_type=="KS":
            test_result = stats.kstest(residual, 'norm')
        elif test_type=="SW":
            test_result = stats.shapiro(residual)
        return reg_line, slope, std_slope, residual, test_result
    else:
        return reg_line, slope, std_slope

def _residual_log_lin_reg_std_mc(mc_list, nb_point_list, mc_type, fct_name, test_type="KS"):
    std = mc_list[mc_type]["mc_results_" + fct_name]["std_"+ mc_type]
    _,_, _, residual, test_result = regression_line(nb_point_list, std, log=True, test_stat=True, test_type=test_type)
    return residual, test_result

def mc_f_dict(type_mc, se=True):
    d = {}
    d["m_"+type_mc]=[]
    d["std_"+type_mc]=[]
    if se:
        d["error_"+type_mc]=[]
    return d

def plot_mc_results(d, mc_list, nb_point_list, nb_sample, log_scale=True, save_fig=None, plot_dim=2, error_type="SE",  plot_std=True, plot_error=True, plot_fct=False):
    nb_function = len(mc_list["MC"])
    type_mc = mc_list.keys()
    nb_column = _nb_column_plot(plot_std, plot_error, plot_fct)
    color_list = ["b", "k", "g", "m", "gray", "c","y", "darkred", "orange", "pink"]
    fig= plt.figure(figsize=(int(4*nb_column),int(3*nb_function)))
    for j in range(1, nb_function+1) :
        #plot
        if plot_fct:
            add_plot_functions(fig, nb_function, plot_dim, idx_row=j, nb_column=nb_column)
        if plot_std:
            #std
            ax = fig.add_subplot(nb_function, nb_column, nb_column + nb_column*(j-1))
            add_plot_std(d, ax, mc_list, nb_point_list, color_list, idx_row=j)
            if plot_error:
                #error
                ax = fig.add_subplot(nb_function, nb_column, nb_column+ nb_column*(j-1))
                add_plot_error(d, ax, mc_list, type_mc, nb_sample, nb_point_list, error_type, color_list, idx_row=j, log_scale=log_scale)
        else:
            ax = fig.add_subplot(nb_function, nb_column, nb_column+ nb_column*(j-1))
            add_plot_error(d, ax, mc_list, type_mc, nb_sample, nb_point_list, error_type, color_list, idx_row=j, log_scale=log_scale)
        plt.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches='tight')
    plt.show()
    #return ax

def add_plot_functions(fig, nb_function, plot_dim, idx_row, nb_column):
    if plot_dim==2:
        x = np.linspace(-1,1, 100)
        X, Y = np.meshgrid(x, x)
        points = np.array([X.ravel(), Y.ravel()]).T
        z_f = globals()["f_{}".format(idx_row)](points)
        #print("Hello", z_f)
        ax = fig.add_subplot(nb_function, nb_column, 1+ nb_column*(idx_row-1), projection='3d')
        ax.scatter3D(X.ravel(), Y.ravel(), z_f, c=z_f)
    elif plot_dim==1:
        x = np.linspace(-1,1, 300)
        y_f = globals()["f_{}".format(idx_row)](np.atleast_2d(x).T)
        ax = fig.add_subplot(nb_function, nb_column, 1+ nb_column*(idx_row-1))
        ax.plot(x, y_f)
    ax.set_title(r"$f_%s$"%idx_row)

def add_plot_std(d, ax, mc_list, nb_point_list, color_list, idx_row):
    log_nb_pts = np.log([nb_point_list]).T
    type_mc = mc_list.keys()
    i=0
    for t in type_mc:
        if t!="MCCV" or idx_row<6:
            std_f = mc_list[t]["mc_results_f_{}".format(idx_row)]["std_"+ t]
            reg_line, slope, std_reg = regression_line(nb_point_list, std_f)
            label_with_slope = t+": slope={0:.2f}".format(slope)+ ", std={0:.2f}".format(std_reg)
            ax.scatter(log_nb_pts, np.log(std_f), c=color_list[i],s=3, label=label_with_slope)
            ax.plot(log_nb_pts, reg_line, c=color_list[i])
            #ax.plot(log_nb_pts, reg_line + 3*std_reg, c=col[i], linestyle=(0, (5, 10)))
            #ax.plot(log_nb_pts, reg_line - 3*std_reg, c=col[i], linestyle=(0, (5, 10)))
            i=i+1
    ax.set_title("std (d=%s)" %d + r" for $f_{}$".format(idx_row))
    ax.set_xlabel(r"$\log(N) $ ")
    ax.set_ylabel(r"$\log(std)$")
    ax.legend()

def add_plot_error(d, ax, mc_list, type_mc, nb_sample, nb_point_list, error_type, color_list, idx_row, log_scale):
    i=0
    for t in type_mc:
        integ_f = globals()["exact_integral_f_{}".format(idx_row)](d)
        if t!="MCCV" or idx_row<6:
            if error_type=="MSE":
                m_f = mc_list[t]["mc_results_f_{}".format(idx_row)]["m_"+ t]
                std_f = mc_list[t]["mc_results_f_{}".format(idx_row)]["std_"+ t]
                mse_f = mse(m_f, std_f, integ_f, verbose=False)
                err_bar = np.array(std_f/np.sqrt(nb_sample))
                if log_scale:
                    ax.loglog(np.array(nb_point_list) +25*i, mse_f, c=color_list[i], marker=".", label=t)
                else:
                    ax.plot(np.array(nb_point_list) +25*i, mse_f, c=color_list[i], marker=".", label=t)
                ax.errorbar(x=np.array(nb_point_list) +25*i, y=mse_f, yerr=3 *err_bar,
                            color=color_list[i], capsize=4, capthick=1, elinewidth=6)
            if error_type in ["SE", "Error"]:
                error_f = mc_list[t]["mc_results_f_{}".format(idx_row)]["error_"+ t]
                if error_type=="SE":
                    error_f = [e**2 for e in error_f]
                x = np.array(nb_point_list) +25*i
                nb_list_expended = [[n]*len(e) for n,e in zip(nb_point_list, error_f)]
                ax.scatter(np.array(nb_list_expended) +25*i, error_f, c=color_list[i], s=0.1, label=t)
                ax.boxplot(error_f, positions=x.tolist(),
                            widths = 30,
                            manage_ticks=False,
                            patch_artist=True, boxprops=dict(facecolor=color_list[i]),
                            whiskerprops=dict(color=color_list[i]),
                            #showmeans=True,
                            #meanprops=dict(marker='.', color='r', markeredgecolor='r'),
                            sym='',
                            )
                #ax.legend([a["boxes"][0]], [t], loc='lower left')
        i=i+1
    if log_scale:
        #ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(error_type + " (d=%s)" %d + r" for $f_{}$".format(idx_row))
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(error_type)
    ax.legend()


def _nb_column_plot(plot_std, plot_error, plot_fct):
    if plot_std and plot_error:
        if plot_fct:
            nb_column=3
        else:
            nb_column=2
    else:
        if plot_fct:
            nb_column=2
        else: nb_column=1
    return nb_column
# def test_jaccobi_measure():
#     x = np.array([[1, 1/2, 0], [1/2, 0, 0], [0, 1.1, 0]])
#     #x= np.array([ [1/2, 0, 0]])
#     jac_params = np.array([[1, 1, 0], [2, 0, 1]]).T
#     expected = np.array([0, 9/8, 0])
#     if np.isclose(jaccobi_measure(x, jac_params), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, error=", jaccobi_measure(x,jac_params)- expected)
