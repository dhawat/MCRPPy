from GPPY.gravity_point_process import GravityPointProcess
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from GPPY.numerical_integration import (monte_carlo_integration,
                                        sobol_sequence,
                                        control_variate_integration,
                                        delyon_portier_integration,
                                       bandwidth_0_delyon_portier)
import statistics as stat
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.point_pattern import PointPattern
from GPPY.monte_carlo_test_functions import (f_1, f_2, f_3, f_4, f_5, f_6,
                                             exact_integral_f_1,
                                             exact_integral_f_2,
                                             exact_integral_f_3,
                                             exact_integral_f_4,
                                             exact_integral_f_5,
                                             exact_integral_f_6,
                                             cv_setup_f_1)


def mc_results(d, nb_point_list, nb_sample, nb_function, support_window, correction=False):
    print("d=", d, ", nb points simu=", nb_point_list, ", nb samples=", nb_sample)
    print("------------------------------------------------")
    N_output_list=[]
    time_1 = time.time()
    mcp, mc, rqmc, mccv, mcdpp= None, None, None, None, None
    mcks_h0, mcksc_h0 =None,None
    #mcks, mcksc =None,None
    for n in nb_point_list :
        time_mc = {"MC":0, "MCP":0, "MCDPP":0, "RQMC":0,
                   #"MCKS":0, "MCKSc":0,
                   "MCKS_h0":0, "MCKSc_h0":0,
                   "MCCV":0 }
        # Point pattern
        ## Push pp
        time_start1 = time.time()
        push_pp = samples_push(d, support_window=support_window, nb_point=n, nb_sample=nb_sample)
        time_end = time.time() - time_start1
        time_mc["MCP"]=[int(time_end/60), (time_end%60)]
        nb_point_output= int(stat.mean([p.points.shape[0] for p in push_pp]))
        N_output_list.append(nb_point_output)

        ## Binomial pp
        time_start2 = time.time()

        binomial_pp = [PointPattern(points=support_window.rand(n=nb_point_output), window=support_window)
                           for _ in range(nb_sample)]
        time_end = time.time() - time_start2
        time_mc["MC"]=[int(time_end/60), time_end%60]

        ## DPP pp
        time_start3 = time.time()
        #jac_params = -0.5 + np.random.rand(d, 2)
        jac_params = np.array([[0, 0]]*d) #jaccobi measure=1
        dpp = MultivariateJacobiOPE(nb_point_output, jac_params)
        dpp_points = [dpp.sample() for _ in range(nb_sample)]
        #rescale points to be in support_window
        dpp_pp_scaled = [PointPattern(p/2, window=support_window) for p in dpp_points]
        #scaled weight
        weights_dpp = [1/(2**d*dpp.K(p, eval_pointwise=True))
                       for p in dpp_points]
        time_end = time.time() - time_start3
        time_mc["MCDPP"]=[int(time_end/60), time_end%60]

        ## Scrambeled Sobol pp
        time_start4 = time.time()
        sobol_points_list = [sobol_sequence(window=support_window, nb_points=nb_point_output)
                             for _ in range(nb_sample)]
        sobol_pp = [PointPattern(p, window=support_window) for p in sobol_points_list]
        time_end = time.time() - time_start4
        time_mc["RQMC"]=[int(time_end/60), time_end%60]

        # MC
        ## MC classic
        print("N expected=", n, ", N obtained", nb_point_output)
        time_start5 = time.time()
        mc = mc_n_samples(pp_list=binomial_pp, type_mc="MC", mc_f_n=mc, nb_function=nb_function)
        time_end = time.time() - time_start5

        ## MCP
        time_start6 = time.time()
        mcp = mc_n_samples(pp_list=push_pp, type_mc="MCP", mc_f_n=mcp, nb_function=nb_function)
        time_end = time.time() - time_start6

        ## MCDPP (Bardenet Hardy)
        time_start7 = time.time()
        mcdpp = mc_n_samples(pp_list=dpp_pp_scaled, type_mc="MCDPP", mc_f_n=mcdpp,
                             nb_function=nb_function,
                             weights=weights_dpp)
        time_end = time.time() - time_start7
        #time_mc["MCDPP"]+=int(time_end/60)+ time_end%60

        ## RQMC
        time_start8 = time.time()
        rqmc = mc_n_samples(pp_list=sobol_pp, type_mc="RQMC", mc_f_n=rqmc, nb_function=nb_function)
        time_end = time.time() - time_start8

        #! commented because of its computational complexity
        ##MC kernel smoothing (Delyon Portier)
        # time_start9 = time.time()
        # mcks= mc_n_samples(pp_list=binomial_pp, type_mc="MCKS", mc_f_n=mcks,
        #                    nb_function=nb_function,
        #                    correction=False)
        # time_end = time.time() - time_start9
        # time_mc["MCKS"]=[int(time_end/60), time_end%60]
        ##MC kernel smoothing (Delyon Portier) using h0
        time_start9 = time.time()
        mcks_h0= mc_n_samples(pp_list=binomial_pp, type_mc="MCKS_h0", mc_f_n=mcks_h0,
                           nb_function=nb_function,
                           correction=False)
        time_end = time.time() - time_start9
        time_mc["MCKS_h0"]=[int(time_end/60), time_end%60]
        if correction:
            #! commented because of its computational complexity
            ##MC kernel smoothing corrected
            # time_start10 = time.time()
            # mcksc= mc_n_samples(pp_list=binomial_pp, type_mc="MCKSc", mc_f_n=mcksc,
            #                     nb_function=nb_function,
            #                    correction=True)
            # time_end = time.time() - time_start10
            # time_mc["MCKSc"]=[int(time_end/60), time_end%60]
            ##MC kernel smoothing corrected using h0
            time_start10 = time.time()
            mcksc_h0= mc_n_samples(pp_list=binomial_pp, type_mc="MCKSc_h0", mc_f_n=mcksc_h0,
                                nb_function=nb_function,
                               correction=True)
            time_end = time.time() - time_start10
            time_mc["MCKSc_h0"]=[int(time_end/60), time_end%60]
        ##MC Control variate
        time_start11 = time.time()
        mccv = mc_n_samples(pp_list=binomial_pp, type_mc="MCCV", mc_f_n=mccv, nb_function=1)
        time_end = time.time() - time_start11
        time_mc["MCCV"]=[int(time_end/60), time_end%60]
        print("----------------------------------------------")
        print("Time =", time_mc)
        print("----------------------------------------------")
    time_2 = time.time() - time_1
    print("Time all", int(time_2/60), "min", time_2%60, "s")
    if correction:
        results = {"MC": mc, "MCP": mcp,
                   "MCDPP": mcdpp,
                   #"MCKS": mcks,
                   "MCKS_h0": mcks_h0,
                   #"MCKSc":mcksc,
                   "MCKSc_h0":mcksc_h0,
                   "RQMC": rqmc, "MCCV": mccv}
    else:
        results = {"MC": mc, "MCP": mcp,
                   "MCDPP": mcdpp,
                   #"MCKS": mcks,
                   "MCKS_h0": mcks_h0,
                   "RQMC": rqmc,
                   "MCCV": mccv}
    return results, N_output_list

def dataframe_mse_results(d, mc_results, nb_function, nb_sample, idx_nb_point=-1):
    type_mc = mc_results.keys()
    mse_dict={}
    for j in range(1, nb_function+1):
        mse_dict["MSE(f_{})".format(j)]={}
        mse_dict["std(MSE(f_{}))".format(j)]={}
        for t in type_mc:
            integ_f = globals()["exact_integral_f_{}".format(j)](d)
            if t!="MCCV" or j==1:
                m_f = mc_results[t]["mc_results_f_{}".format(j)]["m_"+ t]
                std_f = mc_results[t]["mc_results_f_{}".format(j)]["std_"+ t]
                mse_f = mse(m_f, std_f, integ_f)
                std_mse = np.array(std_f/np.sqrt(nb_sample))
                mse_dict["MSE(f_{})".format(j)][t] = mse_f[idx_nb_point]
                mse_dict["std(MSE(f_{}))".format(j)][t] = std_mse[idx_nb_point]
    return pd.DataFrame(mse_dict)

def mc_n_samples( pp_list, type_mc, mc_f_n=None, nb_function=5,
                 weights=None, correction=True, verbose=True):
    d= pp_list[0].window.dimension
    print("For", type_mc)
    print("---------------")
    if mc_f_n is None:
        mc_f_n = {}
        for i in range(1,nb_function+1):
            mc_f_n["mc_results_f_{}".format(i)] = mc_f_dict(type_mc=type_mc)
    for i in range(1,nb_function+1):
        f = globals()["f_{}".format(i)]
        integ_f = globals()["exact_integral_f_{}".format(i)](d)
        if type_mc=="MCCV":
            c, proposal, m_proposal = globals()["cv_setup_f_{}".format(i)](d)
            mc_values=[control_variate_integration(points=p.points, f=f, c=c, proposal=proposal,
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
        elif type_mc in ["MC", "MCP", "RQMC"]:
            mc_values = [monte_carlo_integration(points=p.points, f=f) for p in pp_list]
        else:
            raise ValueError("Wrong MC type.")
        #print(mc_f_n["mc_results_f_{}".format(i)].keys(), type_mc)
        mc_f_n["mc_results_f_{}".format(i)]["m_"+ type_mc].append(stat.mean(mc_values))
        mc_f_n["mc_results_f_{}".format(i)]["std_"+ type_mc].append(stat.stdev(mc_values))
        m_list = mc_f_n["mc_results_f_{}".format(i)]["m_"+ type_mc]
        std_list = mc_f_n["mc_results_f_{}".format(i)]["std_"+ type_mc]
        if verbose:
            print("FOR f%s"%i)
            print("bias=", error(m_list, integ_f), ", std=", std_list,)
            print("MSE=", mse(m_list, std_list, integ_f))
    return mc_f_n

def samples_push(d, support_window, nb_point, nb_sample, multiprocess=True, plot_fig=False):
    time_start = time.time()
    l = min(np.diff(support_window.bounds, axis=1)) #length side window
    r=np.sqrt(d*(l/2)**2) + 0.1 #radius ball window containing support_window
    simu_window = BallWindow(center=[0]*d, radius=r) #simulation window
    simu_nb_point = int(nb_point/support_window.volume*simu_window.volume) #simulation nb points
    binomial_pp_big = [PointPattern(points=simu_window.rand(n=simu_nb_point),
                                    window=simu_window)
                       for _ in range(nb_sample)]

    gpp_pp = [GravityPointProcess(p) for p in binomial_pp_big]
    epsilon_0 = gpp_pp[0].epsilon_critical
    print("N Big=", binomial_pp_big[0].points.shape[0],
          ", N expected =", nb_point, ", Epsilon=", epsilon_0)
    #time_start = time.time()
    push_pp_big = [g.pushed_point_pattern(epsilon=epsilon_0,
                                            multiprocess=multiprocess)
                    for g in gpp_pp]
    push_pp = [g.restrict_to_window(support_window) for g in push_pp_big]
    time_end = time.time() - time_start
    print("Time Push=", int(time_end/60), "min", time_end%60, "s")
    #plot
    if plot_fig:
        w_obs = BoxWindow([[-1/2, 1/2]]*2)
        _, axis = plt.subplots(1, 4, figsize=(14, 3))
        push_pp_big[0].plot(window=simu_window, axis= axis[0], c="g", s=0.2)
        push_pp[0].plot(window=support_window, axis= axis[0], c="k", s=0.2)
        push_pp[0].plot(window=support_window, axis= axis[1], c="k", s=0.2)
        w_obs.plot(axis=axis[0])
        axis[1].set_title("Push simu and restricted")
        axis[1].set_title("Push")
        #binomial_pp_big[0].plot(window=window_simulation, axis= axis[2], c="b", s=0.2)
        binomial_pp_big[0].plot(window=support_window, axis= axis[2], c="k", s=0.2)
        binomial_pp_big[0].plot(window=support_window, axis= axis[3], c="k", s=0.2)
        push_pp[0].plot(window=support_window, axis= axis[3], c="g", s=0.2)
        axis[2].set_title("Binomial")
        axis[3].set_title("Binomial and Push")
        plt.show()
    return push_pp

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

def mse(mean, std, exact):
    var = np.square(std)
    bias_square = np.square(error(mean, exact))
    return var + bias_square

def abs_error(approx, exact):
    return np.atleast_2d(np.abs(error(approx, exact)))

def regression_line(x, y, log=True):
    if log:
        x = np.log([x]).T
        y = np.log([y]).T
    else:
        x = np.array(x).T
        y = np.array(y).T
    std_reg = LinearRegression().fit(x, y)
    return x*std_reg.coef_ + std_reg.intercept_

def mc_f_dict(type_mc):
    d = {}
    d["m_"+type_mc]=[]
    d["std_"+type_mc]=[]
    return d

def plot_mc_results(d, mc_list, nb_point_list, nb_sample, save_fig=False):
    log_nb_pts = np.log([nb_point_list]).T
    nb_function = len(mc_list["MC"])
    type_mc = mc_list.keys()
    x = np.linspace(-1,1, 60)
    X, Y = np.meshgrid(x, x)
    points = np.array([X.ravel(), Y.ravel()]).T
    col = ["b", "k", "g", "m", "gray", "c","y", "darkred", "orange", "pink"]
    for j in range(1, nb_function+1) :
        z_f = globals()["f_{}".format(j)](points)
        fig= plt.figure(figsize=(14,18))
        #plot
        ax = fig.add_subplot(nb_function, 3, 1+ 3*(j-1), projection='3d')
        ax.scatter3D(X.ravel(), Y.ravel(), z_f, c=z_f)
        ax.set_title(r"$f_%s$"%j)
        #std
        ax = fig.add_subplot(nb_function, 3, 2+ 3*(j-1))
        i=0
        for t in type_mc:
            if t!="MCCV" or j==1:
                std_f = mc_list[t]["mc_results_f_{}".format(j)]["std_"+ t]
                ax.scatter(log_nb_pts, np.log(std_f), c=col[i],s=1, label=t)
                ax.plot(log_nb_pts, regression_line(nb_point_list, std_f), c=col[i])
                i=i+1
        ax.set_title("std (d=%s)" %d)
        ax.set_xlabel(r"$\log(N)$")
        ax.set_ylabel(r"$\log(std)$")
        ax.legend()
        #MSE
        ax = fig.add_subplot(nb_function, 3, 3+ 3*(j-1))
        i=0
        for t in type_mc:
            integ_f = globals()["exact_integral_f_{}".format(j)](d)
            if t!="MCCV" or j==1:
                m_f = mc_list[t]["mc_results_f_{}".format(j)]["m_"+ t]
                std_f = mc_list[t]["mc_results_f_{}".format(j)]["std_"+ t]
                mse_f = mse(m_f, std_f, integ_f)
                err_bar = np.array(std_f/np.sqrt(nb_sample))
                ax.plot(np.array(nb_point_list)+10*i, mse_f, c=col[i], marker=".", label=t)
                ax.errorbar(x=np.array(nb_point_list)+10*i, y=mse_f, yerr=3 *err_bar,
                             color=col[i], capsize=4, capthick=1, elinewidth=6)
            i=i+1

        ax.set_title("MSE (d=%s)" %d)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$MSE$")
        ax.legend()
        plt.tight_layout()
        if save_fig :
            plt.savefig("")
    plt.show()

# def test_jaccobi_measure():
#     x = np.array([[1, 1/2, 0], [1/2, 0, 0], [0, 1.1, 0]])
#     #x= np.array([ [1/2, 0, 0]])
#     jac_params = np.array([[1, 1, 0], [2, 0, 1]]).T
#     expected = np.array([0, 9/8, 0])
#     if np.isclose(jaccobi_measure(x, jac_params), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, error=", jaccobi_measure(x,jac_params)- expected)
