import numpy as np
import matplotlib.pyplot as plt
from mcrppy.utils import regression_line
import statsmodels.api as sm

def _plot_proposal(f, proposal, dim=2):
    x = np.linspace(-1/2,1/2, 100)
    if dim == 2:
        X, Y = np.meshgrid(x, x)
        z = np.array([X.ravel(), Y.ravel()]).T
        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(2, 6, 1, projection='3d')
        ax.set_title(r"$f$")
        ax.scatter3D(X.ravel(), Y.ravel(), f(z), c=f(z))
        ax = fig.add_subplot(2, 6, 2, projection='3d')
        ax.scatter3D(X.ravel(), Y.ravel(), proposal(z), c=proposal(z))
        ax.set_title("Control variate proposal")
        plt.show()
    else:
        raise ValueError("Actually, only available for 2D")

# plot monte_carlo_methods

def plot_mc_results(d, mc_list, nb_points_list, fct_list, fct_names, log_scale=False, save_fig=None, plot_dim=2, error_type="SE",  plot_std=True, plot_error=False, plot_fct=False, nb_subsample_nb_points=None, estimators=None):
    nb_fct = len(fct_list)
    if estimators is None:
        estimators = mc_list.keys()
    nb_column = _nb_column_plot(plot_std, plot_error, plot_fct)
    color_list = ["b", "k", "g", "m",  "orange", "gray", "c","y", "darkred", "pink"]
    marker_list = [ "^", "v", "<", ">", "*","o", "x", "1", ".",]
    fig= plt.figure(figsize=(int(5*nb_fct),int(4*nb_column)))
    for j in range(1, nb_fct+1) :
        #plot
        if plot_fct:
            add_plot_functions(fig, plot_dim, nb_fct=nb_fct,fct=fct_list[j-1], fct_name=fct_names[j-1], idx_row=j, nb_column=nb_column)
        if plot_std:
            #std
            ax = fig.add_subplot(nb_column, nb_fct, nb_column + nb_column*(j-1))
            add_plot_std(d, ax, mc_list, nb_points_list,
                         color_list,
                         marker_list=marker_list,
                         fct_name=fct_names[j-1],
                         estimators=estimators)
            if plot_error:
                #error
                ax = fig.add_subplot(nb_column, nb_fct, nb_column+ nb_column*(j-1))
                add_plot_error(d, ax, mc_list, estimators,
                               nb_points_list,
                               error_type,
                               color_list,
                               marker_list = marker_list,
                               log_scale=log_scale, fct_name=fct_names[j-1], nb_subsample=nb_subsample_nb_points)
        else:
            ax = fig.add_subplot( nb_column, nb_fct, nb_column+ nb_column*(j-1))
            add_plot_error(d, ax, mc_list, estimators,
                           nb_points_list,
                           error_type,
                           color_list,
                           marker_list = marker_list,
                           log_scale=log_scale,
                           fct_name=fct_names[j-1], nb_subsample=nb_subsample_nb_points)
        plt.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches='tight')
    plt.show()
    #return ax

def add_plot_functions(fig, plot_dim, nb_fct, fct,  idx_row, nb_column, fct_name=None):
    if plot_dim==2:
        x = np.linspace(-1,1, 100)
        X, Y = np.meshgrid(x, x)
        points = np.array([X.ravel(), Y.ravel()]).T
        z_f = fct(points)
        ax = fig.add_subplot(nb_fct, nb_column, 1+ nb_column*(idx_row-1), projection='3d')
        ax.scatter3D(X.ravel(), Y.ravel(), z_f, c=z_f)
    elif plot_dim==1:
        x = np.linspace(-1,1, 300)
        y_f = globals()["f_{}".format(idx_row)](np.atleast_2d(x).T)
        ax = fig.add_subplot(nb_fct, nb_column, 1+ nb_column*(idx_row-1))
        ax.plot(x, y_f)
    if fct_name is not None:
        ax.set_title(r"$%s$"%fct_name)

def add_plot_std(d, ax, mc_list, nb_points_list, color_list, marker_list, fct_name=None, estimators=None):
    log_nb_pts = np.log([nb_points_list]).T
    if estimators is None:
        estimators = mc_list.keys()
    i=0
    for t in estimators:
        std_f = mc_list[t]["mc_results_" + fct_name]["std_"+ t]
        reg_line, slope, std_reg = regression_line(nb_points_list, std_f)
        label_with_slope = t+": slope={0:.2f}".format(slope)+ ", std={0:.2f}".format(std_reg)
        ax.scatter(log_nb_pts, np.log(std_f),
                   c=color_list[i],
                   s=20,
                   alpha=0.5,
                   marker=marker_list[i],
                   label=label_with_slope)
        ax.plot(log_nb_pts, reg_line, c=color_list[i], alpha=1)
        #ax.plot(log_nb_pts, reg_line + 3*std_reg, c=col[i], linestyle=(0, (5, 10)))
        #ax.plot(log_nb_pts, reg_line - 3*std_reg, c=col[i], linestyle=(0, (5, 10)))
        i=i+1
    if fct_name is not None:
        ax.set_title( r"For $%s$"%fct_name + " (d=%s)" %d)
    ax.set_xlabel(r"$\log(N) $ ")
    ax.set_ylabel(r"$\log(\sigma)$")
    ax.legend()

def add_plot_error(d, ax, mc_list, estimators, nb_points_list, error_type, color_list, marker_list, log_scale, fct_name=None, nb_subsample=None):
    i=0
    for t in estimators:
        if error_type in ["SE", "Error"]:
            error_f = mc_list[t]["mc_results_"+fct_name]["error_"+ t]
            if error_type=="SE":
                error_f = [e**2 for e in error_f]
            if  nb_subsample is not None:
                idx_subsample = _subsample_nb_points(nb_points_list, nb_subsample=nb_subsample)
                nb_points_list = [nb_points_list[i] for i in idx_subsample]
                error_f = [error_f[i] for i in idx_subsample]
            x = np.array(nb_points_list) +25*i
            nb_list_expended = [[n]*len(e) for n,e in zip(nb_points_list, error_f)]
            #print("here in plot", np.array(nb_list_expended), error_f)
            ax.scatter(np.array(nb_list_expended) +25*i,
                        error_f,
                        c=color_list[i],
                        s=5,
                        marker=marker_list[i],
                        label=t)
            ax.plot(np.array(nb_list_expended),
                     [0]*len(nb_list_expended),
                      color="grey", linestyle="--")
            ax.boxplot(error_f, positions=x.tolist(),
                        widths = 20,
                        manage_ticks=False,
                        patch_artist=True, boxprops=dict(facecolor=color_list[i]),
                        whiskerprops=dict(color=color_list[i]),
                        showmeans=True,
                        meanprops=dict(marker='.', color='r', markeredgecolor='r'),
                        sym='',
                        )
            #ax.legend([a["boxes"][0]], [t], loc='lower left')
        i=i+1
    if log_scale:
        #ax.set_xscale("log")
        ax.set_yscale("log")
    if fct_name is not None:
        ax.set_title(r"For $%s$"%fct_name + " (d=%s)" %d)
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



def qq_plot_residual(mc_list, nb_points_list, fct_names, save_fig=None, **kwargs):
    estimators = mc_list.keys()
    fct_nb = len(fct_names)
    color_list = ["b", "k", "g", "m", "gray", "c","y", "darkred", "orange", "pink"]
    fig, ax = plt.subplots(1, fct_nb, figsize=( int(3*fct_nb), 4))
    for i in range(fct_nb):
        j=0
        for t in estimators:
            std_f = mc_list[t]["mc_results_"+ fct_names[i]]["std_"+ t]
            _, _, _, residual = regression_line(nb_points_list, std_f, residual=True,residual_normality_test=None, **kwargs)
            # pp = sm.ProbPlot(residual, fit=True)
            # qq = pp.qqplot(marker='.', markerfacecolor=color_list[j], markeredgecolor=color_list[j], alpha=0.3, label=t)
            # sm.qqline(qq.axes[0], line='45', fmt='r--', axis=ax[i], label=t)
            sm.qqplot(residual, line='s', markerfacecolor=color_list[j], markeredgecolor=color_list[j], marker='.', alpha=0.5, ax=ax[i],label=t)
            ax[i].get_lines()[1 + 2*j].set_color(color_list[j])
            ax[i].get_lines()[1 + 2*j].set_linewidth("2")
            #ax[j].lines.set_color(color_list[j])
            j+=1
    #plt.tight_layout()
        ax[i].legend()
        ax[i].set_title(r"For $%s$"%fct_names[i] )
    plt.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches='tight')
    plt.show()


def _subsample_nb_points(my_list, nb_subsample=10):
    first_index = 0
    last_index = len(my_list) - 1
    if nb_subsample == 5:
        midle_index = len(my_list)//2
        left_midle_index = (midle_index - first_index)//2
        right_midle_index = midle_index + left_midle_index
        selected_indexes = [first_index, left_midle_index, midle_index,right_midle_index, last_index]
    else:
        selected_indexes = [first_index,last_index]
        step_size = (len(my_list) - 1) // (nb_subsample-2)
        for i in range(1, nb_subsample-1):
            selected_indexes.append(first_index + i * step_size)
    return selected_indexes
