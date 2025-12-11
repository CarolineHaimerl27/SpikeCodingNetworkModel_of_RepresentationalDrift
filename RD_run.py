import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap, Normalize

# own functions
from RD_functions import *
from setup import *


def simulate_RD(name, Tdays, Nsim):
    # parameters
    sigma_thresh = 1
    sigma_v = 1e-2 
    tau_thresh = sigma_thresh/10

    # number of simulations 
    param = {'orien': np.linspace(0, 2*np.pi,16)[:-1], # orientation of drifting gratings
            'dt': 1e-6,
            'tau': 1 / 100,
            'z0': 900,
            'x0': 5.,
            'extra_dim': 1,
            'Ttime': 0.08/10, 
            'Tcut': 0.04/10, 
            'sigma_v': sigma_v,
            'dim_im': 10, # pixel size of image is dim_im**2
            'D0': 1,
            'sigma_w': [2],
            'lam_w': [5],
            'Npos': 12, 
            'mu': 0,
            'freq': 2,
            'repeat': 1}
    param['offset'] = np.linspace(0, 2*np.pi, int(param['Ttime'] / param['dt']))
    param['Tsteps'] = int(param['Ttime'] / param['dt'])
    time = np.arange(0, param['Ttime'], param['dt'])
    param['Scut'] = np.where(time>param['Tcut'])[0][0]
    param['weights'], param['weights0'], param['Thresh'], param['N'], _, _ = gabor_SCN_weights(param['sigma_w'], 
                                                                                               param['lam_w'], 
                                                                                               param['D0'], 
                                                                                               param['mu'], 
                                                                                               duplicate=False,
                                                                                            extra_dim=param['extra_dim'], 
                                                                                            dim_im=param['dim_im'], 
                                                                                            Npos = param['Npos'])

    param_stim = {'Npos': 1,
                'bounds': 0,
                'Norien': 1,
                'sig': 100, 
                'lam': 5, 
                'nsamp': param['dim_im']}
    param['Nsim'] = Nsim
    param = {**param, **param_stim}
    if param['repeat'] > 1:
            param['weights'] = np.repeat(param['weights'], 2, axis=0)
            param['Thresh'] = np.repeat(param['Thresh'], 2)
            param['N'] = param['N'] * 2
            param['weights0'] = np.repeat(param['weights0'], 2, axis=0)
    param['thresh_lower_bound'] = 0.3  # lower bound for threshold


    np.random.seed(3)
    param['sigma_thresh'] = sigma_thresh
    param['thresh_start'] = param['Thresh']+np.random.randn(param['N'])*param['sigma_thresh']
    param['thresh_start'][param['thresh_start'] < param['thresh_lower_bound']] = param['thresh_lower_bound']

    ######### DAY 0 simulation #########
    print('run ', name)

    RSUM = []
    for _ in range(param['Nsim']):
        # day 0 simulation:
        rsum, _, _, spikes = simulate([param['thresh_start']], 
                                            sigma_v = param['sigma_v'],
                                param=param, param_stim=param_stim)
        RSUM.append([rsum])

        spikes = np.concatenate(np.concatenate(spikes))

    #### SAVE ACTIVITY ####
    param['ACT'] = RSUM
    with open(path_save +name+'.pk', 'wb') as f:
                pickle.dump(param, f)


    ######### ALL THE FOLLOWING DAYS #########

    param['Tdays'] = Tdays
    param['tau_thresh'] = tau_thresh
    thresh = [param['thresh_start']]
    for _ in range(param['Tdays']-1):
        thresh.append(thresh[-1]+np.random.randn(param['N'])*param['tau_thresh'])
        thresh[-1][thresh[-1] < param['thresh_lower_bound']] = param['thresh_lower_bound']
    param['thresh'] = np.stack(thresh)

    # WITH INTERPOLATION
    def interpolate_vector(x, Z, kind='linear'):
        T = len(x)
        original_indices = np.arange(T)
        interpolated_indices = np.linspace(0, T - 1, T * Z)
        
        interpolator = interp1d(original_indices, x, kind=kind)
        x_interpolated = interpolator(interpolated_indices)
        
        return x_interpolated[:T]

    param['thresh'] = np.stack([interpolate_vector(param['thresh'][:,nn], 10) for nn in range(param['N'])], axis=1)

    # simulate 
    for cc in range(len(RSUM)):
        print('simulations ', cc)
        for day in range(0,len(thresh)):

            rsum, _, _, _ = simulate([thresh[day]], 
                                sigma_v = param['sigma_v'],
                                param=param, param_stim=param_stim)
            RSUM[cc].append(rsum)

        #### SAVE ACTIVITY ####
        param['ACT'] = RSUM
        with open(path_save +name+'.pk', 'wb') as f:
                    pickle.dump(param, f)

def tuning(name, ampl_same):
       # load simulation:
    with open(path_save +name+'.pk', 'rb') as f:
            sim_data = pickle.load(f)
    print('voltage noise: ', sim_data['sigma_v'], ' stepsize: ', sim_data['tau_thresh'], ' days: ', sim_data['Tdays'])
    
    # compute activation
    rsum = np.stack([sim_data['ACT'][ss] for ss in range(sim_data['Nsim'])])
    rsum = rsum[:,:,0,:,:]

    # compute tuning curves
    par = tuning_fitting(rsum, 
                    Ntrain=1 if sim_data['Nsim'] == 1 else int(sim_data['Nsim']/2),
                    Cfolds = 1 if sim_data['Nsim'] == 1 else 10, 
                    param=sim_data,
                    ampl_same=ampl_same,
                    firstlastonly=False,
                    usetest = False if sim_data['Nsim'] == 1 else True,
                    which_days=None)
    return par, sim_data, rsum
   
def visualize_RD(name, par, sim_data, rsum, savefig, ampl_same):

    ######### THRESHOLD #########
    thresh0 = np.mean(sim_data['Thresh'])
    plt.figure(figsize=(3,3))
    tdiffs = sim_data['thresh']-sim_data['thresh'][0]
    m, s = np.mean(tdiffs,axis=1), np.std(tdiffs, axis=1)
    m += thresh0
    minx = np.min(sim_data['thresh'], axis=1)
    plt.plot(np.arange(sim_data['Tdays']), m, '-', color='k')
    plt.fill_between(np.arange(sim_data['Tdays']), np.max(np.array([m-s, minx]),axis=0), m+s, alpha=0.4, color='k')
    plt.plot(np.arange(sim_data['Tdays'])+thresh0, minx, '--', color='k', label='minimum')
    plt.legend()
    # remove top and right spines
    ax = plt.gca()  # get current axes
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    if savefig: plt.savefig(path_graph+'thresh_change_'+name+'.pdf', bbox_inches='tight')


    ##################
    np.random.seed(0)

    ind_plot = np.arange(sim_data['Tdays'])

    diffs, p, ind = compute_diffs(par, ind_plot, ampl_same)

    plt.figure(figsize=(3,3))
    m = np.nanmedian(np.abs(diffs), axis=(1,2))
    plt.plot(ind_plot, m, '-', color='k', label='median')

    # sign. change neurons
    m = np.array([np.nanmedian(np.abs(diffs[day, :, ind[p[day,:]<0.05]])) for day in range(diffs.shape[0])])
    plt.plot(ind_plot, m, '-', color='r', label='median sign')


    # remove top and right spines
    ax = plt.gca()  # get current axes
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    plt.ylim(-1, 30)
    plt.xticks(ind_plot[::4])
    plt.yticks([0, 5, 10, 15, 20, 30]);
    plt.title('Median change in peak angle')
    plt.legend()
    print('Median change in peak angle: ', m[-1] , ' +/- ', s[-1])

    if savefig: plt.savefig(path_graph+'peak_angle_change'+name+'.pdf', bbox_inches='tight')


    ############################
    p0 = np.stack(par['phases'][0])
    s0 = np.stack(par['signs'][0])
    p0[s0==0] = np.nan

    diffs = []
    day1 = -1
    p1 = np.stack(par['phases'][day1])
    s1 = np.stack(par['signs'][day1])
    p1[s1==0] = np.nan
    diffs.append(np.concatenate([diff_ang_orientation(p0[cc], p1[cc], ampl_same)*180/np.pi for cc in range(len(p0))]))
    diffs = np.concatenate(diffs)
    plt.figure(figsize=(3,3))
    bins = [-180, -90, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 90, 180]
    plt.hist(diffs, bins, alpha=1, color='darkgrey', label='change in peak angle', density=True);
    plt.axvline(0, color='k', linestyle='-');
    plt.xticks([-180, -90, 0, 90, 180]);
    # remove top and right spines
    ax = plt.gca()  # get current axes
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    if savefig: plt.savefig(path_graph+'peak_angle_change_hist_'+name+'.pdf', bbox_inches='tight')


    #########################

    which_days = np.arange(sim_data['Tdays'])
    diffs, p, ind = compute_diffs(par, which_days, ampl_same=ampl_same)

    # Create a custom colormap
    bins = np.linspace(0, 1, 256 * 5)  
    colors = plt.cm.coolwarm(bins)
    custom_cmap = ListedColormap(colors)
    custom_cmap.set_bad(color='white')

    plt.figure(figsize=(5, 5));
    np.random.seed(0)
    cc = 0
    ind = np.random.choice(ind, 100, replace=False)  # Randomly select 100 neurons for visualization
    plt.pcolor(diffs[:,cc,ind].T, vmin = -90, vmax = 90, cmap=custom_cmap);
    plt.colorbar();
    # remove top and right spines
    ax = plt.gca()  # get current axes
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    if savefig: plt.savefig(path_graph+'change_in_peak_angle_heatmap_'+name+'.pdf', bbox_inches='tight')

    ###########################

    fig, axs = plt.subplots(1, 2, figsize=(4, 4))  # 1 row, 2 columns

    which_days = np.arange(sim_data['Tdays']) # np.array([0, 1, 10, 20])
    # Fade-out 
    ax = axs[0]
    sday0 = np.stack(par['signs'][0])
    for day in which_days[1:]:
        sday1 = np.stack(par['signs'][day])
        perc = (np.sum(((sday0==1)&(sday1==0)), axis=1) / np.sum((sday0==1), axis=1)) * 100
        m, s = np.mean(perc), np.std(perc)
        ax.errorbar(day, m, yerr=s, color='k', fmt='o', capsize=0, label='day '+str(day))
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticks(which_days)
    ax.set_title('fade-out')
    # remove top and right spines
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);

    # Fade-in 
    ax = axs[1]
    sday0 = np.stack(par['signs'][0])
    for day in which_days[1:]:
        sday1 = np.stack(par['signs'][day])
        perc = (np.sum(((sday0==0)&(sday1==1)), axis=1) / np.sum((sday0==0), axis=1)) * 100
        m, s = np.mean(perc), np.std(perc)
        ax.errorbar(day, m, yerr=s, color='k', fmt='o', capsize=0, label='day '+str(day))
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(which_days)
    ax.set_title('fade-in')
    # remove top and right spines
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);

    ######## SAME DAY ########
    # Fade-out
    ax = axs[0]
    sday0 = np.stack(par['signs'][0])[::2]
    sday1 = np.stack(par['signs'][0])[1::2]
    perc = (np.sum(((sday0==1)&(sday1==0)), axis=1) / np.sum((sday0==1), axis=1)) * 100
    m, s = np.mean(perc), np.std(perc)
    ax.errorbar(0, m, yerr=s, color='k', fmt='o', capsize=0, label='day '+str(0))

    # Fade-in
    ax = axs[1]
    sday0 = np.stack(par['signs'][0])
    sday0 = np.stack(par['signs'][0])[::2]
    sday1 = np.stack(par['signs'][0])[1::2]
    perc = (np.sum(((sday0==0)&(sday1==1)), axis=1) / np.sum((sday0==0), axis=1)) * 100
    m, s = np.mean(perc), np.std(perc)
    ax.errorbar(0, m, yerr=s, color='k', fmt='o', capsize=0, label='day '+str(0))

    fig.tight_layout()
    if savefig: fig.savefig(path_graph+'fade_in_out.pdf', bbox_inches='tight')

    #################################
    inddays = np.array(np.array([8, 16, 24, 32])/2, dtype = 'int') if sim_data['Tdays'] >= 32 else np.arange(sim_data['Tdays'])

    toplot = np.arange(20) if sim_data['Tdays'] >= 20 else np.arange(sim_data['Tdays'])

    # average population rate correlation over all orientations and simulations, one day-pair
    ccs = []
    ccse = []
    indpairs = np.random.choice(np.arange(rsum.shape[0]), rsum.shape[0], replace=False)
    ccs.append(np.mean([np.mean([np.corrcoef(rsum[sim, 0, oo,:], rsum[indpairs[sim], 0, oo,:])[0,1] for oo in range(rsum.shape[2])]) for sim in range(rsum.shape[0])]))
    ccse.append(np.mean([np.std([np.corrcoef(rsum[sim, 0, oo,:], rsum[indpairs[sim], 0, oo,:])[0,1] for oo in range(rsum.shape[2])])/ np.sqrt(rsum.shape[2]) for sim in range(rsum.shape[0])])/ np.sqrt(rsum.shape[0]))
    for day in range(1, rsum.shape[1]):
        ccs.append(np.mean([np.mean([np.corrcoef(rsum[sim, 0, oo,:], rsum[sim, day, oo,:])[0,1] for oo in range(rsum.shape[2])]) for sim in range(rsum.shape[0])]))
        ccse.append(np.mean([np.std([np.corrcoef(rsum[sim, 0, oo,:], rsum[sim, day, oo,:])[0,1] for oo in range(rsum.shape[2])])/ np.sqrt(rsum.shape[2]) for sim in range(rsum.shape[0])]))
    ccs = np.array(ccs)
    ccse = np.array(ccse)

    plt.figure(figsize=(3,3))
    plt.plot(toplot, ccs[toplot], '-', color='k')
    plt.plot(inddays, ccs[inddays], 'o', color='k')
    for ii in inddays:
        plt.plot([ii, ii], [ccs[ii]-ccse[ii], ccs[ii]+ccse[ii]], '-', color='k')
    plt.ylim(0.1, 0.8)
    plt.xticks(inddays)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    if savefig: plt.savefig(path_graph+'pop_rate_corr_'+name+'.pdf', bbox_inches='tight')



    ################# DECODING #################
    Nperc = 0.05
    Nsim = 100

    toplot = np.array([0, 20, 40, 60]) if sim_data['Tdays'] >= 100 else np.arange(sim_data['Tdays'])

    # Plotting if needed
    plt.figure(figsize=(5, 5))

    acc_mean, acc_all = decode_pairwise_category_same_day(rsum, Nperc = Nperc, Nsim=Nsim, random_seed=0)
    acc_se = np.std(acc_all, axis=(0,2)) / np.sqrt(acc_all.shape[0]*acc_all.shape[2])  # Standard error
    plt.plot(acc_mean, marker='', label='same day', color='purple');
    plt.fill_between(np.arange(len(acc_mean)), acc_mean - acc_se, acc_mean + acc_se, color='purple', alpha=0.2);
    plt.plot(toplot, acc_mean[toplot], marker='o', color='purple');
    for ii in range(len(toplot)):
        plt.plot([toplot[ii], toplot[ii]], [acc_mean[toplot[ii]]-acc_se[toplot[ii]], acc_mean[toplot[ii]]+acc_se[toplot[ii]]], '-', color='purple')


    acc_mean, acc_all = decode_pairwise_category_across_days(rsum, Nperc = Nperc, Nsim=Nsim, random_seed=0)
    acc_se = np.std(acc_all, axis=(0,2)) / np.sqrt(acc_all.shape[0]*acc_all.shape[2])  # Standard error
    plt.plot(acc_mean, marker='', label='day0', color='green');
    plt.fill_between(np.arange(len(acc_mean)), acc_mean - acc_se, acc_mean + acc_se, color='green', alpha=0.2);
    plt.plot(toplot, acc_mean[toplot], marker='o', color='green');
    for ii in range(len(toplot)):
        plt.plot([toplot[ii], toplot[ii]], [acc_mean[toplot[ii]]-acc_se[toplot[ii]], acc_mean[toplot[ii]]+acc_se[toplot[ii]]], '-', color='green')


    acc_mean, acc_all = decode_pairwise_category_general_decoder(rsum, Nperc=Nperc, Nsim=Nsim, random_seed=0, exclude_same_day= True)
    acc_se = np.std(acc_all, axis=(0,2)) / np.sqrt(acc_all.shape[0]*acc_all.shape[2])  # Standard error
    plt.plot(acc_mean, marker='', label='day0', color='grey');
    plt.fill_between(np.arange(len(acc_mean)), acc_mean - acc_se, acc_mean + acc_se, color='grey', alpha=0.2);
    plt.plot(toplot, acc_mean[toplot], marker='o', color='grey');
    for ii in range(len(toplot)):
        plt.plot([toplot[ii], toplot[ii]], [acc_mean[toplot[ii]]-acc_se[toplot[ii]], acc_mean[toplot[ii]]+acc_se[toplot[ii]]], '-', color='grey')


    plt.xlabel('Day')
    plt.ylabel('Mean decoding accuracy')
    plt.title('Category decoding over time with '+str(Nperc*100)+'% neurons')
    plt.yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]));
    ax = plt.gca()  
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    if savefig: plt.savefig(path_graph+'decoding_subsampled'+str(Nperc)+'_'+name+'.pdf', bbox_inches='tight')


if __name__ == "__main__":
        
        name = 'sim_00'
        
        Tdays = 100
        Nsim = 10

        simulate_RD(name, Tdays, Nsim)

        ampl_same=True

        par, sim_data, rsum = tuning(name, ampl_same)

        savefig = True

        visualize_RD(name=name, 
                     par = par, 
                     sim_data = sim_data, 
                     rsum = rsum, 
                     savefig = savefig, 
                     ampl_same=ampl_same)

