import numpy as np
from pandas import DataFrame
import scipy.optimize
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import combinations

########### VOTLAGE ###########


def spikes(v, T, omega):
    vminT = v - T
    s = np.zeros(len(v))
    if np.sum(vminT >= 0) > 0:
        s_ind = np.argmax(vminT)
        s[s_ind] = 1
    return s



def simulate(thresh, sigma_v,
             param, param_stim):
    rsum = []
    ssum = []
    xhat = []
    spikes = []
    for day in range(len(thresh)): 
        rsum.append([])
        ssum.append([])
        xhat.append([])
        spikes.append([])
        for ii in range(len(param['orien'])):
            gab_stim = []
            for oo in range(len(param['offset'])):
                ### SIGNAL ###
                gab_stim_oo, _ = gabor(nsamp=param_stim['nsamp'], 
                                    Gpar=param_stim, 
                                    offset=param['offset'][oo], 
                                    orien = np.array([param['orien'][ii]]))
                gab_stim.append(gab_stim_oo)

            gab_stim = np.concatenate(gab_stim)

            signal_gab = (gab_stim-np.min(gab_stim))/(np.max(gab_stim)-np.min(gab_stim))*2*param['x0']-param['x0']
            # add an extra dimension
            if param['extra_dim']:
                signal_gab = add_dim(signal_gab, z0=param['z0'])
            v, s, _, r = voltage(v0 = np.zeros(param['N']),
                                x = signal_gab,
                                T = thresh[day],
                                weights = param['weights'],
                                dt = param['dt'], 
                                tau= param['tau'],
                                sigma_v = sigma_v, #param['sigma_v'],
                                init_r = True,
                                Scut=param['Scut'],
                                timwind=param['Ttime']-param['Tcut'],
                                printme = False)

            rsum[-1].append(np.mean(r[param['Scut']:,:],axis=0))
            ssum[-1].append(np.sum(s[param['Scut']:,:],axis=0))
            xhat[-1].append(readout(param['weights'], r, param['weights0']))
            spikes[-1].append(s)

    return rsum, ssum, xhat, spikes


def add_dim(x, z0=1000):
    # x is assumed to be time by features
    return np.concatenate((x, np.ones([x.shape[0], 1]) * z0), axis=1)


def readout(weights, r,
            weights0 = None,
            T = None, mu=0):
    # bias correction
    bias = (np.linalg.norm(weights.T.dot(r.T)) +
            0.5*np.sum(-np.diag(weights.dot(weights.T)) + np.diag(weights0.dot(weights0.T)))) / \
           np.linalg.norm(weights.T.dot(r.T))
    return (weights.T.dot(r.T)).T * bias


def voltage(v0,
            x,
            T,
            weights,
            dt,
            tau,
            sigma_v,
            init_r = True,
            Scut = 0,
            timwind = 1,
            omega = None,
            printme = False):

    leak = 1 / tau
    V = np.zeros((len(T), x.shape[0]))  # voltages
    V[:, 0] = v0

    s = [np.zeros(len(T))]

    # initialize r
    if init_r:
        u_, s_, v_ = np.linalg.svd(weights)
        r = [x[0, :].dot(v_.T.dot(np.eye(len(s_)) * 1 / s_).dot(u_[:, :len(s_)].T))]
    else:
        r = [np.zeros(len(T))]
    ssum = []
    if omega is None: omega = (weights.dot(weights.T))
    for tt in range(len(x) - 1):
        ### voltage ###
        c_x = x[tt, :] / tau + (x[tt + 1, :] - x[tt, :]) / dt 

        # update membrane potential
        V[:, tt + 1] = V[:, tt] + dt * (-leak * V[:, tt] + weights @ c_x)  #-omega @ s[-1]
        if np.sum(s[-1])>0:
            V[:, tt + 1] -= omega[:, s[-1]==1].ravel()
        if np.sum(s[-1])>1:
            print('ERROR: more than one spike at a time')
        if sigma_v>0:
            V[:, tt + 1] = V[:, tt + 1] + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(len(T))  # voltage noise

        ### spikes ###
        s_tt = spikes(V[:, tt + 1], T, omega)
        ssum.append(np.sum(s_tt))

        # save
        r.append(r[-1] - r[-1] * dt / tau + s[-1])
        s.append(s_tt)
    s = np.stack(s)
    if printme:
        print(np.round(100 * np.mean(np.sum(s[Scut:,:], axis=1)),2), '% of time steps have spikes, and ',
              np.round(100 * np.mean(np.sum(s[Scut:,:], axis=0) > 0),2), '% of neurons are active at some points',
              '\n and have a population rate (sum of all spikes) ', np.round(np.sum(s[Scut:,:]),3),
              ' (rate per neuron=', np.round(np.mean(np.sum(s[Scut:,:], axis=0))/timwind,3), ')',
              'max rate=',np.round(np.max(np.sum(s[Scut:,:], axis=0))/timwind,3) )
    return V.T, s, ssum, np.stack(r)

########### WEIGHTS ###########

def gabor_SCN_weights(sigma, lam, D0, mu, extra_dim, duplicate, dim_im, Npos):
    ### CHARACTERISTICS ###
    prop_pop = []
    for sigma_ss in sigma:
        for lam_ll in lam:
            Gpar = {'Npos': Npos,
                    'bounds': 0,
                    'Norien': 9+duplicate*9,
                    'sig': sigma_ss,  # .7,
                    'lam': lam_ll,  # 1.5,
                    'nsamp': dim_im}

            _, prop = gabor(nsamp=Gpar['nsamp'], Gpar=Gpar)
            prop_pop.append(np.concatenate((prop,
                                            np.repeat(np.array([sigma_ss, lam_ll]), prop.shape[0]).reshape(2,
                                                                                                           prop.shape[
                                                                                                               0]).T,
                                            np.ones([prop.shape[0], 1])),
                                           axis=1))
            prop_pop.append(np.concatenate((prop,
                                            np.repeat(np.array([sigma_ss, lam_ll]), prop.shape[0]).reshape(2,
                                                                                                           prop.shape[
                                                                                                               0]).T,
                                            np.ones([prop.shape[0], 1]) * -1),
                                           axis=1))
    prop_pop = np.concatenate(prop_pop, axis=0)
    ### WEIGHTS ###
    weights = []
    for sigma_ss in sigma:
        for lam_ll in lam:
            Gpar = {'Npos': Npos,
                    'bounds': 0,
                    'Norien': 9+duplicate*9,
                    'sig': sigma_ss,  # .7,
                    'lam': lam_ll,  # 1.5,
                    'nsamp': dim_im}

            gab, prop = gabor(nsamp=Gpar['nsamp'], Gpar=Gpar)
            weights.append(np.concatenate((np.stack(gab).T, -np.stack(gab).T), axis=1))

    weights = np.concatenate(weights, axis=1)

    ### WEIGHTS ###
    N = weights.shape[1]
    if extra_dim:
        # normalize
        weights = (weights / np.sqrt(np.sum(weights ** 2, axis=0)))
        # add an extra dimension (cone effect)
        weights = np.concatenate((weights, np.ones([1, N])*np.max(weights)),axis=0).T
        print(N, ' neurons distributed across ', Gpar['Npos'], '^2 positions and ', Gpar['Norien'], '*2 orientations, cone-dim with weights=1 ')
        weights0 = weights.copy()
    else:
        # normalize
        weights0 = (weights / np.sqrt(np.sum(weights ** 2, axis=0))).T
        # scale weights
        weights = weights0 * D0
    # compute threshold on the unscaled weights
    T = np.sqrt(np.sum(weights0 ** 2, axis=1) + mu) / 2

    # properties of population
    prop_pop = DataFrame(prop_pop, columns=['x', 'y', 'theta', 'sigma', 'lambda', 'sign'])
    # position from left up to right up to left down to right down
    pos = prop_pop.loc[:, ['x', 'y']].dot(np.array([1, 100]))
    upos = np.unique(pos)
    pos = np.concatenate([np.where(pos[ii] == upos)[0] for ii in range(len(pos))])
    prop_pop['pos'] = pos
    prop_pop['key'] = np.arange(N)
    return weights, weights0, T, N, prop_pop, Gpar

def gabor(nsamp, Gpar, offset=0, orien = None):
    # define pixel space
    t = np.linspace(-2*np.pi, 2*np.pi, nsamp)
    x, y = np.meshgrid(t, t)
    # along each spatial dimension there are Gpar['Npos'] positions
    # define position grid
    pos = np.linspace(-2*np.pi, 2*np.pi, Gpar['Npos'])[Gpar['bounds']:(Gpar['Npos']-Gpar['bounds'])]
    posx, posy = np.meshgrid(pos,pos)
    posx, posy = posx.ravel(), posy.ravel()
    if orien is None:
        orien = np.arange(0, np.pi,np.pi/Gpar['Norien'])
    gab, prop = [], []
    for pp in range(len(posx)):
        mu = [posx[pp], posy[pp]]
        for oo in range(Gpar['Norien']):
            theta = orien[oo]
            # orientation
            xp = (x.ravel()-mu[0])*np.cos(theta)+(y.ravel()-mu[1])*np.sin(theta)
            yp = -(x.ravel()-mu[0])*np.sin(theta)+(y.ravel()-mu[1])*np.cos(theta)
            # gaussian times cosine
            gab.append(np.exp(-(xp**2+yp**2)/(2*Gpar['sig']**2))*np.cos(2*np.pi*xp/Gpar['lam']+np.pi/2+offset))
            gab[-1][np.abs(gab[-1]) < .1] = 0
            prop.append([mu[0], mu[1], theta])
    return gab, np.stack(prop)


####### TUNING ##########


def tuning_fitting(act, Ntrain, param, 
                   ampl_same, firstlastonly=False, usetest=True,
                   which_days = None, Cfolds = 1):
    peaks, phases, signs, freqs, ampls, kappas, mus = [], [], [], [], [], [], []
    if Ntrain is None:
        Ntrain = int(param['Nsim']/2)
    if isinstance(Ntrain, int) == False:
        print('WARNING: Ntrain is not an integer!!!')
    for day in range(act.shape[1]):
        if which_days is not None:
            if (day not in which_days) & (day!=0):
                peaks.append(np.nan)
                signs.append(np.nan)
                freqs.append(np.nan)
                phases.append(np.nan)
                ampls.append(np.nan)
                kappas.append(np.nan)
                mus.append(np.nan)
                continue
        if (day!=0) & (day!=(act.shape[1]-1)) & firstlastonly:
            peaks.append(np.nan)
            signs.append(np.nan)
            freqs.append(np.nan)
            phases.append(np.nan)
            ampls.append(np.nan)
            kappas.append(np.nan)
            mus.append(np.nan)
            continue
        act_day = np.stack([normalize_x(act[ii,day])[0] for ii in range(len(act))]) # orientations x neurons

        peaks.append([])
        signs.append([])
        freqs.append([])
        phases.append([])
        ampls.append([])
        kappas.append([])
        mus.append([])
        for cc in range(Cfolds):
            train_ind = np.random.choice(act_day.shape[0], size=Ntrain, replace=False)
            if Ntrain==1:
                act_train = act_day[train_ind]
                act_train = act_day[0]
            else:
                act_train = np.mean(act_day[train_ind],axis=0)
            if usetest:
                test_ind = np.ones(len(act_day),dtype='bool')
                test_ind[train_ind] = False
                test_ind = np.where(test_ind)[0]
            else:
                test_ind = []
            act_test = act_day[test_ind] if len(test_ind)>0 else None
            act_test = np.mean(act_test,axis=0) if act_test is not None else None

            peak = np.zeros(param['N'])*np.nan
            sign = np.zeros(param['N'],dtype='bool')
            freq = np.zeros(param['N'])*np.nan
            phase = np.zeros(param['N'])*np.nan
            ampl = np.zeros([param['N'],2])*np.nan
            kappa = np.zeros(param['N'])*np.nan
            mu = np.zeros(param['N'])*np.nan
            
            if act_test is not None:
                ind = np.where((np.mean(act_train==0,axis=0)<1)&\
                            (np.mean(act_test==0,axis=0)<1))[0]
            else:
                ind = np.where(np.mean(act_train==0,axis=0)<1)[0]
            for ii in ind:
                if ampl_same:
                    res = fit_sin(param['orien'], yy=act_train[:,ii], f=2, 
                                yytest = act_test[:,ii] if act_test is not None else None)
                # if neuron is not orientation tuned, check if it is direciton tuned before disgarding it as not tuned
                else:
                    res = fit_sin(param['orien'], yy=act_train[:,ii], f=1, 
                            yytest = act_test[:,ii] if act_test is not None else None)
                peak[ii] = res['peak']
                sign[ii] = res['sign']
                freq[ii] = res['freq']
                phase[ii] = res['phase']

            peaks[-1].append(np.array(peak))
            signs[-1].append(np.array(sign))
            freqs[-1].append(np.array(freq))
            phases[-1].append(np.array(phase))
            ampls[-1].append(np.array(ampl))
            kappas[-1].append(np.array(kappa))
            mus[-1].append(np.array(mu))
    
    return {'peaks': peaks, 'signs': signs, 'freqs': freqs, 
            'phases': phases, 'ampls': ampls, 'kappas': kappas, 'mus': mus}

def normalize_x(x, minx=None, maxx=None):
    
    if minx is None or maxx is None:
        minx = np.min(x,axis=0)
        maxx = np.max(x,axis=0)
    norm = (x - minx) / (maxx - minx)
    norm[:,minx==maxx] = 0  # avoid division by zero
    return norm, minx, maxx


def fit_sin(time, yy, yytest=None, f=None):
    # make sure yy is normalized
    if (np.min(yy)!=0)|(np.max(yy)!=1):
        yy = (yy-np.min(yy))/(np.max(yy)-np.min(yy))
    '''Fit sin to the input time sequence, and return fitting parameters 
    "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(time)
    yy = np.array(yy)

    guess = np.array([tt[np.argmax(yy)]]) #, 1])#, 0.])
    if f==2:
        guess = guess-np.pi if (guess-np.pi)>0 else (guess+np.pi)
    if np.isnan(guess):
        print('WARNING: guess is NaN, returning NaN')
    def wrap_sin(t, p): #, A):
        return sinfunc(t, p, off=0, A=np.max(yy), f=f)
    try:
        popt, _ = scipy.optimize.curve_fit(wrap_sin, tt, yy, p0=guess) #, full_output=True)
        p, off, A = popt[0], 0, np.max(yy) #popt[1] # popt[1]
    except RuntimeError:
        p, off, A = np.nan, np.nan, np.nan #, np.nan
    # want p to be encoded positively
    if p<0: p += 2*np.pi 
    # p cannot be larger than 360
    if p>= (2*np.pi): p = p -2*np.pi

    # test significance
    if np.isnan(p):
        sign = False
        t_peak = np.nan
    else:
        # is the fit better than a mean fit
        if yytest is not None:
            sign = (np.nanmean((yytest-wrap_sin(tt, p))**2)-np.nanmean((yytest-np.nanmean(yy))**2))<0
        else:
            sign = (np.nanmean((yy-wrap_sin(tt, p))**2)-np.nanmean((yy-np.nanmean(yy))**2))<0
        # compute peak
        t_peak = (-p / f) % (2 * np.pi)
        if f == 2 and t_peak > np.pi:
            t_peak -= np.pi
    return {"phase": p, "offset": off,  "freq": f, 'ampl': A, 'sign': sign, 'peak': t_peak}

def sinfunc(t, p, off, A, f):
    out = np.sin(f*t + p)*A**2+off
    out[out<0] = 0
    return out



def compute_diffs(par_ori, which_days, ampl_same):
    np.random.seed(0)

    p0 = np.stack(par_ori['phases'][0])
    s0 = np.stack(par_ori['signs'][0])
    p0[s0==0] = np.nan
    diffs = []
    for day in which_days:
        p1 = np.stack(par_ori['phases'][day])
        s1 = np.stack(par_ori['signs'][day])
        p1[s1==0] = np.nan
        if day==0:
            ccother = np.random.choice(len(p0), len(p0), replace=False) #np.concatenate((np.arange(int(len(p0)/2), len(p0)), np.arange(0, int(len(p0)/2)))) # use a different one
            diffs.append((np.stack([diff_ang_orientation(p0[cc], p1[ccother[cc]], ampl_same)*180/np.pi for cc in range(int(len(p0)))])))
        else:
            diffs.append((np.stack([diff_ang_orientation(p0[cc], p1[cc], ampl_same)*180/np.pi for cc in range(len(p0))])))

    diffs = np.stack(diffs)

    # only inlcude neurons which are significantly tuned on first day in all simulations
    ind = np.where(np.sum(s0==False,axis=0)==0)[0] #  # <(sim_data['Nsim']*0.5)
    pval = []
    for day in range(diffs.shape[0]):
        pval.append([])
        for nn in range(len(ind)):
            tmp = diffs[day, :, ind[nn]]
            if np.sum(np.isnan(tmp)==False)<5:
                pval[-1].append(np.nan)
            else:
                if np.nansum(np.abs(tmp))==0:
                    pval[-1].append(2)            
                else:
                    pval[-1].append(stats.ttest_1samp(tmp[np.isnan(tmp)==False], popmean=0).pvalue)
        pval[-1] = np.array(pval[-1])
    pval = np.stack(pval)
    return diffs, pval, ind

def diff_ang_orientation(x_ang_old, x_ang_new, ampl_same):
    # ampl_same = True --> orientation tuning
    # ampl_same = False --> direction tuning
    if ampl_same:
        x_old = np.mod(x_ang_old, np.pi)
        x_new = np.mod(x_ang_new, np.pi)
        # Wrap into [-π/2, π/2]
        diff = (x_ang_new - x_ang_old + np.pi) % (2 * np.pi) - np.pi
        diff = (x_new - x_old + np.pi/2) % np.pi - np.pi/2
        return diff
    else:
        diff = (x_ang_new - x_ang_old + np.pi) % (2 * np.pi) - np.pi
        return diff
    

######### DECODING #########


def decode_pairwise_category_same_day(data, Nperc=1.0, Nsim = 10, perctrain = 0.7, random_seed=0):
    """
    data shape: (n_sim, n_days, n_cat, n_neurons)
    Decodes category using pairwise comparisons across simulations (sim = trial).
    """
    np.random.seed(random_seed)
    _, n_days, n_cat, n_neurons = data.shape
    n_subsample = int(Nperc * n_neurons)
        
    category_pairs = list(combinations(range(n_cat), 2))
    acc_all = np.zeros((len(category_pairs), n_days, Nsim))  # acc per pair, per day

    for pair_idx, (cat1, cat2) in enumerate(category_pairs):
        for day in range(n_days):

            for nn in range(Nsim):
                # Get all simulations as trials for current day and categories
                X_cat1 = data[:, day, cat1, :]  # (n_sim, n_neurons)
                X_cat2 = data[:, day, cat2, :]  # (n_sim, n_neurons)

                # Subsample neurons
                selected_neurons = np.random.choice(n_neurons, n_subsample, replace=False)
                X_cat1 = X_cat1[:, selected_neurons]
                X_cat2 = X_cat2[:, selected_neurons]

                # Subsample training trials
                n_sim = int(perctrain * X_cat1.shape[0])
                train = np.random.choice(X_cat1.shape[0], int(perctrain * X_cat1.shape[0]), replace=False)
                X_cat1_train = X_cat1[train]
                X_cat2_train = X_cat2[train]
                # Ensure we have enough trials for both categories
                if X_cat1.shape[0] < 2 or X_cat2.shape[0] < 2:
                    continue

                # Build data and labels
                X = np.vstack([X_cat1_train, X_cat2_train])  # (2*n_sim, n_neurons)
                y = np.array([0] * n_sim + [1] * n_sim)

                # create testing data
                test = np.setdiff1d(np.arange(X_cat1.shape[0]), train)
                X_cat1_test = X_cat1[test]
                X_cat2_test = X_cat2[test]
                X_test = np.vstack([X_cat1_test, X_cat2_test])
                y_test = np.array([0] * len(X_cat1_test) + [1] * len(X_cat2_test))

                # Train-test split (e.g. 80-20), or LOOCV — here we'll use 5-fold CV
                clf = LogisticRegression(max_iter=1000, solver='lbfgs')
                clf.fit(X, y)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_all[pair_idx, day, nn] = acc

    acc_mean = acc_all.mean(axis=(0,2))
    return acc_mean, acc_all


def decode_pairwise_category_across_days(data, Nperc=1.0, Nsim = 10, random_seed=0):
    np.random.seed(random_seed)
    
    n_sim, n_days, n_cat, n_neurons = data.shape
    n_subsample = int(Nperc * n_neurons)

    category_pairs = list(combinations(range(n_cat), 2))
    acc_all = np.zeros((len(category_pairs), n_days, Nsim))  # accuracy per pair, per day

    for pair_idx, (cat1, cat2) in enumerate(category_pairs):
        # Build data for all simulations for the two categories
        X = data[:, :, [cat1, cat2], :]  # shape: (n_sim, n_days, 2, n_neurons)
        X = X.reshape(n_sim, n_days, 2, n_neurons)

        # Labels for the pair
        y_pair = np.array([0, 1])  # cat1 → 0, cat2 → 1

        for nn in range(Nsim):
            # Subsample neurons
            selected_neurons = np.random.choice(n_neurons, n_subsample, replace=False)

            # -------- Train on Day 0 across all simulations --------
            X_train = X[:, 0, :, :][:, :, selected_neurons]  # (n_sim, 2, n_subsample)
            X_train = X_train.reshape(-1, n_subsample)       # (n_sim * 2, n_subsample)
            y_train = np.tile(y_pair, n_sim)                 # (n_sim * 2,)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_train)

            # -------- Test on all days --------
            for day in range(n_days):
                X_test = X[:, day, :, :][:, :, selected_neurons]  # (n_sim, 2, n_subsample)
                X_test = X_test.reshape(-1, n_subsample)           # (n_sim * 2, n_subsample)
                y_test = np.tile(y_pair, n_sim)

                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_all[pair_idx, day, nn] = acc

    acc_mean = acc_all.mean(axis=(0,2))
    return acc_mean, acc_all


def decode_pairwise_category_general_decoder(data, Nperc=1.0, Nsim=10, random_seed=0, exclude_same_day = True):
    np.random.seed(random_seed)
    
    n_sim, n_days, n_cat, n_neurons = data.shape
    n_subsample = int(Nperc * n_neurons)

    category_pairs = list(combinations(range(n_cat), 2))
    acc_all = np.zeros((len(category_pairs), n_days, Nsim))  # accuracy per pair, per day

    for pair_idx, (cat1, cat2) in enumerate(category_pairs):
        ################# Extract relevant data #################
        X = data[:, :, [cat1, cat2], :]  # shape: (n_sim, n_days, 2, n_neurons)

        y_pair = np.array([0, 1])  # cat1 → 0, cat2 → 1

        for nn in range(Nsim):
            selected_neurons = np.random.choice(n_neurons, n_subsample, replace=False)

            ################# Build training pool from ALL days #################
            
            if exclude_same_day:
                # Test on all days
                for day in range(n_days):
                    X_all_days = X[:, :, :, selected_neurons]               # (n_sim, n_days, 2, n_subsample)
                    ind_without_day = np.ones(n_days, dtype=bool)
                    ind_without_day[day] = False
                    X_all_days = X_all_days[:,ind_without_day]
                    X_all_days = X_all_days.reshape(-1, n_subsample)        # (n_sim * n_days * 2, n_subsample)
                    y_all_days = np.tile(y_pair, n_sim * n_days)            # (n_sim * n_days * 2,)
                    if np.sum(np.diff(y_all_days)!=0)==0:
                        print('WARNING: no variation in labels, skipping')
                        continue

                    # Subsample the same number of training examples as one-day setup (n_sim * 2)
                    train_indices = np.random.choice(X_all_days.shape[0], size=n_sim * 2, replace=False)
                    while np.sum(np.diff(y_all_days[train_indices])!=0)==0:
                        train_indices = np.random.choice(X_all_days.shape[0], size=n_sim * 2, replace=False)
                    X_train = X_all_days[train_indices]
                    y_train = y_all_days[train_indices]

                    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
                    clf.fit(X_train, y_train)
                
                    X_test = X[:, day, :, :][:, :, selected_neurons]  # (n_sim, 2, n_subsample)
                    X_test = X_test.reshape(-1, n_subsample)           # (n_sim * 2, n_subsample)
                    y_test = np.tile(y_pair, n_sim)

                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    acc_all[pair_idx, day, nn] = acc

            else:
                X_all_days = X[:, :, :, selected_neurons]               # (n_sim, n_days, 2, n_subsample)
                X_all_days = X_all_days.reshape(-1, n_subsample)        # (n_sim * n_days * 2, n_subsample)
                y_all_days = np.tile(y_pair, n_sim * n_days)            # (n_sim * n_days * 2,)

                # Subsample the same number of training examples as one-day setup (n_sim * 2)
                train_indices = np.random.choice(X_all_days.shape[0], size=n_sim * 2, replace=False)
                X_train = X_all_days[train_indices]
                y_train = y_all_days[train_indices]

                clf = LogisticRegression(max_iter=1000, solver='lbfgs')
                clf.fit(X_train, y_train)

                ################# Test on all days #################
                for day in range(n_days):
                    X_test = X[:, day, :, :][:, :, selected_neurons]  # (n_sim, 2, n_subsample)
                    X_test = X_test.reshape(-1, n_subsample)           # (n_sim * 2, n_subsample)
                    y_test = np.tile(y_pair, n_sim)

                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    acc_all[pair_idx, day, nn] = acc

    acc_mean = acc_all.mean(axis=(0, 2))
    return acc_mean, acc_all

