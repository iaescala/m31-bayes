"""
Author: I. Escala (Caltech 2019-2020)
See Escala et al. 2020b for details.

Usage: Given spectra, use MeasureEW to first measure the Na I 8190 doublet
and Ca II triplet EWs. Note that spectra should be continuum normalized
before using with MeasureEW. This program will source those files. Similarly,
you should have your accompanying photometry in a given filter. You should
also have RVs measured.

File formats should be "slitmask_na_ews.npz" and "slitmask_cat_ews.npz"
containing columns "EW", "EWERR", and "OBJNAME", where each column
has size (N,nlines), N = number of stars on slitmask, nlines = number
of lines measured (in order of increasing wavelength), e.g. nlines = 2
for Na EW and nlines = 3 for CaT. NaNs expected where measurements
failed/are otherwise unavailable

Options to use different Ca Triplet metallicity calibrations, default
Rutledge et al. 1997a. Can also assume different ages for use with
the stellar isochrones, and option to include (rv_flag=True) or
exclude (rv_flag=False) RV as a diagnostic
"""

import numpy as np
import pickle
import os

import Utilities as Ut

def bayesian_prob_model(slitmask, objname, ra, dec, color, mag, vhelio, vrerr,
                        colorerr=None, magerr=None,
                        filter='vi', age=12., dm=24.47, ddm=None, calibration='rutledge',
                        rv_flag=True, exclude_blue=True, data_path='data',
                        root=os.getcwd()):

    membership_dict = {}

    na, na_err, na_flag = Ut.load_ew(slitmask, objname, data_path=data_path)

    cat, cat_err, cat_flag = Ut.load_cat(slitmask, objname, mag, magerr, dm=dm, ddm=ddm,
                                      calibration=calibration, data_path=data_path)

    xcmd, xcmd_err = Ut.calculate_xcmd(slitmask, color, mag, colorerr=colorerr,
                                    magerr=magerr, filter=filter, age=age, dm=dm,
                                    ddm=ddm, data_path=data_path, root=root)

    membership_dict['xcmd'] = xcmd
    membership_dict['e_xcmd'] = xcmd_err

    membership_dict['vhelio'] = vhelio
    membership_dict['e_vhelio'] = vrerr

    membership_dict['na'] = na
    membership_dict['e_na'] = na_err
    membership_dict['na_flag'] = na_flag

    membership_dict['fehspec'] = cat
    membership_dict['e_fehspec'] = cat_err
    membership_dict['cat_flag'] = cat_flag

    ########################################
    ### START BAYESIAN PROBABILITY MODEL ###
    ########################################

    #### Probability that a star observed on a DEIMOS slitmask is a M31 member ####

    def pdeimos(r):
        b = 45.518
        return np.exp(-r/b)

    rproj = Ut.calculate_rproj(ra, dec, dm=dm)

    #### Now define the conditional probabilites under the assumption of M31 and MW membership ####
    #### based on an ND Gaussian from the SPLASH data ####

    pm31 = pdeimos(rproj)
    membership_dict['log_prior'] = np.log10(pm31)

    log_pxm31 = gaussian_nd(membership_dict['vhelio'], membership_dict['na'], membership_dict['xcmd'],
                            membership_dict['fehspec'],
                            membership_dict['e_vhelio'], membership_dict['e_na'], membership_dict['e_xcmd'],
                            membership_dict['e_fehspec'],
                            na_flag, cat_flag, rv_flag=rv_flag, type='m31', root=root)

    log_pxmw = gaussian_nd(membership_dict['vhelio'], membership_dict['na'], membership_dict['xcmd'],
                           membership_dict['fehspec'],
                           membership_dict['e_vhelio'], membership_dict['e_na'], membership_dict['e_xcmd'],
                           membership_dict['e_fehspec'],
                           na_flag, cat_flag, rv_flag=rv_flag, type='mw', root=root)

    membership_dict['log_likelihood_m31'] = log_pxm31
    membership_dict['log_likelihood_mw'] = log_pxmw

    posterior_m31 = np.log10(pm31) + log_pxm31
    posterior_mw = np.log10(1.-pm31) + log_pxmw

    membership_dict['log_posterior_m31'] = posterior_m31
    membership_dict['log_posterior_mw'] = posterior_mw

    membership_dict['rproj'] = rproj

    membership_dict['objname'] = objname

    membership_dict['log_bayes'] = np.log10(10**posterior_m31/10**posterior_mw)

    #Cap the likelihood values
    membership_dict['log_bayes'][membership_dict['log_bayes'] > 5.] = 5.
    membership_dict['log_bayes'][membership_dict['log_bayes'] < -5.] = -5.

    #Assign stars with very negative values of X_CMD (which suffer from numerical effects) as MW
    if exclude_blue:
        wb = membership_dict['xcmd'] + np.nan_to_num(membership_dict['e_xcmd']) < 0.
        membership_dict['log_bayes'][wb] = -5.

    membership_dict['prob'] = 10**membership_dict['log_bayes']/(1. + 10**membership_dict['log_bayes'])

    return membership_dict


def gaussian_nd(xdata, ydata, zdata, wdata,
                dxdata, dydata, dzdata, dwdata,
                na_flag, cat_flag, rv_flag=True, type='m31', root=os.getcwd()):

    # x, y, z, w = vhelio, na, xcmd, cat

    with open(f'{root}/data/splash_distribution_final_{type}.pkl', 'rb') as f:
        splash_final = pickle.load(f)

    x = splash_final['x']; dx = splash_final['dx']
    y = splash_final['y']; dy = splash_final['dy']
    z = splash_final['z']; dz = splash_final['dz']
    w = splash_final['w']; dw = splash_final['dw']

    xdata = np.array(xdata); ydata = np.array(ydata); zdata = np.array(zdata); wdata = np.array(wdata)
    dxdata = np.array(dxdata); dydata = np.array(dydata); dzdata = np.array(dzdata); dwdata = np.array(dwdata)

    #If some of the errors are NaN, replace with the median error
    #of the no-NaN sample
    wnan_ddz = np.isnan(dzdata)
    if dzdata[~wnan_ddz].size > 2:
        dzdata[wnan_ddz] = np.nanmedian(dzdata[~wnan_ddz])
    else: #the case where there aren't any photometric errors
        dzdata = np.zeros(dzdata.size)

    wnan_ddy = np.isnan(dydata)
    dydata[wnan_ddy] = np.nanmedian(dydata[~wnan_ddy])

    wnan_ddw = np.isnan(dwdata)
    dwdata[wnan_ddw] = np.nanmedian(dwdata[~wnan_ddw])

    p = np.zeros(len(xdata))

    for i in range(len(xdata)):

        xerr = np.sqrt(dx**2 + dxdata[i]**2)
        yerr = np.sqrt(dy**2 + dydata[i]**2)
        zerr = np.sqrt(dz**2 + dzdata[i]**2)
        werr = np.sqrt(dw**2 + dwdata[i]**2)

        xdist = (xdata[i] - x)**2/xerr**2
        ydist = (ydata[i] - y)**2/yerr**2
        zdist = (zdata[i] - z)**2/zerr**2
        wdist = (wdata[i] - w)**2/werr**2

        norm = 1./np.sqrt(2 * np.pi)

        if rv_flag:
            if not na_flag[i] and not cat_flag[i]: #both Na and CaT are measured
                pi = norm**4 * np.exp(-0.5 * (xdist + ydist + zdist + wdist) )/(xerr * yerr * zerr * werr)
            if not na_flag[i] and cat_flag[i]: #Na is measured, but not CaT
                pi = norm**3 * np.exp(-0.5 * (xdist + ydist + zdist) ) / (xerr * yerr * zerr)
            if na_flag[i] and not cat_flag[i]: #Na is not measured, and CaT is
                pi = norm**3 * np.exp(-0.5 * (xdist + zdist + wdist) ) / (xerr * zerr * werr )
            if na_flag[i] and cat_flag[i]: #neither Na nor CaT are measured
                pi = norm**2 * np.exp(-0.5 * (xdist + zdist) )/ (xerr * zerr )

        else:

            if not na_flag[i] and not cat_flag[i]: #both Na and CaT are measured
                pi = norm**3 * np.exp(-0.5 * (ydist + zdist + wdist) ) / (yerr * zerr * werr)
            if not na_flag[i] and cat_flag[i]: #Na is measured, but not CaT
                pi = norm**2 * np.exp(-0.5 * (ydist + zdist) ) / (yerr * zerr)
            if na_flag[i] and not cat_flag[i]: #Na is not measured, but CaT is
                pi = norm**2 * np.exp(-0.5 * (zdist + wdist) ) / (zerr * werr)
            if na_flag[i] and cat_flag[i]: #neither Na nor CaT are measured
                pi = norm * np.exp(-0.5 * (zdist) ) / (zerr)

        log_pi_sum = np.log10( np.sum(pi) / float(len(pi)) )
        p[i] = log_pi_sum

    return p
