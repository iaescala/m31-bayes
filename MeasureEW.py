from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.special import wofz
import numpy as np
import sys

linelist_root = '/Users/iescala/Documents/projects/mahler/linelists/'

# Lorentzian parameters - amplitude (a), central wavelength (x0), half-width at half-maximum (gamma)
def lorentzian(x, x0, a, gamma):
    return a * gamma**2. / ( (x- x0)**2. + gamma**2. )

def gaussian(x, x0, a, sigma):
    return a * np.exp( -0.5 * (x - x0)**2. / sigma**2.)

def voigt(x, x0, a, gamma, sigma):
    return a * np.real(wofz(((x - x0) + 1j*gamma)/sigma/np.sqrt(2)))

def chisq(obs, model, weights, fittype=''):

    if fittype == 'lorentzian': nvarys = 3
    if fittype == 'gauss': nvarys = 3
    if fittype == 'voigt': nvarys = 4

    devs = (obs - model)**2. * weights
    stat = np.nansum(devs)
    stat /= (float(devs.size) - float(nvarys))
    return stat

def get_linelist_info(line_i):

    linefile_b = f'{linelist_root}valdnist.4163'
    linefile_r = f'{linelist_root}vald.allmol.6391'
    linefile_strong = f'{linelist_root}vald.strong'

    linelist_b = np.loadtxt(linefile_b, dtype={'names': ('wvl', 'species',\
     'ep', 'loggf'), 'formats': ('f4', 'f4', 'f4', 'f4')}, skiprows=1, usecols=(0,1,2,3))

    linelist_r =  np.loadtxt(linefile_r, dtype={'names': ('wvl', 'species',\
     'ep', 'loggf'), 'formats': ('f4', 'f4', 'f4', 'f4')}, skiprows=1, usecols=(0,1,2,3))

    linelist_strong = np.loadtxt(linefile_strong, dtype={'names': ('wvl', 'species',\
     'ep', 'loggf'), 'formats': ('f4', 'f4', 'f4', 'f4')}, skiprows=1, usecols=(0,1,2,3))

    if line_i == 'cat':

        #Beginning and end wavelengths for the calcium triplet
        linepass1 = [8490.5, 8534.5, 8654.5]
        linepass2 = [8505.5, 8549.5, 8669.5]

        wcat = np.where( (linelist_r['species'] == 20.0) & ( (np.round(linelist_r['wvl']) == 8498) |\
                                                (np.round(linelist_r['wvl']) == 8542) |\
                                                (np.round(linelist_r['wvl']) == 8662) ) )[0]
        wcat = wcat[[0,2,3]]

        nlines = 3
        linelist = linelist_r[wcat]

    if line_i == 'na':

        linepass1 = [8178, 8190]
        linepass2 = [8189, 8200]

        wna = np.where( (linelist_r['species'] == 11.0) & ( (np.round(linelist_r['wvl']) == 8183) |\
                                                            (np.round(linelist_r['wvl']) == 8195) ) )[0]
        wna = wna[[0,2]]

        nlines = 2
        linelist = linelist_r[wna]

    if line_i == 'mg':

        linepass1 = [8800]
        linepass2 = [8815]

        wmg = np.where( (np.round(linelist_strong['wvl']) == 8807) & (linelist_strong['species'] == 12.0)  )[0]

        nlines = 1
        linelist = linelist_strong[wmg]

    return nlines, linepass1, linepass2, linelist


def measure_ew(wvl, spec, ivar,
               line_i, nlines, linepass1, linepass2, linelist,
               fittype='gauss', halfwindow=50., halfwidth=1.5):

    residual = 1. - spec

    winf = np.where(~np.isfinite(residual))[0]
    if len(winf) > 0:
        residual[winf] = 0.

    residsm = convolve(residual, Box1DKernel(10))

    lambdas = linelist['wvl']
    species = linelist['species']

    if fittype == 'lorentzian': nvarys = 3
    if fittype == 'gauss': nvarys = 3
    if fittype == 'voigt': nvarys = 4

    params = np.full((nlines, nvarys), np.nan)

    ews = np.full(nlines, np.nan)
    ewerrs = np.full(nlines, np.nan)
    chis = np.full(nlines, np.nan)

    for k in range(nlines):

        wb = np.where( (wvl > (linelist['wvl'][k] - halfwindow)) &\
                       (wvl < linelist['wvl'][k]) )[0]

        wr = np.where( (wvl > linelist['wvl'][k]) &\
                       (wvl < (linelist['wvl'][k] + halfwindow)) )[0]

        if (len(wb) == 0) or (len(wr) == 0):
            sys.stderr.write(r'Line at '+str(linelist['wvl'][k])+' out of spectral range\n')
            continue

        nsig = 0.2
        medb = np.nanmedian( nsig * ivar[wb]**(-0.5) )
        medr = np.nanmedian( nsig * ivar[wr]**(-0.5) )

        if line_i == 'cat' or line_i == 'na':

            ww1 = np.where(residsm[wb] < medb )[0]
            ww2 = np.where(residsm[wr] < medr )[0]

        else:

            ww1 = np.where( ( residsm[wb] < medb ) | ( wvl[wb] < (linelist['wvl'][k] - halfwidth) ) )[0]
            ww2 = np.where( ( residsm[wr] < medr ) | ( wvl[wr] > (linelist['wvl'][k] + halfwidth) ) )[0]

        if (len(ww1) == 0) or (len(ww2) == 0):
            sys.stderr.write(r'Line at '+str(linelist['wvl'][k])+' too weak to measure\n')
            continue

        if line_i in ['cat', 'na', 'mg']:

            lambdamin = linepass1[k]
            lambdamax = linepass2[k]

        else:

            wmin = wb[np.nanmax(ww1)]
            wmax = wr[np.nanmin(ww2)]

            lambdamin = wvl[wmin]
            lambdamax = wvl[wmax]

        win = np.where( (wvl >= lambdamin) & (wvl <= lambdamax ) )[0]

        if len(win) < 5:
            sys.stderr.write('Insufficient number of points to measure for line '+str(linelist['wvl'][k])+'\n')
            continue

        ivartemp = ivar[win]
        winvalid = np.where( ivartemp == 0. )[0]
        if len(winvalid) > 0:
            sys.stderr.write('Invalid inverse variance array in range for line '+str(linelist['wvl'][k])+'\n')
            continue

        lambdatemp = wvl[win]
        spectemp = spec[win]

        specmin = np.nanmin(spectemp)
        specmax = np.nanmax(spectemp)

        lambdamid = np.nanmedian(lambdatemp)
        lambdatemp -= lambdamid

        lambda0 = linelist['wvl'][k] - lambdamid

        inv_specmin = 1. - specmin
        if inv_specmin > 1.: inv_specmin = 1.
        a0 = inv_specmin

        sigma_ewerr = (ivartemp/2.)**(-0.5)
        sigma_ew = (ivartemp)**(-0.5)

        if fittype == 'lorentzian':

            gamma0 = 0.04 * ( np.nanmax(lambdatemp) - np.nanmin(lambdatemp) )
            p0 = [lambda0, a0, gamma0]

            try: p,_ = curve_fit(lorentzian, lambdatemp, 1.-spectemp, p0=p0, sigma=sigma_ew, absolute_sigma=True)

            except RuntimeError:
                sys.stderr.write('Maximum number of iterations exceeded: measurement failed for line '+\
                             str(linelist['wvl'][k])+'\n')
                continue

            ew = 1.e3 * p[1] * (np.pi * p[2]) #units of mA, a0 = A/(pi gamma)

            model = lorentzian(lambdatemp, p[0], p[1], p[2])

        if fittype == 'gauss':

            sigma0 = 0.2 * ( np.nanmax(lambdatemp) - np.nanmin(lambdatemp) )
            p0 = [lambda0, a0, sigma0]

            try: p,_ = curve_fit(gaussian, lambdatemp, 1.-spectemp, p0=p0, sigma=sigma_ew, absolute_sigma=True)

            except RuntimeError:
                sys.stderr.write('Maximum number of iterations exceeded: measurement failed for line '+\
                             str(linelist['wvl'][k])+'\n')
                continue

            ew = 1.e3 * p[1] * (p[2] * np.sqrt( 2. * np.pi )) #a0 = A/sqrt(2 pi sigma^2)

            model = gaussian(lambdatemp, p[0], p[1], p[2])

        if fittype == 'voigt':

            gamma0 = 0.04 * ( np.nanmax(lambdatemp) - np.nanmin(lambdatemp) )
            sigma0 = 0.2 * ( np.nanmax(lambdatemp) - np.nanmin(lambdatemp) )
            #alpha0 = sigma0 * np.sqrt( 2 * np.log(2) )

            p0 = [lambda0, a0, gamma0, sigma0]

            try: p,_ = curve_fit(voigt, lambdatemp, 1.-spectemp, p0=p0, sigma=sigma_ew, absolute_sigma=True)

            except RuntimeError:
                sys.stderr.write('Maximum number of iterations exceeded: measurement failed for line '+\
                             str(linelist['wvl'][k])+'\n')
                continue

            ew = 1.e3 * p[1] * (np.sqrt(2*np.pi) * p[3] ) #a0 = A/sqrt(2 pi sigma^2)

            model = voigt(lambdatemp, p[0], p[1], p[2], p[3])

        if fittype == 'numerical':

            #wm = np.abs(lambdatemp) < 3.
            wm = np.full(len(lambdatemp), True)
            ew = 1.e3 * simps(1.-spectemp[wm], x=lambdatemp[wm]) #in units of mA

            lambdatemp += lambdamid

            lambdas.append(linelist['wvl'][k])
            species.append(linelist['species'][k])

            ews[k] = ew

        if fittype != 'numerical':

            chi = chisq(1.-spectemp, model, ivartemp, fittype=fittype)

            n = 0
            ntrials = 100
            ewerr = np.zeros(ntrials)

            while n < ntrials:

                spectemperr = spectemp + (ivartemp**(-0.5)*np.random.normal(size=len(win)))
                specminerr = np.nanmin(spectemperr)

                inv_specminerr = 1.0 - specminerr
                if inv_specminerr > 1.: inv_specminerr = 1.
                a0 = inv_specminerr

                if fittype == 'lorentzian':

                    p0 = [lambda0, a0, gamma0]

                    try: p,_ = curve_fit(lorentzian, lambdatemp, 1.-spectemperr, p0=p0, sigma=sigma_ewerr,
                                 absolute_sigma=True)

                    except RuntimeError:
                        ewerr[n] = np.nan
                        continue

                    ewerr[n] = 1.e3 * p[1] * (np.pi * p[2]) #units of mA

                if fittype == 'gauss':

                    p0 = [lambda0, a0, sigma0]

                    try: p,_ = curve_fit(gaussian, lambdatemp, 1.-spectemperr, p0=p0, sigma=sigma_ewerr,
                                     absolute_sigma=True)

                    except RuntimeError:
                        ewerr[n] = np.nan
                        continue

                    ewerr[n] = 1.e3 * p[1] * (p[2] * np.sqrt( 2 * np.pi ))

                if fittype == 'voigt':

                    p0 = [lambda0, a0, gamm0, sigma0]

                    try: p,_ = curve_fit(voigt, lambdatemp, 1.-spectemperr, p0=p0, sigma=sigma_ewerr,
                                     absolute_sigma=True)

                    except RuntimeError:
                        ewerr[n] = np.nan
                        continue

                    ewerr[n] = 1.e3 * p[1] * (np.sqrt(2*np.pi) * p[3] )

                n += 1

            lambdatemp += lambdamid
            p[0] += lambdamid

            q1, q2, q3 = np.nanpercentile(ewerr, [16, 50, 84])
            ewerr_u = q3-q2
            ewerr_l = q2-q1

            #ewerr = np.nanstd(ewerr)
            ewerr = (ewerr_u + ewerr_l)/2.
            ew = q2

            params[k] = p
            ews[k] = ew
            ewerrs[k] = ewerr
            chis[k] = chi

    if fit_type != 'numerical':
        return lambdas, species, params, ews, ewerrs, chis
    else:
        return lambdas, species, ews


def call_measure_ew(wvl, spec, ivar, line_i, fittype='gauss'):

    nlines, linepass1, linepass2, linelist = get_linelist_info(line_i)

    ew_result = measure_ew(wvl, spec, ivar,
                           line_i, nlines, linepass1, linepass2, linelist,
                           fittype=fittype)

    return ew_result
