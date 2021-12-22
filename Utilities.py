"""
Note that right now only the Rutledge et al. 1997a calibration is implemented
here
"""

import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, SmoothBivariateSpline
import astropy.units as u
import os

def load_ew(slitmask, objname, data_path='data'):

    measure = np.load(f'{data_path}/{slitmask}_na_ews.npz')

    wmatch = np.array([np.where(measure['objname'] == obj)[0][0] for obj in objname])

    ewsum = np.sum(measure['ew'][wmatch], axis=1)/1.e3
    ewsum_err = np.sqrt(np.sum(measure['ewerr'][wmatch]**2., axis=1))/1.e3
    na_flag = np.isnan(ewsum) | np.isnan(ewsum_err)

    return ewsum, ewsum_err, na_flag

def load_cat(slitmask, objname, mag, magerr, dm=24.47, ddm=0., calibration='rutledge',
data_path='data'):

    calibrations = ['stark', 'ho', 'rutledge']
    assert calibration in calibrations

    measure = np.load(f'{data_path}/{slitmask}_cat_ews.npz')
    wmatch = np.array([np.where(measure['objname'] == obj)[0][0] for obj in objname])

    if not os.path.exists(f'{data_path}/{slitmask}_cat_{calibration}.npz'):
        pass
    else:
        cat = np.load(f'{data_path}/{slitmask}_cat_{calibration}.npz')
        return cat['cat'], cat['cat_err'], cat['cat_flag']

    if calibration == 'stark':

        cat_flag = np.isnan(measure['ew'][wmatch][:,1]) | np.isnan(measure['ew'][wmatch][:,2]) |\
                   np.isnan(measure['ewerr'][wmatch][:,1]) | np.isnan(measure['ewerr'][wmatch][:,2])

        cat, cat_err = cat_stark(slitmask, measure['ew'][wmatch][:,1]/1.e3,
                                 measure['ew'][wmatch][:,2]/1.e3,
                                 dm=dm, ddm=ddm, dew1=measure['ewerr'][wmatch][:,1]/1.e3,
                                 dew2=measure['ew'][wmatch][:,2]/1.e3)
    if calibration == 'ho':

        cat_flag = np.isnan(measure['ew'][wmatch][:,1]) | np.isnan(measure['ew'][wmatch][:,2]) |\
                   np.isnan(measure['ewerr'][wmatch][:,1]) | np.isnan(measure['ewerr'][wmatch][:,2])

        cat, cat_err = cat_ho(slitmask, measure['ew'][wmatch][:,1]/1.e3,
                              measure['ew'][wmatch][:,2]/1.e3,
                              dm=dm, ddm=ddm, dew1=measure['ewerr'][wmatch][:,1]/1.e3,
                              dew2=measure['ew'][wmatch][:,2]/1.e3)

    if calibration == 'rutledge':

        cat_flag = np.isnan(np.sum(measure['ew'][wmatch], axis=1)) |\
                   np.isnan(np.sum(measure['ewerr'][wmatch], axis=1))

        cat, cat_err, cat_flag = cat_rutledge(slitmask, measure['objname'][wmatch], mag, magerr,
                                 measure['ew'][wmatch][:,1]/1.e3, measure['ew'][wmatch][:,2]/1.e3,
                                 measure['ew'][wmatch][:,0]/1.e3,
                                 cat_flag, dm=dm, ddm=ddm,
                                 dew1=measure['ewerr'][wmatch][:,1]/1.e3,
                                 dew2=measure['ewerr'][wmatch][:,2]/1.e3,
                                 dew3=measure['ewerr'][wmatch][:,0]/1.e3)

    np.savez(f'{data_path}/{slitmask}_cat_{calibration}.npz', cat=cat,
    cat_err=cat_err, cat_flag=cat_flag)

    return cat, cat_err, cat_flag



def cat_rutledge(slitmask, objname, mag, magerr, ew1, ew2, ew3, cat_flag,
                 dm=24.47, ddm=0., dew1=None, dew2=None, dew3=None,
                 filter='vi'):

    #Equation 4 and 5 of Gilbert 2006
    #assume that V - V_HB is equivalent to g - g_HB because V - g is approximately constant
    #assume g-V = 0.56*(g-r+0.23)/1.05 + 0.12 (from Ivezic 2006?), and assume g-r = 0.6, such that g-V = 0.56
    #which means that M_g_HB = M_V_HB + (g-V) = +0.55 + 0.56 = 1.11, assuming that M_V_HB = +0.55

    #V_HB = 25.17 at the distance of M31, found in G06 (Holland et al. 1996)

    M_V_hb = 0.55 #8/9/21 -- this was 0.7 before. I don't know why.
    m_v_hb = M_V_hb + dm

    M_g_hb = 1.11 #8/9/21 -- this was 1.26 before.
    m_g_hb = M_g_hb + dm

    ew1 = np.array(ew1); ew2 = np.array(ew2); ew3 = np.array(ew3)

    if dew1 is not None:
        dew1 = np.array(dew1)
    if dew2 is not None:
        dew2 = np.array(dew2)
    if dew3 is not None:
        dew3 = np.array(dew3)

    if filter == 'vi':
        mag_above_hb = mag - m_v_hb
    if filter == 'cfht':
        mag_above_hb = mag - M_g_hb

    if magerr is not None:
        dmag = np.sqrt(magerr**2. + ddm**2.)
    else:
        dmag = np.zeros(mag.size)

    wbad = mag_above_hb < -5.
    mag_above_hb[wbad] = np.nan
    dmag[wbad] = np.nan
    cat_flag[wbad] = True

    a = 0.5; b = 1.; c = 0.6
    sigca = a*ew3 + b*ew1 + c*ew2

    if dew3 is not None and dew1 is not None and dew2 is not None:
        sigcaerr = np.sqrt( (a*dew3)**2. + (b*dew1)**2. + (c*dew2)**2. )
    else:
        sigcaerr = np.zeros(ew1.size)

    a = 0.42; b = 0.269; c = -2.66
    feh = c + a*sigca + b*mag_above_hb

    fehvar = (a * sigcaerr)**2. + (b * dmag)**2.
    feherr =  np.sqrt(fehvar)
    feherr[feherr == 0.] = np.nan

    return feh, feherr, cat_flag


def calculate_rproj(ra, dec, dm=24.47):

    distance = 10**(1 + dm/5.)*u.pc
    distance_kpc = distance.to(u.kpc)

    m31 = SkyCoord('0h42m44.3s', '41d16m09.0s', distance=distance_kpc)
    c = SkyCoord(ra, dec, unit=(u.deg, u.deg), distance=distance_kpc)

    rproj = c.separation_3d(m31).value

    return rproj


def calculate_xcmd(slitmask, color, mag, colorerr=None, magerr=None,
 filter='vi', age=12., dm=24.47, ddm=None, data_path='data',
 root=os.getcwd()):

    if os.path.exists(f'{data_path}/{slitmask}_xcmd.npy'):

        xcmd = np.load(f'{data_path}/{slitmask}_xcmd.npy')
        try: xcmd_err = np.load(f'{data_path}/{slitmask}_xcmderr.npy')
        except: xcmd_err = np.zeros(len(xcmd))

        return xcmd, xcmd_err

    if ddm is not None:
        if magerr is not None:
            magerr = np.sqrt(magerr**2. + ddm**2.)
        else:
            magerr = np.full(mag.size, ddm)
        if colorerr is None:
            colorerr = np.zeros(mag.size)

    Zsun = 0.0152

    if age <= 12.:
        age_str = str(int(age))
    else:
        age_str = str(int(age*10))

    if filter == 'vi':

        usecols = (0,9,30,32)

        Zi, label, magb, magr = np.loadtxt(f'{root}/isochrones/rgb/feh_vi_'+age_str+'_parsec.dat',
                                           skiprows=0, unpack=True,usecols=usecols)

    if (filter == 'cfht'):

        usecols = (0,7,24,26)

        Zi, label, magb, magr = np.loadtxt(f'{root}/isochrones/feh_cfht_'+age_str+'_parsec.dat',
                                           skiprows=0, unpack=True, usecols=usecols)

    feh_all = np.log10(Zi/Zsun)
    feh_min = feh_all.min()
    feh_max = feh_all.max()

    color_grid = []
    mag_grid = []
    xcmd_grid = []

    feh_val = np.unique(feh_all)
    for ii in range(len(feh_val)):

        wrgb = np.where( (label == 3) & (feh_all == feh_val[ii]) )[0]

        clr_f = magb[wrgb]-magr[wrgb]
        mag_f = magr[wrgb]+dm

        xcmd_f = ( feh_all[wrgb] - feh_min ) / (feh_max - feh_min )

        #Extrapolate beyond the tip of the red giant branch
        m, b = np.polyfit(clr_f[-2:], mag_f[-2:], 1)
        xx = np.arange(clr_f[-1], clr_f[-1]+1.2, 0.1)
        yy = xx*m + b

        zz = np.full(len(xx), xcmd_f[-1])

        clr_f = np.append(clr_f, xx)
        mag_f = np.append(mag_f, yy)
        xcmd_f = np.append(xcmd_f, zz)

        mag_grid.extend(mag_f)
        color_grid.extend(clr_f)
        xcmd_grid.extend(xcmd_f)

    xcmd = griddata((color_grid, mag_grid), xcmd_grid, (color, mag))

    out_of_bounds = np.isnan(xcmd)

    if len(xcmd[out_of_bounds]) > 0:

        arg = np.arange(0, len(xcmd))[out_of_bounds]

        f = SmoothBivariateSpline(color_grid, mag_grid, xcmd_grid, kx=1, ky=1)

        for ll in arg:

            xcmd_l = f(color[ll], mag[ll])[0][0]
            xcmd[ll] = xcmd_l

    if colorerr is not None and magerr is not None:

        nmc = 1000

        xcmd_err = np.zeros(len(xcmd))

        for ii in range(len(color)):

            rand = np.random.normal(size=(nmc,2))
            clr_mc = color[ii] + rand[:,0]*colorerr[ii]
            mag_mc = mag[ii] + rand[:,1]*magerr[ii]

            xcmd_mc = griddata((color_grid, mag_grid), xcmd_grid, (clr_mc, mag_mc), fill_value=np.nan)

            wgood_mc = np.where( (xcmd_mc >= -0.5) & (xcmd_mc <= 1.5) )[0]

            if len(wgood_mc) > 2:
                xcmd_err[ii] = np.std(xcmd_mc[wgood_mc])

            else:
                xcmd_err[ii] = np.nan

    else:
        xcmd_err = np.full(len(xcmd), np.nan)

    np.save(f'{data_path}/{slitmask}_xcmderr.npy', xcmd_err)
    np.save(f'{data_path}/{slitmask}_xcmd.npy', xcmd)

    return xcmd, xcmd_err
