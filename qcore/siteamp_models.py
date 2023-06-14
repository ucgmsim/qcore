"""
site amplification functionality from wcc_siteamp.c
Acceleration amplification models.

@date 24 June 2016
@author Viktor Polak
@contact viktor.polak@canterbury.ac.nz

Implemented Models
==============================
cb_amp (version = "2008"):
    Based on Campbell and Bozorgnia 2008 - added 24 June 2016
cb_amp (version = "2014"):
    Based on Campbell and Bozorgnia 2014 - added 22 September 2016
ba18_amp (version 2018):
    Based on Bayless Fourier Amplitude Spectra Empirical Model - added 11 June 2020

bssa14_amp (version 2014):
    Based on Boore et all 2014  - added 7 June 2023

Usage
==============================
from siteamp_models import cb_amp (or *)
cb_amp(variables, ...)
"""

# math functions faster than numpy for non-vector data
from math import ceil, exp, log
import os

import numpy as np
import pandas as pd

from qcore.uncertainties import distributions

ba18_coefs_df = None
bssa14_coefs_df = None

def amplification_uncertainty(
    amplification_factors, frequencies, seed=None, std_dev_limit=2
):
    """
    Applies an uncertainty factor to each value in an amplification spectrum
    :param amplification_factors: numpy array of amplification factors
    :param frequencies: numpy array of frequencies for the amplification factors
    :param seed: Seed to use for the prng. None for a random seed
    :return: amplification factors with uncertainty applied
    """
    sigma_x = (
        0.6167
        - 0.1495 / (1 + np.exp(-3.6985 * np.log(frequencies / 0.7248)))
        + 0.3640 / (1 + np.exp(-2.2497 * np.log(frequencies / 13.457)))
    )
    amp_function_output = np.ones_like(amplification_factors)
    amp_function_output[1:] = distributions.truncated_log_normal(
        amplification_factors[1:],
        sigma_x,
        std_dev_limit=std_dev_limit,
        seed=seed,
    )
    return amp_function_output


def init_ba18():
    global ba18_coefs_df
    __location__ = os.path.realpath(os.path.dirname(__file__))
    ba18_coefs_file = os.path.join(
        __location__, "siteamp_coefs_files", "Bayless_ModelCoefs.csv"
    )
    ba18_coefs_df = pd.read_csv(ba18_coefs_file, index_col=0)

def init_bssa14():
    global bssa14_coefs_df
    __location__ = os.path.realpath(os.path.dirname(__file__))
    bssa14_coefs_file = os.path.join(
        __location__, "siteamp_coefs_files", "05_eqs-070113eqs184m_suppl_es1_030915.csv"
    )
    bssa14_coefs_df = pd.read_csv(bssa14_coefs_file, index_col=0, skiprows=1)


def nt2n(nt):
    """
    Length the fourier transform should be
    given timeseries length nt.
    """
    return int(2 ** ceil(log(nt) / log(2)))


def cb_amp(
    dt,
    n,
    vref,
    vsite,
    vpga,
    pga,
    version="2014",
    flowcap=0.0,
    fmin=0.2,
    fmidbot=0.5,
    fmid=1.0,
    fhigh=10 / 3.0,
    fhightop=10.0,
    fmax=15.0,
):
    # cb constants
    scon_c = 1.88
    scon_n = 1.18

    # fmt: off
    freqs = 1.0 / np.array([0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
                            0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75,
                            1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.50, 10.0])
    if version == '2008':
        c10 = np.array([1.058, 1.058, 1.102, 1.174, 1.272, 1.438, 1.604,
                        1.928, 2.194, 2.351, 2.460, 2.587, 2.544, 2.133,
                        1.571, 0.406,-0.456,-0.82, -0.82, -0.82, -0.82, -0.82])
    elif version == '2014':
        # named c11 in cb2014
        c10 = np.array([1.090, 1.094, 1.149, 1.290, 1.449, 1.535, 1.615,
                        1.877, 2.069, 2.205, 2.306, 2.398, 2.355, 1.995,
                        1.447, 0.330, -0.514, -0.848, -0.793, -0.748, -0.664,
                        -0.576])
    else:
        raise Exception(f"BAD CB AMP version specified: {version}")
    k1 = np.array([865.0, 865.0, 865.0, 908.0, 1054.0, 1086.0, 1032.0,
                   878.0, 748.0, 654.0, 587.0,  503.0,  457.0,  410.0,
                   400.0, 400.0, 400.0, 400.0,  400.0,  400.0,  400.0, 400.0])
    k2 = np.array([-1.186, -1.186, -1.219, -1.273, -1.346, -1.471, -1.624,
                   -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.401,
                   -1.955, -1.025, -0.299,  0.0,    0.0,    0.0,    0.0, 0.0])
    # fmt: on

    # f_site function domains
    def fs_low(T, vs30, a1100):
        return c10[T] * log(vs30 / k1[T]) + k2[T] * log(
            (a1100 + scon_c * exp(scon_n * log(vs30 / k1[T]))) / (a1100 + scon_c)
        )

    def fs_mid(T, vs30, a1100=None):
        return (c10[T] + k2[T] * scon_n) * log(vs30 / k1[T])

    def fs_high(T, vs30=None, a1100=None):
        return (c10[T] + k2[T] * scon_n) * log(1100.0 / k1[T])

    def fs_auto(T, vs30):
        return fs_low if vs30 < k1[T] else fs_mid if vs30 < 1100.0 else fs_high

    #                 fs1100     - fs_vpga
    a1100 = pga * exp(fs_high(0) - fs_auto(0, vpga)(0, vpga, pga))

    # calculate factor for each period
    it = (
        exp(fs_auto(T, vsite)(T, vsite, a1100) - fs_auto(T, vref)(T, vref, a1100))
        for T in range(freqs.size)
    )
    ampf0 = np.fromiter(it, dtype=np.float64)
    try:
        # T is the first occurance of a value <= flowcap
        T = np.flatnonzero((freqs <= flowcap))[0]
        ampf0[T:] = ampf0[T]
    except IndexError:
        pass

    ampv, ftfreq = interpolate_frequency(freqs, ampf0, dt, n)
    ampf = amp_bandpass(ampv, fhightop, fmax, fmidbot, fmin, ftfreq)

    return ampf


def interpolate_frequency(freqs, ampf0, dt, n):
    # frequencies of fourier transform
    ftfreq = get_ft_freq(dt, n)
    # transition indexes
    digi = np.digitize(freqs, ftfreq)[::-1]
    # only go down to 2nd frequency
    ampf0[0] = ampf0[1]
    freqs[0] = freqs[1]
    # special case, first frequency in transition range
    ftfreq0 = int(digi[0] == 0)
    # all possible dadf factors
    dadf0 = np.zeros(freqs.size)
    for i in range(1, freqs.size - 1):
        # start with dadf = 0.0 if no freq change at pos 0
        dadf0[-i - 1 - ftfreq0] = (ampf0[i] - ampf0[i + 1]) / log(
            freqs[i] / freqs[i + 1]
        )
    # calculate amplification factors
    digi = np.hstack((digi, [ftfreq.size]))
    a0 = np.zeros(ftfreq.size)
    f0 = np.zeros(ftfreq.size)
    dadf = np.zeros(ftfreq.size)
    start = 0
    start_dadf = 0
    for i in range(freqs.size):
        end = max(start + 1, digi[i + 1])
        end_dadf = max(start_dadf + 1, digi[i + ftfreq0])
        a0[start:end] = ampf0[-i - 1]
        f0[start:end] = freqs[-i - 1]
        dadf[start_dadf:end_dadf] = dadf0[i]
        start = max(end, digi[i + 1])
        start_dadf = end_dadf
    ampv = a0 + dadf * np.log(ftfreq / f0)
    return ampv, ftfreq


def get_ft_freq(dt, n):
    return np.arange(1, n / 2) * (1.0 / (n * dt))


def amp_bandpass(ampv, fhightop, fmax, fmidbot, fmin, ftfreq):
    # default amplification is 1.0 (keeping values the same)
    ampf = np.ones(ftfreq.size + 1, dtype=np.float64)
    # amplification factors applied differently at different bands
    ampf[1:] += (
        np.where(
            (ftfreq >= fhightop) & (ftfreq < fmax),
            -1 + ampv + np.log(ftfreq / fhightop) * (1.0 - ampv) / log(fmax / fhightop),
            0,
        )
        + np.where((ftfreq >= fmidbot) & (ftfreq < fhightop), -1 + ampv, 0)
        + np.where(
            (ftfreq >= fmin) & (ftfreq < fmidbot),
            np.log(ftfreq / fmin) * (ampv - 1.0) / log(fmidbot / fmin),
            0,
        )
    )
    return ampf


def ba18_amp(
    dt,
    n,
    vref,
    vs,
    vpga,
    pga,
    flowcap=0.0,
    fmin=0.00001,
    fmidbot=0.0001,
    fmid=1.0,
    fhigh=10 / 3.0,
    fhightop=999.0,
    fmax=1000,
    z1=None,
):
    """

    :param dt:
    :param n:
    :param vref: Reference vs used for waveform
    :param vs: Actual vs30 value of the site
    :param vpga: Reference vs for HF
    :param pga: PGA value from HF
    :param version: unused
    :param flowcap: unused
    :param fmin:
    :param fmidbot:
    :param fmid:
    :param fhigh:
    :param fhightop:
    :param fmax:
    :param kwargs: to pass optional arguments such as include_fZ1=True
    :return:
    """
    if vs > 1000:
        vs = 999  # maximum vs30 supported by the model is 999, so caps the vsite to that value

    # overwrite these two values to their default value, so changes by the caller function do not override this
    fmin = 0.00001
    fmidbot = 0.0001


    ref, __ = ba_18_site_response_factor(vref, pga, vpga, z1)
    vsite, freqs = ba_18_site_response_factor(vs, pga, vpga, z1)

    amp = np.exp(vsite - ref)
    ftfreq = get_ft_freq(dt, n)

    ampi = np.interp(ftfreq, freqs, amp)
    ampfi = amp_bandpass(
        ampi, fhightop, fmax, fmidbot, fmin, ftfreq
    )  # With these values it is effectively no filtering
    ampfi[0] = ampfi[1]  # Copies the first value, which isn't necessarily 1

    return ampfi


def ba_18_site_response_factor(vs, pga, vpga, z1=None, f=None):
    vsref = 1000

    if ba18_coefs_df is None:
        print(
            "You need to call the init_ba18 function before using the site_amp functions"
        )
        exit()
    coefs = type("coefs", (object,), {})  # creates a custom object for coefs

    if f is None:
        freq_indices = ...
        coefs.freq = ba18_coefs_df.index.values
    else:
        freq_index = np.argmin(np.abs(ba18_coefs_df.index.values - f))
        if freq_index > f:
            freq_indices = [freq_index - 1, freq_index]
        else:
            freq_indices = [freq_index, freq_index + 1]
        coefs.freq = ba18_coefs_df.index.values[freq_indices]

    # Non-linear site parameters
    coefs.f3 = ba18_coefs_df.f3.values[freq_indices]
    coefs.f4 = ba18_coefs_df.f4.values[freq_indices]
    coefs.f5 = ba18_coefs_df.f5.values[freq_indices]
    coefs.b8 = ba18_coefs_df.c8.values[freq_indices]
    coefs.c11a = ba18_coefs_df.c11a.values[freq_indices]
    coefs.c11b = ba18_coefs_df.c11b.values[freq_indices]
    coefs.c11c = ba18_coefs_df.c11c.values[freq_indices]
    coefs.c11d = ba18_coefs_df.c11d.values[freq_indices]


    lnfas = coefs.b8 * np.log(min(vs, 1000) / vsref)

    fas_lin = np.exp(lnfas)

    if f is None:
        # Extrapolate to 100 Hz
        maxfreq = 23.988321
        imax = np.where(coefs.freq == maxfreq)[0][0]
        fas_maxfreq = fas_lin[imax]
        # Kappa
        kappa = np.exp(-0.4 * np.log(vs / 760) - 3.5)
        # Diminuition operator
        D = np.exp(-np.pi * kappa * (coefs.freq[imax:] - maxfreq))

        fas_lin = np.append(fas_lin[:imax], fas_maxfreq * D)
        lnfas = np.log(fas_lin)

    if vs <= 200:
        coefs.c11 = coefs.c11a
    elif 200 < vs <= 300:
        coefs.c11 = coefs.c11b
    elif 300 < vs <= 500:
        coefs.c11 = coefs.c11c
    elif vs > 500:
        coefs.c11 = coefs.c11d
    z1ref = (1/1000) * np.exp((-7.67/4) * np.log((vs**4 + 610**4)/(1360**4 + 610**4)))

    if z1 is not None:
        fZ1 = coefs.c11 * np.log((min(z1,2)+0.01)/(z1ref+0.01))
    else:
        fZ1 = 0



    # Compute non-linear site response
    if pga is not None:
        v_model_ref = 760
        if vpga != v_model_ref:
            IR = pga * exp(
                ba_18_site_response_factor(
                    vs=v_model_ref, pga=None, vpga=v_model_ref, z1=z1, f=5
                )[0]
                - ba_18_site_response_factor(vs=vpga, pga=pga, vpga=v_model_ref, z1=z1, f=5)[0]
            )
        else:
            IR = pga

        coefs.f2 = coefs.f4 * (
            np.exp(coefs.f5 * (min(vs, v_model_ref) - 360))
            - np.exp(coefs.f5 * (v_model_ref - 360))
        )
        fnl0 = coefs.f2 * np.log((IR + coefs.f3) / coefs.f3)

        fnl0[np.where(fnl0 == min(fnl0))[0][0] :] = min(fnl0)

    else:
        fnl0 = 0

    result = fnl0 + lnfas + fZ1

    if f is not None:
        return np.interp(f, coefs.freq, result), f
    else:
        return result, coefs.freq

def bssa14_amp(
    dt,
    n,
    vref,
    vs,
    vpga,
    pga,
    flowcap=0.0,
    fmin=0.00001,
    fmidbot=0.0001,
    fmid=1.0,
    fhigh=10 / 3.0,
    fhightop=999.0,
    fmax=1000,
    z1=None,
):
    """
    :param dt:
    :param n:
    :param vref: Reference vs used for waveform
    :param vs: Actual vs30 value of the site
    :param vpga: Reference vs for HF
    :param pga: PGA value from HF 
    :param version: unused
    :param flowcap: unused
    :param fmin:
    :param fmidbot:
    :param fmid:
    :param fhigh:
    :param fhightop:
    :param fmax:
    :param z1: 
    :return:
    """
    if vs > 1000:
        vs = 999  # maximum vs30 supported by the model is 999, so caps the vsite to that value

    # overwrite these two values to their default value, so changes by the caller function do not override this
    fmin = 0.00001
    fmidbot = 0.0001


    ref, __ = bssa_14_site_response_factor(vref, pga, vpga, z1)
    vsite, freqs = bssa_14_site_response_factor(vs, pga, vpga, z1)

    amp = np.exp(vsite - ref)
    ftfreq = get_ft_freq(dt, n)

    ampi = np.interp(ftfreq, freqs, amp)
    ampfi = amp_bandpass(
        ampi, fhightop, fmax, fmidbot, fmin, ftfreq
    )  # With these values it is effectively no filtering
    ampfi[0] = ampfi[1]  # Copies the first value, which isn't necessarily 1

    return ampfi


def bssa_14_site_response_factor(vs, pga, vpga, z1=None):

    if bssa14_coefs_df is None:
        print(
            "You need to call the init_bssa18 function before using the site_amp functions"
        )
        exit()
    coefs = type("coefs", (object,), {})  # creates a custom object for coefs

    period_indices = bssa14_coefs_df.index.values
    

    # Lnear site parameters
    coefs.c = bssa14_coefs_df.c
    coefs.vc = bssa14_coefs_df.vc
    coefs.vref = bssa14_coefs_df.vref
    
    # Non-linear site parameters
    coefs.f1 = bssa14_coefs_df.f1
    coefs.f3 = bssa14_coefs_df.f3
    coefs.f4 = bssa14_coefs_df.f4
    coefs.f5 = bssa14_coefs_df.f5

    # basin depth
    coefs.f6 = bssa14_coefs_df.f6
    coefs.f7 = bssa14_coefs_df.f7

    min_vc_vs=coefs.vc.copy()
    min_vc_vs.loc[min_vc_vs>=vs]=vs # minimum of vc and vs is taken

    lnFlin = coefs.c*np.log(min_vc_vs/ coefs.vref)

    # eq 8.
    f2 = coefs.f4*(np.exp(coefs.f5*(min(vs,760)-360))-np.exp(coefs.f5*(760-360)))

    if pga is not None: # 
        v_model_ref = 760
        if vpga != v_model_ref:
            fs760 = bssa_14_site_response_factor(vs=v_model_ref, pga=None, vpga=v_model_ref, z1=z1)
            fs_vpga = bssa_14_site_response_factor(vs=vpga, pga=pga, vpga=v_model_ref, z1=z1)
            pga_r = pga*np.exp(fs760[0] - fs_vpga[0]) #for period 0
                               
        else:
            pga_r = pga
        
        lnFnl = coefs.f1 + f2 * np.log((pga_r+coefs.f3)/coefs.f3)
    else:        
        lnFnl = coefs.f1

    # prediction of an empirical model relating z1 to vs30. Using Califorrnia model    
    Mu_z1 = np.exp(-7.15/4*np.log((vs**4+570.94**4)/(1360**4+570.94**4))-np.log(1000))

    if z1 is not None:
        dZ1 = z1 - Mu_z1
        f7_f6_ratio = coefs.f7 / coefs.f6

        fdz1=pd.Series(0,index=coefs.f7.index)
        fdz1.loc[(fdz1.index>0.65) & (f7_f6_ratio >= dZ1)] = coefs.f6*dZ1
        fdz1.loc[(fdz1.index>0.65) & (f7_f6_ratio < dZ1)] = coefs.f7
        
    else:
        fdz1 = 0

    result = lnFlin + lnFnl + fdz1

    return result





def hashash_get_pgv(fnorm, mag, rrup, ztor):
    # get the EAS_rock at 5 Hz (no c8, c11 terms)
    b4a = -0.5
    mbreak = 6.0

    coefs = type("coefs", (object,), {})  # creates a custom object for coefs
    coefs.freq = ba18_coefs_df.index.values

    coefs.b1 = ba18_coefs_df.c1.values
    coefs.b2 = ba18_coefs_df.c2.values
    coefs.b3quantity = ba18_coefs_df["(c2-c3)/cn"].values
    coefs.bn = ba18_coefs_df.cn.values
    coefs.bm = ba18_coefs_df.cM.values
    coefs.b4 = ba18_coefs_df.c4.values
    coefs.b5 = ba18_coefs_df.c5.values
    coefs.b6 = ba18_coefs_df.c6.values
    coefs.bhm = ba18_coefs_df.chm.values
    coefs.b7 = ba18_coefs_df.c7.values
    coefs.b8 = ba18_coefs_df.c8.values
    coefs.b9 = ba18_coefs_df.c9.values
    coefs.b10 = ba18_coefs_df.c10.values
    # row = df.iloc[df.index == 5.011872]
    i5 = np.where(coefs.freq == 5.011872)
    lnfasrock5Hz = coefs.b1[i5]
    lnfasrock5Hz += coefs.b2[i5] * (mag - mbreak)
    lnfasrock5Hz += coefs.b3quantity[i5] * np.log(
        1 + np.exp(coefs.bn[i5] * (coefs.bm[i5] - mag))
    )
    lnfasrock5Hz += coefs.b4[i5] * np.log(
        rrup + coefs.b5[i5] * np.cosh(coefs.b6[i5] * max(mag - coefs.bhm[i5], 0))
    )
    lnfasrock5Hz += (b4a - coefs.b4[i5]) * np.log(np.sqrt(rrup**2 + 50**2))
    lnfasrock5Hz += coefs.b7[i5] * rrup
    lnfasrock5Hz += coefs.b9[i5] * min(ztor, 20)
    lnfasrock5Hz += coefs.b10[i5] * fnorm
    # Compute PGA_rock extimate from 5 Hz FAS
    IR = np.exp(1.238 + 0.846 * lnfasrock5Hz)
    return IR
