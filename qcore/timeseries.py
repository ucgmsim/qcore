"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016
"""

from math import ceil, log, pi
import os

try:
    from scipy.signal import butter
except ImportError:
    print('SciPy not installed. Certain functions will fail.')
# sosfilt new in scipy 0.16
# sosfiltfilt new in scipy 0.18
try:
    butter
    from scipy.signal import sosfiltfilt
except NameError:
    pass
except ImportError:
    from qcore.sosfiltfilt import sosfiltfilt
import numpy as np
rfft = np.fft.rfft
irfft = np.fft.irfft

# butterworth filter
# bandpass not necessary as sampling frequency too low
def bwfilter(data, dt, freq, band, match_powersb = True):
    """
    data: np.array to filter
    freq: cutoff frequency
    band: 'highpass' or 'lowpass'
    """
    # power spectrum based LF/HF filter (shift cutoff)
    # readable code commented, fast code uncommented
    #order = 4
    #x = 1.0 / (2.0 * order)
    #if band == 'lowpass':
    #    x *= -1
    #freq *= exp(x * log(sqrt(2.0) - 1.0))
    nyq = 1.0 / (2.0 * dt)
    if match_powersb:
        if band == 'highpass':
            freq *= 0.8956803352330285
        else:
            freq *= 1.1164697500474103
    return sosfiltfilt( \
            butter(4, freq / nyq, btype = band, output = 'sos'), \
            data, padtype = None)

def get_ft_len(nt):
    """
    Length the fourier transform should be
    given timeseries length nt.
    """
    return int(2 ** ceil(log(nt) / log(2)))

def ampdeamp(timeseries, ampf, amp = True):
    """
    Amplify or Deamplify timeseries.
    """
    nt = len(timeseries)

    # length the fourier transform should be
    ft_len = get_ft_len(nt)

    # taper 5% on the right using the hanning method
    ntap = int(nt * 0.05)
    timeseries[nt - ntap:] *= np.hanning(ntap * 2 + 1)[ntap + 1:]

    # extend array, fft
    timeseries = np.resize(timeseries, ft_len)
    timeseries[nt:] = 0
    fourier = rfft(timeseries)

    # ampf modified for de-amplification
    if not amp:
        ampf = 1 / ampf
    # last value of fft is some identity value
    fourier[:-1] *= ampf

    return irfft(fourier)[:nt]

def transf(vs_soil, rho_soil, damp_soil, height_soil, \
        vs_rock, rho_rock, damp_rock, nt, dt):
    """
    Used in deconvolution. Made by Chris de la Torre.
    vs = shear wave velocity (upper soil or rock)
    rho = density
    damp = damping ratio
    height_soil = height of soil above rock
    nt = number of timesteps
    dt = delta time in timestep (seconds)
    """
    ft_len = get_ft_len(nt)
    # TODO: before it was ft_len / 2 + 1 but this may be an error
    # the last value isn't an ft value
    ft_freq = np.arange(0, ft_len / 2) * (1 / (ft_len * dt))

    omega = 2.0 * pi * ft_freq
    Gs = rho_soil * vs_soil ** 2.0
    Gr = rho_rock * vs_rock ** 2.0

    kS = omega / (vs_soil * (1.0 + 1j * damp_soil))
    kR = omega / (vs_rock * (1.0 + 1j * damp_rock))

    alpha = Gs * kS / (Gr * kR)

    H = 2.0 / ((1.0 + alpha) * np.exp(1j * jS * hS) + (1.0 - alpha) \
            * np.exp(-1j * kS * hS))
    H[0] = 1
    return H

def read_ascii(filepath, meta = False, t0 = False):
    """
    Read timeseries data from standard ascii file to numpy array.
    meta: also return first 2 lines (metadata) as string lists
    t0: adjust data to start from t = 0 (uses secs and not hr:min)
    """
    with open(filepath, 'r') as ts:
        info1 = ts.readline().split()
        info2 = ts.readline().split()

        vals = np.array(map(float, \
                ' '.join(map(str.rstrip, ts.readlines())).split()))

    # check if header length correct for integrity
    try:
        assert(len(vals) == int(info2[0]))
    except AssertionError:
        print('File entries don\'t match NT: %s' % (filepath))
        raise

    # start from t = 0 by (truncating) or (inserting 0s) at beginning
    if t0:
        # t start / dt = number of missing points
        diff = int(round(float(info2[4]) / float(info2[1])))
        if diff < 0:
            # remove points before t = 0
            vals = vals[abs(diff):]
        elif diff > 0:
            # insert zeros between t = 0 and start
            vals = np.append(np.zeros(diff), vals)

    if meta:
        return info1, info2, vals
    return vals

def vel2acc(timeseries, dt):
    """
    Differentiate following Rob Graves' code logic.
    """
    return np.diff(np.hstack(([0], timeseries)) * (1.0 / dt))

def acc2vel(timeseries, dt):
    """
    Integrates following Rob Graves' code logic (simple).
    """
    return np.cumsum(timeseries) * dt

def pgv2MMI(pgv):
    """
    Calculates MMI from pgv based on Worden et al (2012)
    """
    return np.where(np.log10(pgv) < 0.53,
                    3.78 + 1.47 * np.log10(pgv),
                    2.89 + 3.16 * np.log10(pgv))

###
### PROCESSING OF LF BINARY CONTAINER
###
class LFSeis:
    pass

###
### PROCESSING OF HF BINARY CONTAINER
###
class HFSeis:
    HEAD_SIZE = 0x200
    HEAD_STAT = 0x18
    N_COMP = 3

    def __init__(self, hf_path):
        """
        Load HF binary store.
        hf_path: path to the HF binary file
        """

        hfs = os.stat(hf_path).st_size
        hff = open(hf_path, 'rb')
        # determine endianness by checking file size
        nstat, nt = np.fromfile(hff, dtype = '<i4', count = 2)
        if hfs == self.HEAD_SIZE + nstat * self.HEAD_STAT \
                                   + nstat * nt * self.N_COMP * 4:
            endian = '<'
        elif hfs == self.HEAD_SIZE \
                    + nstat.byteswap() * self.HEAD_STAT \
                    + nstat.byteswap() * nt.byteswap() * self.N_COMP * 4:
            endian = '>'
        else:
            hff.close()
            raise ValueError('File is not an HF seis file: %s' % (hf_path))
        hff.seek(0)

        # read header - integers
        self.nstat, self.nt, self.seed, siteamp, self.pdur_model, \
                nrayset, rayset1, rayset2, rayset3, rayset4, \
                self.nbu, self.ift, self.nlskip, icflag, same_seed, \
                site_specific_vm = \
                        np.fromfile(hff, dtype = '%si4' % (endian), count = 16)
        self.siteamp = bool(siteamp)
        self.rayset = [rayset1, rayset2, rayset3, rayset4][:nrayset]
        self.icflag = bool(icflag)
        self.seed_inc = not bool(same_seed)
        self.site_specific_vm = bool(site_specific_vm)
        # read header - floats
        self.duration, self.dt, self.start_sec, self.sdrop, self.flo, self.fhi, \
                self.rvfac, self.rvfac_shal, self.rvfac_deep, \
                self.czero, self.calpha, self.mom, self.rupv, self.vs_moho, \
                self.vp_sig, self.vsh_sig, self.rho_sig, self.qs_sig, \
                self.fa_sig1, self.fa_sig2, self.rv_sig1 = \
                        np.fromfile(hff, dtype = '%sf4' % (endian), count = 21)
        # read header - strings
        self.stoch_file, self.velocity_model = \
                np.fromfile(hff, dtype = '|S64', count = 2)

        # load station info
        hff.seek(self.HEAD_SIZE)
        self.stations = np.fromfile(hff, count = self.nstat, \
                dtype = [('lon', 'f4'), ('lat', 'f4'), ('name', '|S8'), \
                         ('e_dist', 'f4'), ('seed_inc', 'i4')])
        hff.close()

        # only map the timeseries
        self.data = np.memmap(hf_path, dtype = '%sf4' % (endian), \
                mode = 'r', offset = self.HEAD_SIZE + nstat * self.HEAD_STAT, \
                shape = (self.nstat, self.nt, self.N_COMP))

###
### PROCESSING OF BB BINARY CONTAINER
###
class BBSeis:
    pass
