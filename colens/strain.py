from pycbc import DYN_RANGE_FAC
from pycbc.filter import highpass, resample_to_delta_t
from pycbc.types import float32


def process_strain_dict(strain_dict, strain_high_pass, sample_rate, pad):
    for ifo in strain_dict:
        strain_tmp = highpass(strain_dict[ifo], frequency=strain_high_pass)
        strain_tmp = resample_to_delta_t(strain_tmp, 1.0 / sample_rate, method="ldas")
        strain_tmp = (strain_tmp * DYN_RANGE_FAC).astype(float32)
        strain_tmp = highpass(strain_tmp, frequency=strain_high_pass)
        start = int(pad * sample_rate)
        end = int(len(strain_tmp) - sample_rate * pad)
        strain_tmp = strain_tmp[start:end]
        strain_tmp.gating_info = dict()
        strain_dict[ifo] = strain_tmp
    return strain_dict
