import logging

import h5py
import numpy as np
import pycbc
from pycbc.waveform.bank import FilterBank


class MyFilterBank(FilterBank):
    def __init__(
        self,
        filename,
        filter_length,
        delta_f,
        dtype,
        out=None,
        max_template_length=None,
        approximant=None,
        parameters=None,
        enable_compressed_waveforms=True,
        low_frequency_cutoff=None,
        waveform_decompression_method=None,
        **kwds
    ):
        self.out = out
        self.dtype = dtype
        self.f_lower = low_frequency_cutoff
        self.filename = filename
        self.delta_f = delta_f
        self.N = (filter_length - 1) * 2
        self.delta_t = 1.0 / (self.N * self.delta_f)
        self.filter_length = filter_length
        self.max_template_length = max_template_length
        self.enable_compressed_waveforms = enable_compressed_waveforms
        self.waveform_decompression_method = waveform_decompression_method

        self.template_bank__init__(
            filename, approximant=approximant, parameters=parameters, **kwds
        )
        self.ensure_standard_filter_columns(low_frequency_cutoff=low_frequency_cutoff)

    def template_bank__init__(
        self, filename, approximant=None, parameters=None, **kwds
    ):
        self.has_compressed_waveforms = False

        self.indoc = None
        f = pycbc.io.HFile(filename, "r")
        # just assume all of the top-level groups are the parameters
        fileparams = list(f.keys())
        logging.info(
            "WARNING: no parameters attribute found. "
            "Assuming that %s " % (", ".join(fileparams)) + "are the parameters."
        )
        tmp_params = []
        # At this point fileparams might be bytes. Fix if it is
        for param in fileparams:
            try:
                param = param.decode()
                tmp_params.append(param)
            except AttributeError:
                tmp_params.append(param)
        fileparams = tmp_params

        # use WaveformArray's syntax parser to figure out what fields
        # need to be loaded
        if parameters is None:
            parameters = fileparams
        common_fields = list(pycbc.io.WaveformArray(1, names=parameters).fieldnames)
        add_fields = list(set(parameters) & (set(fileparams) - set(common_fields)))
        # load
        dtype = []
        data = {}
        for key in common_fields + add_fields:
            data[key] = f[key][:]
            dtype.append((key, data[key].dtype))
        num = f[fileparams[0]].size
        self.table = pycbc.io.WaveformArray(num, dtype=dtype)
        for key in data:
            self.table[key] = data[key]
        # add the compressed waveforms, if they exist
        self.has_compressed_waveforms = "compressed_waveforms" in f

        # if approximant is specified, override whatever was in the file
        # (if anything was in the file)
        if approximant is not None:
            # get the approximant for each template
            dtype = h5py.string_dtype(encoding="utf-8")
            apprxs = np.array(self.parse_approximant(approximant), dtype=dtype)
            if "approximant" not in self.table.fieldnames:
                self.table = self.table.add_fields(apprxs, "approximant")
            else:
                self.table["approximant"] = apprxs
        self.extra_args = kwds
        self.ensure_hash()
