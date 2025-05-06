from itertools import groupby
from operator import itemgetter

from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr
from colens.data_loader import DataLoader


class Runner:
    def __init__(self, coherent_func, get_snr, output_data, data_loader: DataLoader):
        self.output_data = output_data
        self.data_loader = data_loader
        self.coherent_func = coherent_func
        self.get_snr = get_snr
        self.data_loader.sky_position_index = 0

    def run(self):
        new_iterator = _create_iterator(
            self.data_loader.timing_iterator,
            [self.first_function, self.second_function],
        )
        for _ in new_iterator:
            self._run_single()
            self.write_output()

    def first_function(self, arg):
        self.data_loader.lensed_trigger_time_seconds = (
            self.data_loader.time_gps_future_seconds_for_iterator[arg]
        )

    def second_function(self, arg):
        self.data_loader.ra = self.data_loader.ra_for_iterator[arg]
        self.data_loader.dec = self.data_loader.dec_for_iterator[arg]
        self.data_loader.original_trigger_time_seconds = (
            self.data_loader.time_gps_past_seconds_for_iterator[arg]
        )
        self.data_loader.calculate_antenna_pattern(
            self.data_loader.ra,
            self.data_loader.dec,
            self.data_loader.original_trigger_time_seconds,
            self.data_loader.lensed_trigger_time_seconds,
        )
        self.data_loader.get_time_delay_at_zerolag_seconds(
            self.data_loader.original_trigger_time_seconds,
            self.data_loader.lensed_trigger_time_seconds,
            self.data_loader.ra,
            self.data_loader.dec,
        )
        self.data_loader.get_time_delay_indices()
        self.data_loader.get_snr_at_trigger(
            self.get_snr,
            self.data_loader.sky_position_index,
            self.data_loader.original_trigger_time_seconds,
            self.data_loader.lensed_trigger_time_seconds,
            self.data_loader.time_slide_index,
        )
        self.fp = [
            self.data_loader.unlensed_antenna_pattern[ifo][
                self.data_loader.sky_position_index
            ][0]
            for ifo in self.data_loader.unlensed_detectors
        ]
        self.fc = [
            self.data_loader.unlensed_antenna_pattern[ifo][
                self.data_loader.sky_position_index
            ][1]
            for ifo in self.data_loader.unlensed_detectors
        ]
        self.fp += [
            self.data_loader.lensed_antenna_pattern[ifo][
                self.data_loader.sky_position_index
            ][0]
            for ifo in self.data_loader.lensed_detectors
        ]
        self.fc += [
            self.data_loader.lensed_antenna_pattern[ifo][
                self.data_loader.sky_position_index
            ][1]
            for ifo in self.data_loader.lensed_detectors
        ]

    def _run_single(self):
        self.rho_coinc = coincident_snr(self.data_loader.snr_at_trigger)
        M_mu_nu, x_mu = coherent_statistic_adapter(
            self.data_loader.snr_at_trigger, self.data_loader.sigma, self.fp, self.fc
        )
        self.rho_coh = self.coherent_func(M_mu_nu, x_mu) ** 0.5

    def write_output(self):
        self.output_data.original_trigger_time_seconds.append(
            self.data_loader.original_trigger_time_seconds
        )
        self.output_data.lensed_trigger_time_seconds.append(
            self.data_loader.lensed_trigger_time_seconds
        )
        self.output_data.time_slide_index.append(self.data_loader.time_slide_index)
        self.output_data.ra.append(self.data_loader.ra)
        self.output_data.dec.append(self.data_loader.dec)
        self.output_data.H1.snr_real.append(
            float(self.data_loader.snr_at_trigger_original[0].real)
        )
        self.output_data.H1.snr_imag.append(
            float(self.data_loader.snr_at_trigger_original[0].imag)
        )
        self.output_data.L1.snr_real.append(
            float(self.data_loader.snr_at_trigger_original[1].real)
        )
        self.output_data.L1.snr_imag.append(
            float(self.data_loader.snr_at_trigger_original[1].imag)
        )
        self.output_data.H1_lensed.snr_real.append(
            float(self.data_loader.snr_at_trigger_lensed[0].real)
        )
        self.output_data.H1_lensed.snr_imag.append(
            float(self.data_loader.snr_at_trigger_lensed[0].imag)
        )
        self.output_data.L1_lensed.snr_real.append(
            float(self.data_loader.snr_at_trigger_lensed[1].real)
        )
        self.output_data.L1_lensed.snr_imag.append(
            float(self.data_loader.snr_at_trigger_lensed[1].imag)
        )
        self.output_data.rho_coinc.append(float(self.rho_coinc[0]))
        self.output_data.rho_coh.append(float(self.rho_coh))


def _create_iterator(generator, functions):
    def inner(gen, func_idx):
        for i, group in groupby(gen, key=itemgetter(func_idx)):
            functions[func_idx](i)
            if func_idx < len(functions) - 2:  # TODO find a better way to do this
                yield from inner(group, func_idx + 1)
            else:  # pause iteration on the innermost for loop
                for i_, group_ in groupby(group, key=itemgetter(func_idx + 1)):
                    functions[func_idx + 1](i_)
                    yield

    iterator = inner(generator, 0)
    return iterator
