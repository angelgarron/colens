import numpy as np
from pycbc.events import findchirp_cluster_over_window

from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr


class Runner:
    def __init__(
        self,
        coherent_func,
        output_data,
        snr_handler,
        snr_handler_lensed,
        iterator_handler,
    ):
        self.output_data = output_data
        self.snr_handler = snr_handler
        self.snr_handler_lensed = snr_handler_lensed
        self.coherent_func = coherent_func
        self.iterator_handler = iterator_handler

    def run(self):
        for _ in self.iterator_handler.timing_iterator:
            self._run_single()
            self.write_output()

        self.cluster_output()

    def _run_single(self):
        self.rho_coinc = coincident_snr(
            self.snr_handler.snr_at_trigger + self.snr_handler_lensed.snr_at_trigger
        )
        M_mu_nu, x_mu = coherent_statistic_adapter(
            self.snr_handler.snr_at_trigger + self.snr_handler_lensed.snr_at_trigger,
            self.snr_handler.sigma + self.snr_handler_lensed.sigma,
            self.snr_handler.fp + self.snr_handler_lensed.fp,
            self.snr_handler.fc + self.snr_handler_lensed.fc,
        )
        self.rho_coh = self.coherent_func(M_mu_nu, x_mu) ** 0.5

    def write_output(self):
        self.output_data.original_trigger_time_seconds.append(
            self.snr_handler.trigger_time_seconds
        )
        self.output_data.lensed_trigger_time_seconds.append(
            self.snr_handler_lensed.trigger_time_seconds
        )
        self.output_data.time_slide_index.append(self.snr_handler.time_slide_index)
        self.output_data.original_output[0].snr_real.append(
            float(self.snr_handler.snr_at_trigger[0].real)
        )
        self.output_data.original_output[0].snr_imag.append(
            float(self.snr_handler.snr_at_trigger[0].imag)
        )
        self.output_data.original_output[1].snr_real.append(
            float(self.snr_handler.snr_at_trigger[1].real)
        )
        self.output_data.original_output[1].snr_imag.append(
            float(self.snr_handler.snr_at_trigger[1].imag)
        )
        self.output_data.lensed_output[0].snr_real.append(
            float(self.snr_handler_lensed.snr_at_trigger[0].real)
        )
        self.output_data.lensed_output[0].snr_imag.append(
            float(self.snr_handler_lensed.snr_at_trigger[0].imag)
        )
        self.output_data.lensed_output[1].snr_real.append(
            float(self.snr_handler_lensed.snr_at_trigger[1].real)
        )
        self.output_data.lensed_output[1].snr_imag.append(
            float(self.snr_handler_lensed.snr_at_trigger[1].imag)
        )
        self.output_data.rho_coinc.append(float(self.rho_coinc[0]))
        self.output_data.rho_coh.append(float(self.rho_coh))
        self.output_data.ra.append(self.iterator_handler.ra)
        self.output_data.dec.append(self.iterator_handler.dec)

    def cluster_output(self):
        unique_lensed_times_indexes = np.concatenate(
            (
                np.unique(
                    self.output_data.lensed_trigger_time_seconds, return_index=True
                )[1],
                [
                    len(self.output_data.lensed_trigger_time_seconds)
                ],  # add the last index too
            )
        )
        # for each lensed time, get the maximum of rho_coh
        max_rho_indexes = []
        for i, start in enumerate(unique_lensed_times_indexes[:-1]):
            end = unique_lensed_times_indexes[i + 1]
            max_rho_indexes.append(
                np.argmax(self.output_data.rho_coh[start:end]) + start
            )
        # as was used in the multi_inspiral code
        cluster_window = 0.1
        clustered_indexes = np.arange(len(max_rho_indexes))[
            findchirp_cluster_over_window(
                np.arange(
                    len(max_rho_indexes)
                ),  # FIXME one should only consider the triggers that are over some threshold
                np.array(self.output_data.rho_coh)[max_rho_indexes],
                cluster_window * 4096,
            )
        ]
        self.output_data.original_trigger_time_seconds = np.array(
            self.output_data.original_trigger_time_seconds
        )[max_rho_indexes][clustered_indexes]
        self.output_data.lensed_trigger_time_seconds = np.array(
            self.output_data.lensed_trigger_time_seconds
        )[max_rho_indexes][clustered_indexes]
        self.output_data.time_slide_index = np.array(self.output_data.time_slide_index)[
            max_rho_indexes
        ][clustered_indexes]
        self.output_data.original_output[0].snr_real = np.array(
            self.output_data.original_output[0].snr_real
        )[max_rho_indexes][clustered_indexes]
        self.output_data.original_output[0].snr_imag = np.array(
            self.output_data.original_output[0].snr_imag
        )[max_rho_indexes][clustered_indexes]
        self.output_data.original_output[1].snr_real = np.array(
            self.output_data.original_output[1].snr_real
        )[max_rho_indexes][clustered_indexes]
        self.output_data.original_output[1].snr_imag = np.array(
            self.output_data.original_output[1].snr_imag
        )[max_rho_indexes][clustered_indexes]
        self.output_data.lensed_output[0].snr_real = np.array(
            self.output_data.lensed_output[0].snr_real
        )[max_rho_indexes][clustered_indexes]
        self.output_data.lensed_output[0].snr_imag = np.array(
            self.output_data.lensed_output[0].snr_imag
        )[max_rho_indexes][clustered_indexes]
        self.output_data.lensed_output[1].snr_real = np.array(
            self.output_data.lensed_output[1].snr_real
        )[max_rho_indexes][clustered_indexes]
        self.output_data.lensed_output[1].snr_imag = np.array(
            self.output_data.lensed_output[1].snr_imag
        )[max_rho_indexes][clustered_indexes]
        self.output_data.rho_coinc = np.array(self.output_data.rho_coinc)[
            max_rho_indexes
        ][clustered_indexes]
        self.output_data.rho_coh = np.array(self.output_data.rho_coh)[max_rho_indexes][
            clustered_indexes
        ]
        self.output_data.ra = np.array(self.output_data.ra)[max_rho_indexes][
            clustered_indexes
        ]
        self.output_data.dec = np.array(self.output_data.dec)[max_rho_indexes][
            clustered_indexes
        ]
