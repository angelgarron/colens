from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr


class Runner:
    def __init__(
        self,
        coherent_func,
        output_data,
        snr_handler,
        snr_handler_lensed,
        timing_iterator,
    ):
        self.output_data = output_data
        self.snr_handler = snr_handler
        self.snr_handler_lensed = snr_handler_lensed
        self.coherent_func = coherent_func
        self.timing_iterator = timing_iterator

    def run(self):
        for _ in self.timing_iterator:
            self._run_single()
            self.write_output()

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
        self.output_data.ra.append(self.snr_handler.ra)
        self.output_data.dec.append(self.snr_handler.dec)
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
