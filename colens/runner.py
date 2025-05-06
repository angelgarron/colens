from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr
from colens.data_loader import DataLoader


class Runner:
    def __init__(self, coherent_func, output_data, data_loader: DataLoader):
        self.output_data = output_data
        self.data_loader = data_loader
        self.coherent_func = coherent_func

    def run(self):
        for _ in self.data_loader.timing_iterator:
            self._run_single()
            self.write_output()

    def _run_single(self):
        self.rho_coinc = coincident_snr(self.data_loader.snr_at_trigger)
        M_mu_nu, x_mu = coherent_statistic_adapter(
            self.data_loader.snr_at_trigger,
            self.data_loader.sigma,
            self.data_loader.fp,
            self.data_loader.fc,
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
