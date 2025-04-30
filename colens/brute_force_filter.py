import logging

import numpy as np

from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr, get_coinc_indexes
from colens.io import Output


def brute_force_filter_template(
    coherent_func,
    output_data: Output,
    get_snr,
    data_loader,
):

    for (
        original_trigger_time_seconds,
        lensed_trigger_time_seconds,
        ra,
        dec,
    ) in data_loader.timing_iterator:
        sky_position_index = 0
        data_loader.calculate_antenna_pattern(
            ra,
            dec,
            original_trigger_time_seconds,
            lensed_trigger_time_seconds,
        )
        data_loader.get_time_delay_at_zerolag_seconds(
            original_trigger_time_seconds,
            lensed_trigger_time_seconds,
            ra,
            dec,
        )
        data_loader.get_time_delay_indices()

        # Loop over (short) time-slides, staring with the zero-lag
        for time_slide_index in range(data_loader.num_slides):
            data_loader.get_snr_at_trigger(
                get_snr,
                sky_position_index,
                original_trigger_time_seconds,
                lensed_trigger_time_seconds,
                time_slide_index,
            )

            fp = [
                data_loader.unlensed_antenna_pattern[ifo][sky_position_index][0]
                for ifo in data_loader.unlensed_detectors
            ]
            fc = [
                data_loader.unlensed_antenna_pattern[ifo][sky_position_index][1]
                for ifo in data_loader.unlensed_detectors
            ]
            fp += [
                data_loader.lensed_antenna_pattern[ifo][sky_position_index][0]
                for ifo in data_loader.lensed_detectors
            ]
            fc += [
                data_loader.lensed_antenna_pattern[ifo][sky_position_index][1]
                for ifo in data_loader.lensed_detectors
            ]

            rho_coinc = coincident_snr(data_loader.snr_at_trigger)

            M_mu_nu, x_mu = coherent_statistic_adapter(
                data_loader.snr_at_trigger, data_loader.sigma, fp, fc
            )
            rho_coh = coherent_func(M_mu_nu, x_mu) ** 0.5

            # writting output
            output_data.original_trigger_time_seconds.append(
                original_trigger_time_seconds
            )
            output_data.lensed_trigger_time_seconds.append(lensed_trigger_time_seconds)
            output_data.time_slide_index.append(time_slide_index)
            output_data.ra.append(ra)
            output_data.dec.append(dec)
            output_data.H1.snr_real.append(
                float(data_loader.snr_at_trigger_original[0].real)
            )
            output_data.H1.snr_imag.append(
                float(data_loader.snr_at_trigger_original[0].imag)
            )
            output_data.L1.snr_real.append(
                float(data_loader.snr_at_trigger_original[1].real)
            )
            output_data.L1.snr_imag.append(
                float(data_loader.snr_at_trigger_original[1].imag)
            )
            output_data.H1_lensed.snr_real.append(
                float(data_loader.snr_at_trigger_lensed[0].real)
            )
            output_data.H1_lensed.snr_imag.append(
                float(data_loader.snr_at_trigger_lensed[0].imag)
            )
            output_data.L1_lensed.snr_real.append(
                float(data_loader.snr_at_trigger_lensed[1].real)
            )
            output_data.L1_lensed.snr_imag.append(
                float(data_loader.snr_at_trigger_lensed[1].imag)
            )
            output_data.rho_coinc.append(float(rho_coinc[0]))
            output_data.rho_coh.append(float(rho_coh))
