import logging

import numpy as np

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr, get_coinc_indexes
from colens.detector import calculate_antenna_pattern
from colens.io import Output


def brute_force_filter_template(
    lensed_detectors,
    unlensed_detectors,
    num_slides,
    SLIDE_SHIFT_SECONDS,
    SAMPLE_RATE,
    GPS_START_SECONDS,
    coherent_func,
    output_data: Output,
    get_snr,
    data_loader,
):
    time_slides_seconds = get_time_slides_seconds(
        num_slides,
        SLIDE_SHIFT_SECONDS,
        list(unlensed_detectors),
        list(lensed_detectors),
    )

    for (
        original_trigger_time_seconds,
        lensed_trigger_time_seconds,
        ra,
        dec,
    ) in data_loader.timing_iterator:
        sky_position_index = 0
        unlensed_antenna_pattern = calculate_antenna_pattern(
            unlensed_detectors,
            ra,
            dec,
            original_trigger_time_seconds,
        )
        unlensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            original_trigger_time_seconds,
            ra,
            dec,
            unlensed_detectors,
        )
        unlensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            unlensed_time_delay_zerolag_seconds,
            time_slides_seconds,
        )
        lensed_antenna_pattern = calculate_antenna_pattern(
            lensed_detectors,
            ra,
            dec,
            lensed_trigger_time_seconds,
        )
        lensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            lensed_trigger_time_seconds,
            ra,
            dec,
            lensed_detectors,
        )
        lensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            lensed_time_delay_zerolag_seconds,
            time_slides_seconds,
        )

        # Loop over (short) time-slides, staring with the zero-lag
        for time_slide_index in range(num_slides):
            snr_at_trigger_original = [
                get_snr(
                    time_delay_zerolag_seconds=unlensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ][ifo],
                    timeseries=data_loader.snr_dict[ifo],
                    trigger_time_seconds=original_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS[ifo],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=unlensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ][ifo],
                    cumulative_index=data_loader.segments[ifo][
                        data_loader.segment_index
                    ].cumulative_index,
                    time_slides_seconds=time_slides_seconds[ifo][time_slide_index],
                )
                for ifo in unlensed_detectors
            ]
            snr_at_trigger_lensed = [
                get_snr(
                    time_delay_zerolag_seconds=lensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ][ifo],
                    timeseries=data_loader.snr_dict[ifo],
                    trigger_time_seconds=lensed_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS[ifo],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=lensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ][ifo],
                    cumulative_index=data_loader.segments[ifo][
                        data_loader.segment_index
                    ].cumulative_index,
                    time_slides_seconds=time_slides_seconds[ifo][time_slide_index],
                )
                for ifo in lensed_detectors
            ]

            snr_at_trigger = snr_at_trigger_original + snr_at_trigger_lensed

            fp = [
                unlensed_antenna_pattern[ifo][sky_position_index][0]
                for ifo in unlensed_detectors
            ]
            fc = [
                unlensed_antenna_pattern[ifo][sky_position_index][1]
                for ifo in unlensed_detectors
            ]
            fp += [
                lensed_antenna_pattern[ifo][sky_position_index][0]
                for ifo in lensed_detectors
            ]
            fc += [
                lensed_antenna_pattern[ifo][sky_position_index][1]
                for ifo in lensed_detectors
            ]

            rho_coinc = coincident_snr(snr_at_trigger)

            M_mu_nu, x_mu = coherent_statistic_adapter(
                snr_at_trigger, data_loader.sigma, fp, fc
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
            output_data.H1.snr_real.append(float(snr_at_trigger_original[0].real))
            output_data.H1.snr_imag.append(float(snr_at_trigger_original[0].imag))
            output_data.L1.snr_real.append(float(snr_at_trigger_original[1].real))
            output_data.L1.snr_imag.append(float(snr_at_trigger_original[1].imag))
            output_data.H1_lensed.snr_real.append(float(snr_at_trigger_lensed[0].real))
            output_data.H1_lensed.snr_imag.append(float(snr_at_trigger_lensed[0].imag))
            output_data.L1_lensed.snr_real.append(float(snr_at_trigger_lensed[1].real))
            output_data.L1_lensed.snr_imag.append(float(snr_at_trigger_lensed[1].imag))
            output_data.rho_coinc.append(float(rho_coinc[0]))
            output_data.rho_coh.append(float(rho_coh))
