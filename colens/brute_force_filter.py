import logging

import numpy as np
from pycbc.events import ranking

from colens import timing
from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.coherent import coherent_statistic_adapter
from colens.coincident import coincident_snr, get_coinc_indexes
from colens.detector import calculate_antenna_pattern
from colens.io import Output
from colens.sky import SkyGrid


def brute_force_filter_template(
    lensed_detectors,
    unlensed_detectors,
    segments,
    instruments,
    num_slides,
    SLIDE_SHIFT_SECONDS,
    SAMPLE_RATE,
    GPS_START_SECONDS,
    coherent_func,
    output_data: Output,
    get_snr,
    sigma,
    snr_dict,
    segment_index,
    timing_iterator,
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
        theta,
        phi,
    ) in timing_iterator:
        sky_grid = SkyGrid([phi], [theta])
        unlensed_antenna_pattern = calculate_antenna_pattern(
            unlensed_detectors,
            sky_grid,
            original_trigger_time_seconds,
        )
        unlensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            original_trigger_time_seconds,
            sky_grid,
            unlensed_detectors,
        )
        unlensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            unlensed_time_delay_zerolag_seconds,
            time_slides_seconds,
        )
        lensed_antenna_pattern = calculate_antenna_pattern(
            lensed_detectors,
            sky_grid,
            lensed_trigger_time_seconds,
        )
        lensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            lensed_trigger_time_seconds,
            sky_grid,
            lensed_detectors,
        )
        lensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            lensed_time_delay_zerolag_seconds,
            time_slides_seconds,
        )

        # Loop over (short) time-slides, staring with the zero-lag
        for time_slide_index in range(num_slides):
            # loop over sky positions
            for sky_position_index, sky_position in enumerate(sky_grid):
                snr_H1_at_trigger_original = get_snr(
                    time_delay_zerolag_seconds=unlensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ]["H1"],
                    timeseries=snr_dict["H1"],
                    trigger_time_seconds=original_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS["H1"],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=unlensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ]["H1"],
                    cumulative_index=segments["H1"][segment_index].cumulative_index,
                    time_slides_seconds=time_slides_seconds["H1"][time_slide_index],
                    margin=10,
                )
                snr_L1_at_trigger_original = get_snr(
                    time_delay_zerolag_seconds=unlensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ]["L1"],
                    timeseries=snr_dict["L1"],
                    trigger_time_seconds=original_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS["L1"],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=unlensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ]["L1"],
                    cumulative_index=segments["L1"][segment_index].cumulative_index,
                    time_slides_seconds=time_slides_seconds["L1"][time_slide_index],
                    margin=10,
                )
                snr_H1_at_trigger_lensed = get_snr(
                    time_delay_zerolag_seconds=lensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ]["H1_lensed"],
                    timeseries=snr_dict["H1_lensed"],
                    trigger_time_seconds=lensed_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS["H1_lensed"],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=lensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ]["H1_lensed"],
                    cumulative_index=segments["H1_lensed"][
                        segment_index
                    ].cumulative_index,
                    time_slides_seconds=time_slides_seconds["H1_lensed"][
                        time_slide_index
                    ],
                    margin=10,
                )
                snr_L1_at_trigger_lensed = get_snr(
                    time_delay_zerolag_seconds=lensed_time_delay_zerolag_seconds[
                        sky_position_index
                    ]["L1_lensed"],
                    timeseries=snr_dict["L1_lensed"],
                    trigger_time_seconds=lensed_trigger_time_seconds,
                    gps_start_seconds=GPS_START_SECONDS["L1_lensed"],
                    sample_rate=SAMPLE_RATE,
                    time_delay_idx=lensed_time_delay_idx[time_slide_index][
                        sky_position_index
                    ]["L1_lensed"],
                    cumulative_index=segments["L1_lensed"][
                        segment_index
                    ].cumulative_index,
                    time_slides_seconds=time_slides_seconds["L1_lensed"][
                        time_slide_index
                    ],
                    margin=10,
                )

                fp = {
                    ifo: unlensed_antenna_pattern[ifo][sky_position_index][0]
                    for ifo in unlensed_detectors
                }
                fc = {
                    ifo: unlensed_antenna_pattern[ifo][sky_position_index][1]
                    for ifo in unlensed_detectors
                }
                fp.update(
                    {
                        ifo: lensed_antenna_pattern[ifo][sky_position_index][0]
                        for ifo in lensed_detectors
                    }
                )
                fc.update(
                    {
                        ifo: lensed_antenna_pattern[ifo][sky_position_index][1]
                        for ifo in lensed_detectors
                    }
                )

                rho_coinc = coincident_snr(
                    snr_H1_at_trigger_original,
                    snr_L1_at_trigger_original,
                    snr_H1_at_trigger_lensed,
                    snr_L1_at_trigger_lensed,
                )

                M_mu_nu, x_mu = coherent_statistic_adapter(
                    snr_H1_at_trigger_original,
                    snr_L1_at_trigger_original,
                    snr_H1_at_trigger_lensed,
                    snr_L1_at_trigger_lensed,
                    sigma,
                    fp,
                    fc,
                    instruments,
                )
                rho_coh = (
                    coherent_func(
                        M_mu_nu,
                        x_mu,
                    )
                    ** 0.5
                )

                # writting output
                output_data.original_trigger_time_seconds.append(
                    original_trigger_time_seconds
                )
                output_data.lensed_trigger_time_seconds.append(
                    lensed_trigger_time_seconds
                )
                output_data.time_slide_index.append(time_slide_index)
                output_data.ra.append(sky_position.ra)
                output_data.dec.append(sky_position.dec)
                output_data.H1.snr_real.append(float(snr_H1_at_trigger_original.real))
                output_data.H1.snr_imag.append(float(snr_H1_at_trigger_original.imag))
                output_data.L1.snr_real.append(float(snr_L1_at_trigger_original.real))
                output_data.L1.snr_imag.append(float(snr_L1_at_trigger_original.imag))
                output_data.H1_lensed.snr_real.append(
                    float(snr_H1_at_trigger_lensed.real)
                )
                output_data.H1_lensed.snr_imag.append(
                    float(snr_H1_at_trigger_lensed.imag)
                )
                output_data.L1_lensed.snr_real.append(
                    float(snr_L1_at_trigger_lensed.real)
                )
                output_data.L1_lensed.snr_imag.append(
                    float(snr_L1_at_trigger_lensed.imag)
                )
                output_data.rho_coinc.append(float(rho_coinc[0]))
                output_data.rho_coh.append(float(rho_coh))
