import logging

import numpy as np
from pycbc.events import ranking

from colens import timing
from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.coincident import coincident_snr, get_coinc_indexes
from colens.detector import calculate_antenna_pattern
from colens.filter import filter_ifos
from colens.interpolate import get_snr_interpolated, interpolate_timeseries_at
from colens.io import Output
from colens.sky import SkyGrid


def brute_force_filter_template(
    lensed_detectors,
    unlensed_detectors,
    segments,
    instruments,
    template,
    matched_filter,
    num_slides,
    coinc_threshold,
    null_min,
    null_grad,
    null_step,
    power_chisq,
    chisq_index,
    chisq_nhigh,
    cluster_window,
    SLIDE_SHIFT_SECONDS,
    SAMPLE_RATE,
    GPS_START_SECONDS,
    TIME_GPS_PAST_SECONDS,
    TIME_GPS_FUTURE_SECONDS,
    coherent_func,
    output_data: Output,
    is_time_precise=False,
):
    # TODO loop over segments (or maybe we just create a big segment)
    # get the single detector snrs
    segment_index = 0
    stilde = {ifo: segments[ifo][segment_index] for ifo in instruments}
    snr_dict, norm_dict, corr_dict, idx, snr = filter_ifos(
        instruments, template, matched_filter, segment_index, stilde
    )

    sigmasq = {
        ifo: template.sigmasq(segments[ifo][segment_index].psd) for ifo in instruments
    }
    sigma = {ifo: np.sqrt(sigmasq[ifo]) for ifo in instruments}
    for ifo in instruments:
        output_data.__getattribute__(ifo).sigma.append(sigma[ifo])
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
    ) in timing.get_timing_iterator(
        TIME_GPS_PAST_SECONDS, TIME_GPS_FUTURE_SECONDS, snr_dict["H1"]._delta_t
    ):
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
                index_trigger_H1_original = int(
                    (original_trigger_time_seconds - GPS_START_SECONDS["H1"])
                    * SAMPLE_RATE
                    + unlensed_time_delay_idx[time_slide_index][sky_position_index][
                        "H1"
                    ]
                    - segments["H1"][segment_index].cumulative_index
                )

                index_trigger_L1_original = int(
                    (original_trigger_time_seconds - GPS_START_SECONDS["L1"])
                    * SAMPLE_RATE
                    + unlensed_time_delay_idx[time_slide_index][sky_position_index][
                        "L1"
                    ]
                    - segments["L1"][segment_index].cumulative_index
                )
                index_trigger_H1_lensed = int(
                    (lensed_trigger_time_seconds - GPS_START_SECONDS["H1_lensed"])
                    * SAMPLE_RATE
                    + lensed_time_delay_idx[time_slide_index][sky_position_index][
                        "H1_lensed"
                    ]
                    - segments["H1_lensed"][segment_index].cumulative_index
                )
                index_trigger_L1_lensed = int(
                    (lensed_trigger_time_seconds - GPS_START_SECONDS["L1_lensed"])
                    * SAMPLE_RATE
                    + lensed_time_delay_idx[time_slide_index][sky_position_index][
                        "L1_lensed"
                    ]
                    - segments["L1_lensed"][segment_index].cumulative_index
                )

                if not is_time_precise:
                    snr_H1_at_trigger_original = snr_dict["H1"][
                        index_trigger_H1_original
                    ]
                    snr_L1_at_trigger_original = snr_dict["L1"][
                        index_trigger_L1_original
                    ]
                    snr_H1_at_trigger_lensed = snr_dict["H1_lensed"][
                        index_trigger_H1_lensed
                    ]
                    snr_L1_at_trigger_lensed = snr_dict["L1_lensed"][
                        index_trigger_L1_lensed
                    ]
                else:
                    snr_H1_at_trigger_original = interpolate_timeseries_at(
                        time=original_trigger_time_seconds
                        + unlensed_time_delay_zerolag_seconds[sky_position_index]["H1"]
                        + time_slides_seconds["H1"][time_slide_index]
                        - GPS_START_SECONDS["H1"],
                        timeseries=snr_dict["H1"],
                        index=index_trigger_H1_original,
                        margin=10,
                    )
                    snr_H1_at_trigger_original = get_snr_interpolated(
                        unlensed_time_delay_zerolag_seconds=unlensed_time_delay_zerolag_seconds[
                            sky_position_index
                        ][
                            "H1"
                        ],
                        timeseries=snr_dict["H1"],
                        original_trigger_time_seconds=original_trigger_time_seconds,
                        gps_start_seconds=GPS_START_SECONDS["H1"],
                        sample_rate=SAMPLE_RATE,
                        unlensed_time_delay_idx=unlensed_time_delay_idx[
                            time_slide_index
                        ][sky_position_index]["H1"],
                        cumulative_index=segments["H1"][segment_index].cumulative_index,
                        time_slides_seconds=time_slides_seconds["H1"][time_slide_index],
                        margin=10,
                    )
                    snr_L1_at_trigger_original = interpolate_timeseries_at(
                        time=original_trigger_time_seconds
                        + unlensed_time_delay_zerolag_seconds[sky_position_index]["L1"]
                        + time_slides_seconds["L1"][time_slide_index]
                        - GPS_START_SECONDS["L1"],
                        timeseries=snr_dict["L1"],
                        index=index_trigger_L1_original,
                        margin=10,
                    )
                    snr_H1_at_trigger_lensed = interpolate_timeseries_at(
                        time=lensed_trigger_time_seconds
                        + lensed_time_delay_zerolag_seconds[sky_position_index][
                            "H1_lensed"
                        ]
                        + time_slides_seconds["H1_lensed"][time_slide_index]
                        - GPS_START_SECONDS["H1_lensed"],
                        timeseries=snr_dict["H1_lensed"],
                        index=index_trigger_H1_lensed,
                        margin=10,
                    )
                    snr_L1_at_trigger_lensed = interpolate_timeseries_at(
                        time=lensed_trigger_time_seconds
                        + lensed_time_delay_zerolag_seconds[sky_position_index][
                            "L1_lensed"
                        ]
                        + time_slides_seconds["L1_lensed"][time_slide_index]
                        - GPS_START_SECONDS["L1_lensed"],
                        timeseries=snr_dict["L1_lensed"],
                        index=index_trigger_L1_lensed,
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

                rho_coh = (
                    coherent_func(
                        snr_H1_at_trigger_original,
                        snr_L1_at_trigger_original,
                        snr_H1_at_trigger_lensed,
                        snr_L1_at_trigger_lensed,
                        sigma,
                        fp,
                        fc,
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
