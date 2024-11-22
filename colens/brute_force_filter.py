import logging

import numpy as np
from pycbc.events import coherent as coh
from pycbc.events import ranking

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.coincident import get_coinc_indexes
from colens.detector import calculate_antenna_pattern
from colens.filter import filter_ifos


def coherent_snr(
    snr_H1_at_trigger,
    snr_L1_at_trigger,
    snr_triggers,
    index,
    threshold,
    projection_matrix,
    coinc_snr=None,
):
    # Calculate rho_coh
    snr_array = np.array([snr_triggers[ifo] for ifo in sorted(snr_triggers.keys())])
    snr_array = np.vstack(
        [
            np.ones(snr_array.shape[1]) * snr_H1_at_trigger,
            snr_array[0],
            np.ones(snr_array.shape[1]) * snr_L1_at_trigger,
            snr_array[1],
        ]
    )
    snr_proj = np.inner(snr_array.conj().transpose(), projection_matrix)
    rho_coh2 = sum(snr_proj.transpose() * snr_array)
    rho_coh = abs(np.sqrt(rho_coh2))
    # Apply thresholds
    above = rho_coh > threshold
    index = index[above]
    coinc_snr = [] if coinc_snr is None else coinc_snr
    if len(coinc_snr) != 0:
        coinc_snr = coinc_snr[above]
    snrv = {ifo: snr_triggers[ifo][above] for ifo in snr_triggers.keys()}
    rho_coh = rho_coh[above]
    return rho_coh, index, snrv, coinc_snr


def get_coinc_triggers(snrs, idx, t_delay_idx):
    # loops through snrs
    # %len(snrs[ifo]) was included as part of a wrap-around solution
    coincs = {
        ifo: snrs[ifo][(idx + t_delay_idx[ifo]) % len(snrs[ifo])]
        for ifo in ["H1_lensed", "L1_lensed"]
    }
    return coincs


def coincident_snr(
    snr_H1_at_trigger, snr_L1_at_trigger, snr_dict, index, threshold, time_delay_idx
):
    # Restrict the snr timeseries to just the interesting points
    coinc_triggers = get_coinc_triggers(snr_dict, index, time_delay_idx)
    # Calculate the coincident snr
    snr_array = np.array([coinc_triggers[ifo] for ifo in coinc_triggers.keys()])
    snr_array = np.vstack(
        [
            snr_array,
            np.ones(snr_array.shape[1]) * snr_H1_at_trigger,
            np.ones(snr_array.shape[1]) * snr_L1_at_trigger,
        ]
    )
    rho_coinc = abs(np.sqrt(np.sum(snr_array * snr_array.conj(), axis=0)))
    # Apply threshold
    thresh_indexes = rho_coinc > threshold
    index = index[thresh_indexes]
    coinc_triggers = get_coinc_triggers(snr_dict, index, time_delay_idx)
    rho_coinc = rho_coinc[thresh_indexes]
    return rho_coinc, index, coinc_triggers


def brute_force_filter_template(
    segments,
    instruments,
    template,
    event_mgr,
    matched_filter,
    num_slides,
    lensed_instruments,
    coinc_threshold,
    do_null_cut,
    null_min,
    null_grad,
    null_step,
    power_chisq,
    chisq_index,
    chisq_nhigh,
    ifo_names,
    sky_grid,
    cluster_window,
    sample_rate,
    network_names,
    SLIDE_SHIFT_SECONDS,
    UNLENSED_INSTRUMENTS,
    LENSED_INSTRUMENTS,
    SAMPLE_RATE,
    GPS_START_SECONDS,
    TIME_GPS_PAST_SECONDS,
    TIME_GPS_FUTURE_SECONDS,
):
    # TODO loop over segments
    # get the single detector snrs
    segment_index = 0
    stilde = {ifo: segments[ifo][segment_index] for ifo in instruments}
    snr_dict, norm_dict, corr_dict, idx, snr = filter_ifos(
        instruments, template, matched_filter, segment_index, stilde
    )

    # loop over original geocentric trigger time
    for original_trigger_time_seconds in np.arange(
        TIME_GPS_PAST_SECONDS - 0.001,
        TIME_GPS_PAST_SECONDS + 0.001,
        step=snr_dict["H1"]._delta_t,
    ):
        # loop over lensed geocentric trigger time
        for lensed_trigger_time_seconds in np.arange(
            TIME_GPS_FUTURE_SECONDS - 0.001,
            TIME_GPS_FUTURE_SECONDS + 0.001,
            snr_dict["H1"]._delta_t,
        ):
            trigger_times_seconds = {
                "H1": original_trigger_time_seconds,
                "L1": original_trigger_time_seconds,
                "H1_lensed": lensed_trigger_time_seconds,
                "L1_lensed": lensed_trigger_time_seconds,
            }
            time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
                trigger_times_seconds,
                sky_grid,
                instruments,
            )
            time_slides_seconds = get_time_slides_seconds(
                num_slides,
                SLIDE_SHIFT_SECONDS,
                UNLENSED_INSTRUMENTS,
                LENSED_INSTRUMENTS,
            )
            time_delay_idx = get_time_delay_indices(
                SAMPLE_RATE,
                time_delay_zerolag_seconds,
                time_slides_seconds,
            )

            # loop over sky positions
            for sky_position_index, sky_position in enumerate(sky_grid):
                index_trigger_H1 = int(
                    (
                        original_trigger_time_seconds
                        + time_delay_zerolag_seconds[sky_position_index]["H1"]
                        - GPS_START_SECONDS["H1"]
                    )
                    * SAMPLE_RATE
                    - segments["H1"][0].cumulative_index
                )

                index_trigger_L1 = int(
                    (
                        original_trigger_time_seconds
                        + time_delay_zerolag_seconds[sky_position_index]["L1"]
                        - GPS_START_SECONDS["L1"]
                    )
                    * SAMPLE_RATE
                    - segments["L1"][0].cumulative_index
                )

                snr_H1_at_trigger = snr_dict["H1"][index_trigger_H1]
                snr_L1_at_trigger = snr_dict["L1"][index_trigger_L1]

                logging.info(
                    f"The coincident snr is {(abs(snr_H1_at_trigger) ** 2 + abs(snr_L1_at_trigger) ** 2) ** 0.5}"
                )

                sigmasq = {
                    ifo: template.sigmasq(segments[ifo][segment_index].psd)
                    for ifo in instruments
                }
                sigma = {ifo: np.sqrt(sigmasq[ifo]) for ifo in instruments}

                antenna_pattern = calculate_antenna_pattern(
                    sky_grid, trigger_times_seconds
                )

                fp = {
                    ifo: antenna_pattern[ifo][sky_position_index][0]
                    for ifo in instruments
                }
                fc = {
                    ifo: antenna_pattern[ifo][sky_position_index][1]
                    for ifo in instruments
                }
                project = coh.get_projection_matrix(
                    fp, fc, sigma, projection="standard"
                )

                # Loop over (short) time-slides, staring with the zero-lag
                for time_slide_index in range(num_slides):

                    # Adjust the indices of triggers (if there are any)
                    # and store trigger indices list in a dictionary;
                    # when there are no triggers, the dictionary is empty.
                    # Indices are kept only if they do not get time shifted
                    # out of the time we are looking at, i.e., require
                    # idx[ifo] - time_delay_idx[slide][position_index][ifo]
                    # to be in (0, len(snr_dict[ifo]))
                    idx_dict = {
                        ifo: idx[ifo][
                            np.logical_and(
                                idx[ifo]
                                > time_delay_idx[time_slide_index][sky_position_index][
                                    ifo
                                ],
                                idx[ifo]
                                - time_delay_idx[time_slide_index][sky_position_index][
                                    ifo
                                ]
                                < len(snr_dict[ifo]),
                            )
                        ]
                        for ifo in instruments
                    }

                    # Find triggers that are coincident (in geocent time) in
                    # multiple IFOs. If a single IFO analysis then just use the
                    # indices from that IFO, i.e., IFO 0; otherwise, this
                    # method finds coincidences and applies the single IFO cut,
                    # namely, triggers must have at least 2 IFO SNRs above
                    # args.sngl_snr_threshold.
                    coinc_idx = get_coinc_indexes(
                        idx_dict,
                        time_delay_idx[time_slide_index][sky_position_index],
                        LENSED_INSTRUMENTS,
                    )

                    coinc_idx_detector_frame = {
                        ifo: coinc_idx
                        + time_delay_idx[time_slide_index][sky_position_index][ifo]
                        for ifo in LENSED_INSTRUMENTS
                    }

                    (
                        rho_coinc,
                        coinc_idx,
                        coinc_triggers,
                    ) = coincident_snr(
                        snr_H1_at_trigger,
                        snr_L1_at_trigger,
                        snr_dict,
                        coinc_idx,
                        coinc_threshold,
                        time_delay_idx[time_slide_index][sky_position_index],
                    )
                    logging.info(rho_coinc)

                    (
                        rho_coh,
                        coinc_idx,
                        coinc_triggers,
                        rho_coinc,
                    ) = coherent_snr(
                        snr_H1_at_trigger,
                        snr_L1_at_trigger,
                        coinc_triggers,
                        coinc_idx,
                        0,
                        project,
                        rho_coinc,
                    )
                    logging.info(rho_coh)
                    (
                        null,
                        rho_coh,
                        rho_coinc,
                        coinc_idx,
                        coinc_triggers,
                    ) = coh.null_snr(
                        rho_coh,
                        rho_coinc,
                        apply_cut=do_null_cut,
                        null_min=null_min,
                        null_grad=null_grad,
                        null_step=null_step,
                        snrv=coinc_triggers,
                        index=coinc_idx,
                    )
                    logging.info(null)

                    found_trigger_time_geocenter = (
                        coinc_idx[rho_coinc.argmax()]
                        + segments["H1_lensed"][segment_index].cumulative_index
                    ) / SAMPLE_RATE + GPS_START_SECONDS["H1_lensed"]
                    logging.info(found_trigger_time_geocenter)

                    # consistency tests
                    coherent_ifo_trigs = {
                        ifo: snr_dict[ifo][coinc_idx_detector_frame[ifo]]
                        for ifo in LENSED_INSTRUMENTS
                    }

                    # Calculate the powerchi2 values of remaining triggers
                    # (this uses the SNR timeseries before the time delay,
                    # so we undo it; the same holds for normalisation)
                    chisq = {}
                    chisq_dof = {}
                    for ifo in LENSED_INSTRUMENTS:
                        chisq[ifo], chisq_dof[ifo] = power_chisq.values(
                            corr_dict[ifo],
                            coherent_ifo_trigs[ifo] / norm_dict[ifo],
                            norm_dict[ifo],
                            stilde[ifo].psd,
                            coinc_idx_detector_frame[ifo],
                            template,
                        )
                    logging.info(chisq)
                    network_chisq_values = coh.network_chisq(
                        chisq, chisq_dof, coherent_ifo_trigs
                    )
                    logging.info(network_chisq_values)

                    reweighted_snr = ranking.newsnr(
                        rho_coh,
                        network_chisq_values,
                        q=chisq_index,
                        n=chisq_nhigh,
                    )
                    logging.info(reweighted_snr)

                    reweighted_by_null_snr = coh.reweight_snr_by_null(
                        reweighted_snr,
                        null,
                        rho_coh,
                        null_min=null_min,
                        null_grad=null_grad,
                        null_step=null_step,
                    )
                    logging.info(reweighted_by_null_snr)
