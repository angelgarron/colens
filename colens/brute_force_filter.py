import logging

import numpy as np
from pycbc.events import coherent as coh
from pycbc.events import ranking

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.coherent import coherent_snr
from colens.coincident import coincident_snr, get_coinc_indexes
from colens.detector import calculate_antenna_pattern
from colens.filter import filter_ifos
from colens.io import write_to_json
from colens.null import null_snr


def brute_force_filter_template(
    detectors,
    segments,
    instruments,
    template,
    matched_filter,
    num_slides,
    lensed_instruments,
    coinc_threshold,
    null_min,
    null_grad,
    null_step,
    power_chisq,
    chisq_index,
    chisq_nhigh,
    sky_grid,
    cluster_window,
    sample_rate,
    SLIDE_SHIFT_SECONDS,
    UNLENSED_INSTRUMENTS,
    LENSED_INSTRUMENTS,
    SAMPLE_RATE,
    GPS_START_SECONDS,
    TIME_GPS_PAST_SECONDS,
    TIME_GPS_FUTURE_SECONDS,
):
    output_data = {
        "original_trigger_time_seconds": [],
        "lensed_trigger_time_seconds": [],
        "time_slide_index": [],
        "sky_position": {
            "ra": [],
            "dec": [],
        },
        "H1": {
            "snr_real": [],
            "snr_imag": [],
            "sigma": [],
            "chisq_dof": [],
            "chisq": [],
        },
        "L1": {
            "snr_real": [],
            "snr_imag": [],
            "sigma": [],
            "chisq_dof": [],
            "chisq": [],
        },
        "H1_lensed": {
            "snr_real": [],
            "snr_imag": [],
            "sigma": [],
            "chisq_dof": [],
            "chisq": [],
        },
        "L1_lensed": {
            "snr_real": [],
            "snr_imag": [],
            "sigma": [],
            "chisq_dof": [],
            "chisq": [],
        },
        "rho_coinc": [],
        "rho_coh": [],
        "null": [],
        "network_chisq_values": [],
        "reweighted_snr": [],
        "reweighted_by_null_snr": [],
    }
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
                detectors,
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

            antenna_pattern = calculate_antenna_pattern(
                detectors, sky_grid, trigger_times_seconds
            )

            # Loop over (short) time-slides, staring with the zero-lag
            for time_slide_index in range(num_slides):
                # loop over sky positions
                for sky_position_index, sky_position in enumerate(sky_grid):
                    index_trigger_H1_original = int(
                        (original_trigger_time_seconds - GPS_START_SECONDS["H1"])
                        * SAMPLE_RATE
                        + time_delay_idx[time_slide_index][sky_position_index]["H1"]
                        - segments["H1"][segment_index].cumulative_index
                    )

                    index_trigger_L1_original = int(
                        (original_trigger_time_seconds - GPS_START_SECONDS["L1"])
                        * SAMPLE_RATE
                        + time_delay_idx[time_slide_index][sky_position_index]["L1"]
                        - segments["L1"][segment_index].cumulative_index
                    )
                    index_trigger_H1_lensed = int(
                        (lensed_trigger_time_seconds - GPS_START_SECONDS["H1_lensed"])
                        * SAMPLE_RATE
                        + time_delay_idx[time_slide_index][sky_position_index][
                            "H1_lensed"
                        ]
                        - segments["H1_lensed"][segment_index].cumulative_index
                    )
                    index_trigger_L1_lensed = int(
                        (lensed_trigger_time_seconds - GPS_START_SECONDS["L1_lensed"])
                        * SAMPLE_RATE
                        + time_delay_idx[time_slide_index][sky_position_index][
                            "L1_lensed"
                        ]
                        - segments["L1_lensed"][segment_index].cumulative_index
                    )

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

                    logging.info(
                        f"The coincident snr is {(abs(snr_H1_at_trigger_original) ** 2 + abs(snr_L1_at_trigger_original) ** 2) ** 0.5}"
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

                    rho_coinc = coincident_snr(
                        snr_H1_at_trigger_original,
                        snr_L1_at_trigger_original,
                        snr_H1_at_trigger_lensed,
                        snr_L1_at_trigger_lensed,
                    )
                    logging.info(rho_coinc)

                    rho_coh = coherent_snr(
                        snr_H1_at_trigger_original,
                        snr_L1_at_trigger_original,
                        snr_H1_at_trigger_lensed,
                        snr_L1_at_trigger_lensed,
                        project,
                    )
                    logging.info(rho_coh)

                    null = null_snr(
                        rho_coh,
                        rho_coinc,
                    )
                    logging.info(null)

                    # consistency tests
                    coherent_ifo_trigs = {
                        "H1": np.array([snr_H1_at_trigger_original]),
                        "H1_lensed": np.array([snr_H1_at_trigger_lensed]),
                        "L1": np.array([snr_L1_at_trigger_original]),
                        "L1_lensed": np.array([snr_L1_at_trigger_lensed]),
                    }

                    # Calculate the powerchi2 values of remaining triggers
                    # (this uses the SNR timeseries before the time delay,
                    # so we undo it; the same holds for normalisation)
                    chisq = {}
                    chisq_dof = {}
                    for ifo in instruments:
                        chisq[ifo], chisq_dof[ifo] = power_chisq.values(
                            corr_dict[ifo],
                            coherent_ifo_trigs[ifo] / norm_dict[ifo],
                            norm_dict[ifo],
                            stilde[ifo].psd,
                            {
                                "H1": np.array([index_trigger_H1_original]),
                                "H1_lensed": np.array([index_trigger_H1_lensed]),
                                "L1": np.array([index_trigger_L1_original]),
                                "L1_lensed": np.array([index_trigger_L1_lensed]),
                            }[ifo],
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

                    # writting output
                    output_data["original_trigger_time_seconds"].append(
                        original_trigger_time_seconds
                    )
                    output_data["lensed_trigger_time_seconds"].append(
                        lensed_trigger_time_seconds
                    )
                    output_data["time_slide_index"].append(time_slide_index)
                    output_data["sky_position"]["ra"].append(sky_position.ra)
                    output_data["sky_position"]["dec"].append(sky_position.dec)
                    output_data["H1"]["snr_real"].append(
                        float(snr_H1_at_trigger_original.real)
                    )
                    output_data["H1"]["snr_imag"].append(
                        float(snr_H1_at_trigger_original.imag)
                    )
                    output_data["L1"]["snr_real"].append(
                        float(snr_L1_at_trigger_original.real)
                    )
                    output_data["L1"]["snr_imag"].append(
                        float(snr_L1_at_trigger_original.imag)
                    )
                    output_data["H1_lensed"]["snr_real"].append(
                        float(snr_H1_at_trigger_lensed.real)
                    )
                    output_data["H1_lensed"]["snr_imag"].append(
                        float(snr_H1_at_trigger_lensed.imag)
                    )
                    output_data["L1_lensed"]["snr_real"].append(
                        float(snr_L1_at_trigger_lensed.real)
                    )
                    output_data["L1_lensed"]["snr_imag"].append(
                        float(snr_L1_at_trigger_lensed.imag)
                    )
                    for ifo in instruments:
                        output_data[ifo]["sigma"].append(sigma[ifo])
                        output_data[ifo]["chisq_dof"].append(float(chisq_dof[ifo][0]))
                        output_data[ifo]["chisq"].append(float(chisq[ifo][0]))
                    output_data["rho_coinc"].append(float(rho_coinc[0]))
                    output_data["rho_coh"].append(float(rho_coh[0]))
                    output_data["null"].append(float(null[0]))
                    output_data["network_chisq_values"].append(
                        float(network_chisq_values[0])
                    )
                    output_data["reweighted_snr"].append(float(reweighted_snr[0]))
                    output_data["reweighted_by_null_snr"].append(
                        float(reweighted_by_null_snr[0])
                    )

    output_file = "results.json"
    logging.info(f"Saving results to {output_file}")
    write_to_json(output_file, output_data)
