import logging

import numpy as np
from pycbc.events import coherent as coh
from pycbc.events import ranking

from colens.coincident import get_coinc_indexes


def filter_template(
    segments,
    instruments,
    template,
    event_mgr,
    matched_filter,
    num_slides,
    time_delay_idx,
    lensed_instruments,
    coinc_threshold,
    antenna_pattern,
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
):
    nifo = len(instruments)
    # Loop over segments
    for s_num in range(len(segments[instruments[0]])):
        stilde = {ifo: segments[ifo][s_num] for ifo in instruments}
        # Find how loud the template is in each detector, i.e., its
        # unnormalized matched-filter with itself. This quantity is
        # used to normalize matched-filters with the data.
        sigmasq = {
            ifo: template.sigmasq(segments[ifo][s_num].psd) for ifo in instruments
        }
        sigma = {ifo: np.sqrt(sigmasq[ifo]) for ifo in instruments}
        # Every time s_num is zero, run new_template to increment the
        # template index
        if s_num == 0:
            event_mgr.new_template(tmplt=template.params, sigmasq=sigmasq)
        logging.info(
            "Analyzing segment %d/%d", s_num + 1, len(segments[instruments[0]])
        )
        # The following dicts with IFOs as keys are created to store
        # copies of the matched filtering results computed below.
        # - Complex SNR time series
        snr_dict = dict.fromkeys(instruments)
        # - Its normalization
        norm_dict = dict.fromkeys(instruments)
        # - The correlation vector frequency series
        #   It is the FFT of the SNR (so inverse FFT it to get the SNR)
        corr_dict = dict.fromkeys(instruments)
        # - The trigger indices list (idx_dict will be created out of this)
        idx = dict.fromkeys(instruments)
        # - The list of normalized SNR values at the trigger locations
        snr = dict.fromkeys(instruments)
        for ifo in instruments:
            logging.info("  Filtering ifo %s", ifo)
            # The following lines unpack and store copies of the matched
            # filtering results for the current template, segment, and IFO.
            # No clustering happens in the coherent search until the end.
            snr_ts, norm, corr, ind, snrv = matched_filter[
                ifo
            ].matched_filter_and_cluster(
                s_num, template.sigmasq(stilde[ifo].psd), window=0
            )
            snr_dict[ifo] = snr_ts[matched_filter[ifo].segments[s_num].analyze] * norm
            assert len(snr_dict[ifo]) > 0, f"SNR time series for {ifo} is empty"
            norm_dict[ifo] = norm
            corr_dict[ifo] = corr.copy()
            idx[ifo] = ind.copy()
            snr[ifo] = snrv * norm

        # Move onto next segment if there are no triggers.
        n_trigs = [len(snr[ifo]) for ifo in instruments]
        if not any(n_trigs):
            continue

        # Loop over (short) time-slides, staring with the zero-lag
        for slide in range(num_slides):
            logging.info("  Analyzing slide %d/%d", slide, num_slides)
            # Loop over sky positions
            for position_index in range(len(sky_grid)):
                logging.info(
                    "    Analyzing sky position %d/%d",
                    position_index + 1,
                    len(sky_grid),
                )
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
                            idx[ifo] > time_delay_idx[slide][position_index][ifo],
                            idx[ifo] - time_delay_idx[slide][position_index][ifo]
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
                    idx_dict, time_delay_idx[slide][position_index], lensed_instruments
                )
                logging.info("        Found %d coincident triggers", len(coinc_idx))
                # Calculate the coincident and coherent SNR.
                # First check there is enough data to compute the SNRs.
                if len(coinc_idx) != 0 and nifo > 1:
                    # Find coinc SNR at trigger times and apply coinc SNR
                    # threshold (which depopulates coinc_idx accordingly)
                    (
                        rho_coinc,
                        coinc_idx,
                        coinc_triggers,
                    ) = coh.coincident_snr(
                        snr_dict,
                        coinc_idx,
                        coinc_threshold,
                        time_delay_idx[slide][position_index],
                    )
                    logging.info(
                        "        %d triggers above coincident SNR threshold",
                        len(coinc_idx),
                    )
                    if len(coinc_idx) != 0:
                        logging.info(
                            "        With max coincident SNR = %.2f",
                            max(rho_coinc),
                        )
                else:
                    coinc_triggers = {}
                    logging.info("        No coincident triggers were found")
                # If there are triggers above coinc threshold and more
                # than 2 IFOs, then calculate the coherent statistics for
                # them and apply the cut on coherent SNR (with threshold
                # equal to the coinc SNR one)
                if len(coinc_idx) != 0 and nifo > 2:
                    logging.info("      Calculating their coherent statistics")
                    # Plus and cross antenna pattern dictionaries
                    fp = {
                        ifo: antenna_pattern[ifo][position_index][0]
                        for ifo in instruments
                    }
                    fc = {
                        ifo: antenna_pattern[ifo][position_index][1]
                        for ifo in instruments
                    }
                    project = coh.get_projection_matrix(
                        fp, fc, sigma, projection="standard"
                    )
                    (
                        rho_coh,
                        coinc_idx,
                        coinc_triggers,
                        rho_coinc,
                    ) = coh.coherent_snr(
                        coinc_triggers,
                        coinc_idx,
                        coinc_threshold,
                        project,
                        rho_coinc,
                    )
                    logging.info(
                        "        %d triggers above coherent SNR threshold",
                        len(rho_coh),
                    )
                    if len(coinc_idx) != 0:
                        logging.info(
                            "        With max coherent SNR = %.2f", max(rho_coh)
                        )
                        # Calculate the null SNR and apply the null SNR cut
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
                        logging.info(
                            "        %d triggers above null threshold", len(null)
                        )
                        if len(coinc_idx) != 0:
                            logging.info("        With max null SNR = %.2f", max(null))
                            logging.info(
                                f"        The coinc, coh and null at max(coh) are = {rho_coinc[rho_coh.argmax()]}, {rho_coh.max()} and {null[rho_coh.argmax()]}"
                            )
                # Now calculate the individual detector chi2 values
                # and the SNR reweighted by chi2 and by null SNR
                # (no cut on reweighted SNR is applied).
                # To do this it is useful to find the indices of the
                # (surviving) triggers in the detector frame.
                if len(coinc_idx) != 0:
                    # Updated coinc_idx_det_frame to account for the
                    # effect of the cuts applied to far
                    coinc_idx_det_frame = {
                        ifo: (coinc_idx + time_delay_idx[slide][position_index][ifo])
                        % len(snr_dict[ifo])
                        for ifo in instruments
                    }
                    # Build dictionary with per-IFO complex SNR time series
                    # of the most recent set of triggers
                    coherent_ifo_trigs = {
                        ifo: snr_dict[ifo][coinc_idx_det_frame[ifo]]
                        for ifo in instruments
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
                            coinc_idx_det_frame[ifo] + stilde[ifo].analyze.start,
                            template,
                        )
                    network_chisq_dict = coh.network_chisq(
                        chisq, chisq_dof, coherent_ifo_trigs
                    )
                    # Calculate chisq reweighted SNR
                    if nifo > 2:
                        reweighted_snr = ranking.newsnr(
                            rho_coh,
                            network_chisq_dict,
                            q=chisq_index,
                            n=chisq_nhigh,
                        )
                        reweighted_snr = coh.reweight_snr_by_null(
                            reweighted_snr,
                            null,
                            rho_coh,
                            null_min=null_min,
                            null_grad=null_grad,
                            null_step=null_step,
                        )
                    elif nifo == 2:
                        reweighted_snr = ranking.newsnr(
                            rho_coinc,
                            network_chisq_dict,
                            q=chisq_index,
                            n=chisq_nhigh,
                        )
                    else:
                        rho_sngl = abs(
                            snr[instruments[0]][coinc_idx_det_frame[instruments[0]]]
                        )
                        reweighted_snr = ranking.newsnr(
                            rho_sngl,
                            network_chisq_dict,
                            q=chisq_index,
                            n=chisq_nhigh,
                        )
                    # All out vals must be the same length, so single
                    # value entries are repeated once per event
                    num_events = len(reweighted_snr)
                    ifo_out_vals = dict()
                    for ifo in instruments:
                        ifo_out_vals["chisq"] = chisq[ifo]
                        ifo_out_vals["chisq_dof"] = chisq_dof[ifo]
                        ifo_out_vals["time_index"] = (
                            coinc_idx_det_frame[ifo] + stilde[ifo].cumulative_index
                        )
                        ifo_out_vals["snr"] = coherent_ifo_trigs[ifo]
                        # IFO is stored as an int
                        ifo_out_vals["ifo"] = [event_mgr.ifo_dict[ifo]] * num_events
                        ifo_out_vals["slide_id"] = [slide] * num_events
                        event_mgr.add_template_events_to_ifo(
                            ifo,
                            ifo_names,
                            [ifo_out_vals[n] for n in ifo_names],
                        )
                    network_out_vals = dict()
                    if nifo > 2:
                        network_out_vals["coherent_snr"] = rho_coh
                        network_out_vals["null_snr"] = null
                    elif nifo == 2:
                        network_out_vals["coherent_snr"] = rho_coinc
                        network_out_vals["null_snr"] = None
                    else:
                        network_out_vals["coherent_snr"] = abs(
                            snr[instruments[0]][coinc_idx_det_frame[instruments[0]]]
                        )
                        network_out_vals["null_snr"] = None
                    network_out_vals["reweighted_snr"] = reweighted_snr
                    network_out_vals["my_network_chisq"] = np.real(network_chisq_dict)
                    network_out_vals["time_index"] = (
                        coinc_idx + stilde[ifo].cumulative_index
                    )
                    network_out_vals["nifo"] = [nifo] * num_events
                    network_out_vals["dec"] = [
                        sky_grid[position_index].dec
                    ] * num_events
                    network_out_vals["ra"] = [sky_grid[position_index].ra] * num_events
                    network_out_vals["slide_id"] = [slide] * num_events
                    event_mgr.add_template_events_to_network(
                        network_names,
                        [network_out_vals[n] for n in network_names],
                    )
        # The triggers can be clustered
        # Cluster template events by slide
        for slide in range(num_slides):
            logging.info("  Clustering slide %d", slide)
            event_mgr.cluster_template_network_events(
                "time_index",
                "reweighted_snr",
                int(cluster_window * sample_rate),
                slide=slide,
            )
    event_mgr.finalize_template_events()
