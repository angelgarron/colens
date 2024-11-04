import logging

import h5py
import numpy as np
from ligo import segments as ligo_segments
from pesummary.gw import conversions
from pycbc import init_logging, vetoes, waveform
from pycbc.events import EventManagerCoherent
from pycbc.events import coherent as coh
from pycbc.events import ranking
from pycbc.events.eventmgr import H5FileSyntSugar
from pycbc.filter import MatchedFilterControl
from pycbc.psd.estimate import interpolate, welch
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens.coincident import get_coinc_indexes
from colens.detector import MyDetector, calculate_antenna_pattern
from colens.injection import get_strain_list_from_simulation
from colens.sky import sky_grid
from colens.strain import process_strain_dict

FRAME_FILES = {
    "H1": "H-H1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "L1": "L-L1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "H1_lensed": "H-H1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
    "L1_lensed": "L-L1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
}
CHANNELS = {
    "H1": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1": "L1:GWOSC-4KHZ_R1_STRAIN",
    "H1_lensed": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1_lensed": "L1:GWOSC-4KHZ_R1_STRAIN",
}
TIME_GPS_PAST = 1185389807
TIME_GPS_FUTURE = 1185437144
TRIGGER_TIMES = {
    "H1": TIME_GPS_PAST,
    "L1": TIME_GPS_PAST,
    "H1_lensed": TIME_GPS_FUTURE,
    "L1_lensed": TIME_GPS_FUTURE,
}
RA, DEC = 6.0, -1.2
SAMPLE_RATE = 4096.0
UNLENSED_INSTRUMENTS = ["H1", "L1"]
LENSED_INSTRUMENTS = ["H1_lensed", "L1_lensed"]
INSTRUMENTS = UNLENSED_INSTRUMENTS + LENSED_INSTRUMENTS
INSTRUMENTS.sort()
SEGMENT_LENGTH = 256
SLIDE_SHIFT = 1
LOW_FREQUENCY_CUTOFF = 30.0
SNGL_SNR_THRESHOLD = 3.0
DOWNSAMPLE_FACTOR = 1
UPSAMPLE_THRESHOLD = None
UPSAMPLE_METHOD = "pruned_fft"
BANK_FILE = "gw170729_single_template.hdf"
ORDER = "-1"
TAPER_TEMPLATE = None
APPROXIMANT = "IMRPhenomD"
COINC_THRESHOLD = 0.0
DO_NULL_CUT = False
NULL_MIN = 5.25
NULL_GRAD = 0.2
NULL_STEP = 20.0
CLUSTER_WINDOW = 0.1
OUTPUT = "GW170817_test_output.hdf"
PAD = 8
GPS_START = dict()
GPS_END = dict()
for ifo in INSTRUMENTS:
    GPS_START[ifo] = TRIGGER_TIMES[ifo] - 192 - PAD
    GPS_END[ifo] = TRIGGER_TIMES[ifo] + 192 + PAD
START_PAD = 111  # time in seconds to ignore of the beginning of each segment
END_PAD = 17  # time in seconds to ignore at the end of each segment
TRIG_START_TIME = dict()  # gps time to start recording triggers
TRIG_END_TIME = dict()  # gps time to stop recording triggers
for ifo in INSTRUMENTS:
    TRIG_START_TIME[ifo] = GPS_START[ifo] + START_PAD
    TRIG_END_TIME[ifo] = GPS_END[ifo] - END_PAD
CHISQ_BINS = "0.9*get_freq('fSEOBNRv4Peak',params.mass1,params.mass2,params.spin1z,params.spin2z)**(2./3.)"
AUTOCHI_STRIDE = 0
AUTOCHI_NUMBER_POINTS = 0
AUTOCHI_ONESIDED = None
AUTOCHI_TWO_PHASE = False
AUTOCHI_REVERSE_TEMPLATE = False
AUTOCHI_MAX_VALUED = False
AUTOCHI_MAX_VALUED_DOF = None
CHISQ_INDEX = 6.0
CHISQ_NHIGH = 2.0
PSD_SEGMENT_STRIDE = 8.0
PSD_SEGMENT_LENGTH = 32.0
PSD_NUM_SEGMENTS = 29
STRAIN_HIGH_PASS = 25.0
ANGULAR_SPACING = 1.8 * np.pi / 180  # radians
SKY_ERROR = 0.1 * np.pi / 180  # radians


with h5py.File(BANK_FILE, "w") as file:
    for key, value in {
        "appoximant": APPROXIMANT,
        "f_lower": LOW_FREQUENCY_CUTOFF,
        "mass1": 79.45,
        "mass2": 48.50,
        "spin1z": 0.60,
        "spin2z": 0.05,
    }.items():
        file.create_dataset(
            key, data=[value], compression="gzip", compression_opts=9, shuffle=True
        )


def slide_limiter():
    """
    This function computes the number of shortslides used by the coherent
    matched filter statistic to obtain as most background triggers as
    possible.

    It bounds the number of slides to avoid counting triggers more than once.
    If the data is not time slid, there is a single slide for the zero-lag.
    """
    low, upp = 1, SEGMENT_LENGTH
    n_ifos = len(LENSED_INSTRUMENTS)
    stride_dur = SEGMENT_LENGTH / 2
    num_slides = np.int32(1 + np.floor(stride_dur / (SLIDE_SHIFT * (n_ifos - 1))))
    assert np.logical_and(num_slides >= low, num_slides <= upp), (
        "the combination (slideshift, segment_dur)"
        f" = ({SLIDE_SHIFT:.2f},{stride_dur*2:.2f})"
        f" goes over the allowed upper bound {upp}"
    )
    return num_slides


class MyEventManagerCoherent(EventManagerCoherent):
    def write_to_hdf(self, outname):
        self.events.sort(order="template_id")
        th = np.array([p["tmplt"].template_hash for p in self.template_params])
        f = H5FileSyntSugar(outname)
        self.write_gating_info_to_hdf(f)
        # Output network stuff
        f.prefix = "network"
        network_events = np.array(
            [e for e in self.network_events], dtype=self.network_event_dtype
        )
        for col in network_events.dtype.names:
            if col == "time_index":
                f["end_time_gc"] = (
                    network_events[col] / float(SAMPLE_RATE) + GPS_START[self.ifos[0]]
                )
            else:
                f[col] = network_events[col]
        starts = []
        ends = []
        for seg in self.segments[self.ifos[0]]:
            starts.append(seg.start_time.gpsSeconds)
            ends.append(seg.end_time.gpsSeconds)
        f["search/segments/start_times"] = starts
        f["search/segments/end_times"] = ends
        # Individual ifo stuff
        for i, ifo in enumerate(self.ifos):
            tid = self.events["template_id"][self.events["ifo"] == i]
            f.prefix = ifo
            ifo_events = np.array(
                [e for e in self.events if e["ifo"] == self.ifo_dict[ifo]],
                dtype=self.event_dtype,
            )
            if len(ifo_events):
                f["snr"] = abs(ifo_events["snr"])
                f["event_id"] = ifo_events["event_id"]
                try:
                    # Precessing
                    f["u_vals"] = ifo_events["u_vals"]
                    f["coa_phase"] = ifo_events["coa_phase"]
                    f["hplus_cross_corr"] = ifo_events["hplus_cross_corr"]
                except Exception:
                    f["coa_phase"] = np.angle(ifo_events["snr"])
                f["chisq"] = ifo_events["chisq"]
                f["end_time"] = (
                    ifo_events["time_index"] / float(SAMPLE_RATE) + GPS_START[ifo]
                )
                f["time_index"] = ifo_events["time_index"]
                f["slide_id"] = ifo_events["slide_id"]
                try:
                    # Precessing
                    template_sigmasq_plus = np.array(
                        [t["sigmasq_plus"] for t in self.template_params],
                        dtype=np.float32,
                    )
                    f["sigmasq_plus"] = template_sigmasq_plus[tid]
                    template_sigmasq_cross = np.array(
                        [t["sigmasq_cross"] for t in self.template_params],
                        dtype=np.float32,
                    )
                    f["sigmasq_cross"] = template_sigmasq_cross[tid]
                    # FIXME: I want to put something here, but I haven't yet
                    #      figured out what it should be. I think we would also
                    #      need information from the plus and cross correlation
                    #      (both real and imaginary(?)) to get this.
                    f["sigmasq"] = template_sigmasq_plus[tid]
                except Exception:
                    # Not precessing
                    template_sigmasq = np.array(
                        [t["sigmasq"][ifo] for t in self.template_params],
                        dtype=np.float32,
                    )
                    f["sigmasq"] = template_sigmasq[tid]

                # FIXME: Can we get this value from the autochisq instance?
                # cont_dof = self.opt.autochi_number_points
                # if self.opt.autochi_onesided is None:
                #     cont_dof = cont_dof * 2
                # if self.opt.autochi_two_phase:
                #     cont_dof = cont_dof * 2
                # if self.opt.autochi_max_valued_dof:
                #     cont_dof = self.opt.autochi_max_valued_dof
                # f['cont_chisq_dof'] = np.repeat(cont_dof, len(ifo_events))

                if "chisq_dof" in ifo_events.dtype.names:
                    f["chisq_dof"] = ifo_events["chisq_dof"] / 2 + 1
                else:
                    f["chisq_dof"] = np.zeros(len(ifo_events))

                f["template_hash"] = th[tid]
            f["search/time_slides"] = np.array(self.time_slides[ifo])
            if TRIG_START_TIME:
                f["search/start_time"] = np.array(
                    [TRIG_START_TIME[ifo]], dtype=np.int32
                )
                search_start_time = float(TRIG_START_TIME[ifo])
            else:
                f["search/start_time"] = np.array(
                    [GPS_START[ifo] + START_PAD],
                    dtype=np.int32,
                )
                search_start_time = float(GPS_START[ifo] + START_PAD)
            if TRIG_END_TIME:
                f["search/end_time"] = np.array([TRIG_END_TIME[ifo]], dtype=np.int32)
                search_end_time = float(TRIG_END_TIME[ifo])
            else:
                f["search/end_time"] = np.array(
                    [GPS_END[ifo] - END_PAD[ifo]],
                    dtype=np.int32,
                )
                search_end_time = float(GPS_END[ifo] - END_PAD[ifo])

            if self.write_performance:
                self.analysis_time = search_end_time - search_start_time
                time_ratio = np.array(
                    [float(self.analysis_time) / float(self.run_time)]
                )
                temps_per_core = float(self.ntemplates) / float(self.ncores)
                filters_per_core = float(self.nfilters) / float(self.ncores)
                f["search/templates_per_core"] = np.array(
                    [float(temps_per_core) * float(time_ratio)]
                )
                f["search/filter_rate_per_core"] = np.array(
                    [filters_per_core / float(self.run_time)]
                )
                f["search/setup_time_fraction"] = np.array(
                    [float(self.setup_time) / float(self.run_time)]
                )


init_logging(True)

injection_parameters = dict(
    mass_1=79.45,
    mass_2=48.5,
    a_1=0.6,
    a_2=0.05,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=2000.0,
    theta_jn=0.0,
    psi=0.0,
    phase=0.0,
    geocent_time=TIME_GPS_PAST,
    ra=6.0,
    dec=-1.2,
)
return_value = get_strain_list_from_simulation(
    injection_parameters,
    ["H1", "L1"],
    start_time=GPS_START["H1"] - PAD,
    end_time=GPS_END["H1"] + PAD,
    low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
)
optimal_snrs = return_value[0]
matched_filter_snrs = return_value[1]
strain_dict = dict(zip(["H1", "L1"], return_value[2]))
# the lensed image
injection_parameters["geocent_time"] = TIME_GPS_FUTURE
return_value = get_strain_list_from_simulation(
    injection_parameters,
    ["H1", "L1"],
    start_time=GPS_START["H1_lensed"] - PAD,
    end_time=GPS_END["H1_lensed"] + PAD,
    low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
)
optimal_snrs += return_value[0]
matched_filter_snrs += return_value[1]
strain_dict.update(
    dict(
        zip(
            ["H1_lensed", "L1_lensed"],
            return_value[2],
        )
    )
)

optimal_snrs = np.array(optimal_snrs)
matched_filter_snrs = np.array(matched_filter_snrs)

rho = conversions.network_matched_filter_snr(
    abs(matched_filter_snrs),
    optimal_snrs,
)
print("NETWORK SNR", rho)
rho = np.sqrt(sum(abs(matched_filter_snrs) ** 2))
print("COINCIDENCE SNR", rho)

num_slides = slide_limiter()

strain_dict = process_strain_dict(strain_dict, STRAIN_HIGH_PASS, SAMPLE_RATE, PAD)

# Create a dictionary of Python slice objects that indicate where the segments
# start and end for each detector timeseries.
strain_segments_dict = dict()
for ifo in INSTRUMENTS:
    strain_segments_dict[ifo] = StrainSegments(
        strain_dict[ifo],
        segment_length=SEGMENT_LENGTH,
        segment_start_pad=START_PAD,
        segment_end_pad=END_PAD,
        trigger_start=TRIG_START_TIME[ifo],
        trigger_end=TRIG_END_TIME[ifo],
        filter_inj_only=False,
        allow_zero_padding=False,
    )


# Set some convenience variables: number of IFOs, lower frequency,
# GRB time, sky positions to search (either a grid or single sky point)
nifo = len(INSTRUMENTS[:])
sky_positions = sky_grid(
    ra=RA, dec=DEC, sky_error=SKY_ERROR, angular_spacing=ANGULAR_SPACING
)
sky_pos_indices = np.arange(sky_positions.shape[1])
# the sampling rate, flen, tlen and delta_f agree for all detectors
# taking the zeroth detector in the list as a reference.
flen = strain_segments_dict[INSTRUMENTS[0]].freq_len
tlen = strain_segments_dict[INSTRUMENTS[0]].time_len
delta_f = strain_segments_dict[INSTRUMENTS[0]].delta_f
# segments is a dictionary of frequency domain objects, each one of which
# is the Fourier transform of the segments in strain_segments_dict
logging.info("Making frequency-domain data segments")
segments = {ifo: strain_segments_dict[ifo].fourier_segments() for ifo in INSTRUMENTS}
del strain_segments_dict
# Associate PSDs to segments for all IFOs when using the multi-detector CLI
logging.info("Associating PSDs to them")
for ifo in INSTRUMENTS:
    _strain = strain_dict[ifo]
    _segments = segments[ifo]
    seg_stride = int(PSD_SEGMENT_STRIDE * SAMPLE_RATE)
    seg_len = int(PSD_SEGMENT_LENGTH * SAMPLE_RATE)
    input_data_len = len(_strain)
    num_segments = int(PSD_NUM_SEGMENTS)
    psd_data_len = (num_segments - 1) * seg_stride + seg_len
    num_psd_measurements = int(2 * (input_data_len - 1) / psd_data_len)
    psd_stride = int((input_data_len - psd_data_len) / num_psd_measurements)
    psds_and_times = []
    for idx in range(num_psd_measurements):
        if idx == (num_psd_measurements - 1):
            start_idx = input_data_len - psd_data_len
            end_idx = input_data_len
        else:
            start_idx = psd_stride * idx
            end_idx = psd_data_len + psd_stride * idx
        strain_part = _strain[start_idx:end_idx]
        sample_rate = (flen - 1) * 2 * delta_f
        _psd = welch(
            _strain,
            avg_method="median",
            seg_len=int(PSD_SEGMENT_LENGTH * sample_rate + 0.5),
            seg_stride=int(PSD_SEGMENT_STRIDE * sample_rate + 0.5),
            num_segments=PSD_NUM_SEGMENTS,
            require_exact_data_fit=False,
        )

        if delta_f != _psd.delta_f:
            _psd = interpolate(_psd, delta_f, flen)
        _psd = _psd.astype(float32)
        psds_and_times.append((start_idx, end_idx, _psd))
    for fd_segment in _segments:
        best_psd = None
        psd_overlap = 0
        inp_seg = ligo_segments.segment(
            fd_segment.seg_slice.start, fd_segment.seg_slice.stop
        )
        for start_idx, end_idx, _psd in psds_and_times:
            psd_seg = ligo_segments.segment(start_idx, end_idx)
            if psd_seg.intersects(inp_seg):
                curr_overlap = abs(inp_seg & psd_seg)
                if curr_overlap > psd_overlap:
                    psd_overlap = curr_overlap
                    best_psd = _psd
        fd_segment.psd = best_psd

logging.info("Determining time slide shifts and time delays")
# Create a dictionary of time slide shifts; IFO 0 is unshifted
# ANGEL: Just lensed detectors are shifted
slide_ids = np.arange(num_slides)
time_slides = {
    ifo: SLIDE_SHIFT * slide_ids * ifo_idx
    for ifo_idx, ifo in enumerate(LENSED_INSTRUMENTS)
}
time_slides.update(
    {ifo: time_slides[LENSED_INSTRUMENTS[0]] for ifo in UNLENSED_INSTRUMENTS}
)
# Given the time delays wrt to IFO 0 in time_slides, create a dictionary
# for time delay indices evaluated wrt the geocenter, in units of samples,
# i.e. (time delay from geocenter + time slide)*sampling_rate
time_delay_idx_zerolag = {
    position_index: {
        ifo: MyDetector(ifo).time_delay_from_earth_center(
            sky_positions[0][position_index],
            sky_positions[1][position_index],
            TRIGGER_TIMES[ifo],
        )
        for ifo in INSTRUMENTS
    }
    for position_index in sky_pos_indices
}
time_delay_idx = {
    slide: {
        position_index: {
            ifo: int(
                round(
                    (
                        time_delay_idx_zerolag[position_index][ifo]
                        + time_slides[ifo][slide]
                    )
                    * SAMPLE_RATE
                )
            )
            for ifo in INSTRUMENTS
        }
        for position_index in sky_pos_indices
    }
    for slide in slide_ids
}
del time_delay_idx_zerolag

logging.info("Setting up MatchedFilterControl at each IFO")
# Prototype container for the output of MatchedFilterControl and
# waveform.FilterBank (see below).
# Use tlen of the first IFO as it is the same across IFOs.
template_mem = zeros(tlen, dtype=complex64)

# All MatchedFilterControl instances are initialized in the same way.
# This allows to track where the single detector SNR timeseries are
# greater than args.sngl_snr_threshold. Later, coh.get_coinc_indexes
# will enforce the requirement that at least two single detector SNR
# are above args.sngl_snr_threshold, rescuing, where necessary, SNR
# timeseries points for detectors below that threshold.
# NOTE: Do not cluster here for a coherent search (use_cluster=False).
#       Clustering happens at the end of the template loop.
matched_filter = {
    ifo: MatchedFilterControl(
        LOW_FREQUENCY_CUTOFF,
        None,
        SNGL_SNR_THRESHOLD,
        tlen,
        delta_f,
        complex64,
        segments[ifo],
        template_mem,
        use_cluster=False,
        downsample_factor=DOWNSAMPLE_FACTOR,
        upsample_threshold=UPSAMPLE_THRESHOLD,
        upsample_method=UPSAMPLE_METHOD,
    )
    for ifo in INSTRUMENTS
}

# Chi-squares
logging.info("Initializing signal-based vetoes: power")
# Directly use existing SingleDetPowerChisq to calculate single detector chi-squares for
# multiple IFOs
power_chisq = vetoes.SingleDetPowerChisq(CHISQ_BINS)

# Overwhiten all frequency-domain segments by dividing by the PSD estimate
logging.info("Overwhitening frequency-domain data segments")
for ifo in INSTRUMENTS:
    for seg in segments[ifo]:
        seg /= seg.psd

logging.info("Setting up event manager")
# But first build dictionaries to initialize and feed the event manager
ifo_out_types = {
    "time_index": int,
    "ifo": int,  # IFO is stored as an int internally!
    "snr": complex64,
    "chisq": float32,
    "chisq_dof": int,
    "slide_id": int,
}
ifo_out_vals = {
    "time_index": None,
    "ifo": None,
    "snr": None,
    "chisq": None,
    "chisq_dof": None,
    "slide_id": None,
}
ifo_names = sorted(ifo_out_vals.keys())
network_out_types = {
    "dec": float32,
    "ra": float32,
    "time_index": int,
    "coherent_snr": float32,
    "null_snr": float32,
    "nifo": int,
    "my_network_chisq": float32,
    "reweighted_snr": float32,
    "slide_id": int,
}
network_out_vals = {
    "dec": None,
    "ra": None,
    "time_index": None,
    "coherent_snr": None,
    "null_snr": None,
    "nifo": None,
    "my_network_chisq": None,
    "reweighted_snr": None,
    "slide_id": None,
}
network_names = sorted(network_out_vals.keys())
event_mgr = MyEventManagerCoherent(
    [],
    INSTRUMENTS,
    ifo_names,
    [ifo_out_types[n] for n in ifo_names],
    network_names,
    [network_out_types[n] for n in network_names],
    segments=segments,
    time_slides=time_slides,
    gating_info={det: strain_dict[det].gating_info for det in strain_dict},
)

# Template bank: filtering and thinning
logging.info("Read in template bank")
bank = waveform.FilterBank(
    BANK_FILE,
    flen,
    delta_f,
    complex64,
    low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
    phase_order=ORDER,
    taper=TAPER_TEMPLATE,
    approximant=APPROXIMANT,
    out=template_mem,
)

n_bank = len(bank)
logging.info("Full template bank size: %d", n_bank)

logging.info("Calculating antenna pattern functions at every sky position")
antenna_pattern = calculate_antenna_pattern(
    sky_pos_indices, INSTRUMENTS, sky_positions, TRIGGER_TIMES
)

logging.info("Starting the filtering...")
# Loop over templates
for t_num, template in enumerate(bank):
    # Loop over segments
    # ANGEL: this doesn't make any sense, looping with variable `stilde` and
    # defining variable `stilde` inside the loop body??
    for s_num, stilde in enumerate(segments[INSTRUMENTS[0]]):
        stilde = {ifo: segments[ifo][s_num] for ifo in INSTRUMENTS}
        # Find how loud the template is in each detector, i.e., its
        # unnormalized matched-filter with itself. This quantity is
        # used to normalize matched-filters with the data.
        sigmasq = {
            ifo: template.sigmasq(segments[ifo][s_num].psd) for ifo in INSTRUMENTS
        }
        sigma = {ifo: np.sqrt(sigmasq[ifo]) for ifo in INSTRUMENTS}
        # Every time s_num is zero, run new_template to increment the
        # template index
        if s_num == 0:
            event_mgr.new_template(tmplt=template.params, sigmasq=sigmasq)
        logging.info("Analyzing segment %d/%d", s_num + 1, len(segments[ifo]))
        # The following dicts with IFOs as keys are created to store
        # copies of the matched filtering results computed below.
        # - Complex SNR time series
        snr_dict = dict.fromkeys(INSTRUMENTS)
        # - Its normalization
        norm_dict = dict.fromkeys(INSTRUMENTS)
        # - The correlation vector frequency series
        #   It is the FFT of the SNR (so inverse FFT it to get the SNR)
        corr_dict = dict.fromkeys(INSTRUMENTS)
        # - The trigger indices list (idx_dict will be created out of this)
        idx = dict.fromkeys(INSTRUMENTS)
        # - The list of normalized SNR values at the trigger locations
        snr = dict.fromkeys(INSTRUMENTS)
        for ifo in INSTRUMENTS:
            logging.info("  Filtering template %d/%d, ifo %s", t_num + 1, n_bank, ifo)
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
        n_trigs = [len(snr[ifo]) for ifo in INSTRUMENTS]
        if not any(n_trigs):
            continue

        # Loop over (short) time-slides, staring with the zero-lag
        for slide in range(num_slides):
            logging.info("  Analyzing slide %d/%d", slide, num_slides)
            # Loop over sky positions
            for position_index in sky_pos_indices:
                logging.info(
                    "    Analyzing sky position %d/%d",
                    position_index + 1,
                    len(sky_pos_indices),
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
                    for ifo in INSTRUMENTS
                }
                # Find triggers that are coincident (in geocent time) in
                # multiple IFOs. If a single IFO analysis then just use the
                # indices from that IFO, i.e., IFO 0; otherwise, this
                # method finds coincidences and applies the single IFO cut,
                # namely, triggers must have at least 2 IFO SNRs above
                # args.sngl_snr_threshold.
                coinc_idx = get_coinc_indexes(
                    idx_dict, time_delay_idx[slide][position_index], LENSED_INSTRUMENTS
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
                        COINC_THRESHOLD,
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
                        for ifo in INSTRUMENTS
                    }
                    fc = {
                        ifo: antenna_pattern[ifo][position_index][1]
                        for ifo in INSTRUMENTS
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
                        COINC_THRESHOLD,
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
                            apply_cut=DO_NULL_CUT,
                            null_min=NULL_MIN,
                            null_grad=NULL_GRAD,
                            null_step=NULL_STEP,
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
                        for ifo in INSTRUMENTS
                    }
                    # Build dictionary with per-IFO complex SNR time series
                    # of the most recent set of triggers
                    coherent_ifo_trigs = {
                        ifo: snr_dict[ifo][coinc_idx_det_frame[ifo]]
                        for ifo in INSTRUMENTS
                    }
                    # Calculate the powerchi2 values of remaining triggers
                    # (this uses the SNR timeseries before the time delay,
                    # so we undo it; the same holds for normalisation)
                    chisq = {}
                    chisq_dof = {}
                    for ifo in INSTRUMENTS:
                        chisq[ifo], chisq_dof[ifo] = power_chisq.values(
                            corr_dict[ifo],
                            coherent_ifo_trigs[ifo] / norm_dict[ifo],
                            norm_dict[ifo],
                            stilde[ifo].psd,
                            coinc_idx_det_frame[ifo] + stilde[ifo].analyze.start,
                            template,
                        )
                    # Calculate network chisq value
                    network_chisq_dict = coh.network_chisq(
                        chisq, chisq_dof, coherent_ifo_trigs
                    )
                    # Calculate chisq reweighted SNR
                    if nifo > 2:
                        reweighted_snr = ranking.newsnr(
                            rho_coh,
                            network_chisq_dict,
                            q=CHISQ_INDEX,
                            n=CHISQ_NHIGH,
                        )
                        # Calculate null reweighted SNR
                        reweighted_snr = coh.reweight_snr_by_null(
                            reweighted_snr,
                            null,
                            rho_coh,
                            null_min=NULL_MIN,
                            null_grad=NULL_GRAD,
                            null_step=NULL_STEP,
                        )
                    elif nifo == 2:
                        reweighted_snr = ranking.newsnr(
                            rho_coinc,
                            network_chisq_dict,
                            q=CHISQ_INDEX,
                            n=CHISQ_NHIGH,
                        )
                    else:
                        rho_sngl = abs(
                            snr[INSTRUMENTS[0]][coinc_idx_det_frame[INSTRUMENTS[0]]]
                        )
                        reweighted_snr = ranking.newsnr(
                            rho_sngl,
                            network_chisq_dict,
                            q=CHISQ_INDEX,
                            n=CHISQ_NHIGH,
                        )
                    # All out vals must be the same length, so single
                    # value entries are repeated once per event
                    num_events = len(reweighted_snr)
                    for ifo in INSTRUMENTS:
                        ifo_out_vals["chisq"] = chisq[ifo]
                        ifo_out_vals["chisq_dof"] = chisq_dof[ifo]
                        ifo_out_vals["time_index"] = (
                            coinc_idx_det_frame[ifo] + stilde[ifo].cumulative_index
                        )
                        ifo_out_vals["snr"] = coherent_ifo_trigs[ifo]
                        # IFO is stored as an int
                        ifo_out_vals["ifo"] = [event_mgr.ifo_dict[ifo]] * num_events
                        # Time slide ID
                        ifo_out_vals["slide_id"] = [slide] * num_events
                        event_mgr.add_template_events_to_ifo(
                            ifo,
                            ifo_names,
                            [ifo_out_vals[n] for n in ifo_names],
                        )
                    if nifo > 2:
                        network_out_vals["coherent_snr"] = rho_coh
                        network_out_vals["null_snr"] = null
                    elif nifo == 2:
                        network_out_vals["coherent_snr"] = rho_coinc
                    else:
                        network_out_vals["coherent_snr"] = abs(
                            snr[INSTRUMENTS[0]][coinc_idx_det_frame[INSTRUMENTS[0]]]
                        )
                    network_out_vals["reweighted_snr"] = reweighted_snr
                    network_out_vals["my_network_chisq"] = np.real(network_chisq_dict)
                    network_out_vals["time_index"] = (
                        coinc_idx + stilde[ifo].cumulative_index
                    )
                    network_out_vals["nifo"] = [nifo] * num_events
                    network_out_vals["dec"] = [
                        sky_positions[1][position_index]
                    ] * num_events
                    network_out_vals["ra"] = [
                        sky_positions[0][position_index]
                    ] * num_events
                    network_out_vals["slide_id"] = [slide] * num_events
                    event_mgr.add_template_events_to_network(
                        network_names,
                        [network_out_vals[n] for n in network_names],
                    )
        # Left loops over sky positions and time-slides,
        # but not loops over segments and templates.
        # The triggers can be clustered
        cluster_window = int(CLUSTER_WINDOW * SAMPLE_RATE)
        # Cluster template events by slide
        for slide in range(num_slides):
            logging.info("  Clustering slide %d", slide)
            event_mgr.cluster_template_network_events(
                "time_index", "reweighted_snr", cluster_window, slide=slide
            )
    # Left loop over segments
    event_mgr.finalize_template_events()
# Left loop over templates
logging.info("Filtering completed")

logging.info("Writing output")
event_mgr.write_events(OUTPUT)

logging.info("Finished")
