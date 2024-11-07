import logging

import numpy as np
from pesummary.gw import conversions
from pycbc import init_logging, vetoes, waveform
from pycbc.filter import MatchedFilterControl
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens.background import get_time_delay_indices, slide_limiter
from colens.detector import calculate_antenna_pattern
from colens.filter import filter_template
from colens.injection import get_strain_list_from_simulation
from colens.io import create_filter_bank
from colens.manager import MyEventManagerCoherent
from colens.psd import associate_psd_to_segments
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
TIME_GPS_PAST_SECONDS = 1185389807
TIME_GPS_FUTURE_SECONDS = 1185437144
TRIGGER_TIMES_SECONDS = {
    "H1": TIME_GPS_PAST_SECONDS,
    "L1": TIME_GPS_PAST_SECONDS,
    "H1_lensed": TIME_GPS_FUTURE_SECONDS,
    "L1_lensed": TIME_GPS_FUTURE_SECONDS,
}
RA, DEC = 6.0, -1.2
SAMPLE_RATE = 4096.0
UNLENSED_INSTRUMENTS = ["H1", "L1"]
LENSED_INSTRUMENTS = ["H1_lensed", "L1_lensed"]
INSTRUMENTS = UNLENSED_INSTRUMENTS + LENSED_INSTRUMENTS
INSTRUMENTS.sort()
SEGMENT_LENGTH_SECONDS = 256
SLIDE_SHIFT_SECONDS = 1
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
PAD_SECONDS = 8
GPS_START_SECONDS = dict()
GPS_END_SECONDS = dict()
for ifo in INSTRUMENTS:
    GPS_START_SECONDS[ifo] = TRIGGER_TIMES_SECONDS[ifo] - 192 - PAD_SECONDS
    GPS_END_SECONDS[ifo] = TRIGGER_TIMES_SECONDS[ifo] + 192 + PAD_SECONDS
START_PAD_SECONDS = 111  # time in seconds to ignore of the beginning of each segment
END_PAD_SECONDS = 17  # time in seconds to ignore at the end of each segment
TRIG_START_TIME_SECONDS = dict()  # gps time to start recording triggers
TRIG_END_TIME_SECONDS = dict()  # gps time to stop recording triggers
for ifo in INSTRUMENTS:
    TRIG_START_TIME_SECONDS[ifo] = GPS_START_SECONDS[ifo] + START_PAD_SECONDS
    TRIG_END_TIME_SECONDS[ifo] = GPS_END_SECONDS[ifo] - END_PAD_SECONDS
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


def create_injections(injection_parameters: dict[str, float]):
    return_value = get_strain_list_from_simulation(
        injection_parameters,
        ["H1", "L1"],
        start_time=GPS_START_SECONDS["H1"] - PAD_SECONDS,
        end_time=GPS_END_SECONDS["H1"] + PAD_SECONDS,
        low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
    )
    optimal_snrs = return_value[0]
    matched_filter_snrs = return_value[1]
    strain_dict = dict(zip(["H1", "L1"], return_value[2]))
    # the lensed image
    injection_parameters["geocent_time"] = TIME_GPS_FUTURE_SECONDS
    return_value = get_strain_list_from_simulation(
        injection_parameters,
        ["H1", "L1"],
        start_time=GPS_START_SECONDS["H1_lensed"] - PAD_SECONDS,
        end_time=GPS_END_SECONDS["H1_lensed"] + PAD_SECONDS,
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
    logging.info(f"NETWORK SNR {rho}")
    rho = np.sqrt(sum(abs(matched_filter_snrs) ** 2))
    logging.info(f"COINCIDENCE SNR {rho}")
    strain_dict = process_strain_dict(
        strain_dict, STRAIN_HIGH_PASS, SAMPLE_RATE, PAD_SECONDS
    )
    return strain_dict


def main():
    init_logging(True)

    logging.info("Creating template bank")
    create_filter_bank(
        79.45, 48.50, 0.60, 0.05, BANK_FILE, APPROXIMANT, LOW_FREQUENCY_CUTOFF
    )

    logging.info("Injecting simulated signals on gaussian noise")
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
        geocent_time=TIME_GPS_PAST_SECONDS,
        ra=6.0,
        dec=-1.2,
    )
    strain_dict = create_injections(injection_parameters)

    num_slides = slide_limiter(
        SEGMENT_LENGTH_SECONDS, SLIDE_SHIFT_SECONDS, len(LENSED_INSTRUMENTS)
    )

    # Create a dictionary of Python slice objects that indicate where the segments
    # start and end for each detector timeseries.
    strain_segments_dict = dict()
    for ifo in INSTRUMENTS:
        strain_segments_dict[ifo] = StrainSegments(
            strain_dict[ifo],
            segment_length=SEGMENT_LENGTH_SECONDS,
            segment_start_pad=START_PAD_SECONDS,
            segment_end_pad=END_PAD_SECONDS,
            trigger_start=TRIG_START_TIME_SECONDS[ifo],
            trigger_end=TRIG_END_TIME_SECONDS[ifo],
            filter_inj_only=False,
            allow_zero_padding=False,
        )

    sky_positions = sky_grid(
        ra=RA, dec=DEC, sky_error=SKY_ERROR, angular_spacing=ANGULAR_SPACING
    )
    sky_pos_indices = np.arange(sky_positions.shape[1])
    flen = strain_segments_dict[INSTRUMENTS[0]].freq_len
    tlen = strain_segments_dict[INSTRUMENTS[0]].time_len
    delta_f = strain_segments_dict[INSTRUMENTS[0]].delta_f
    # segments is a dictionary of frequency domain objects, each one of which
    # is the Fourier transform of the segments in strain_segments_dict
    logging.info("Making frequency-domain data segments")
    segments = {
        ifo: strain_segments_dict[ifo].fourier_segments() for ifo in INSTRUMENTS
    }
    del strain_segments_dict
    logging.info("Associating PSDs to them")
    for ifo in INSTRUMENTS:
        associate_psd_to_segments(
            strain_dict[ifo],
            segments[ifo],
            PSD_SEGMENT_STRIDE,
            SAMPLE_RATE,
            PSD_SEGMENT_LENGTH,
            PSD_NUM_SEGMENTS,
            flen,
            delta_f,
        )

    logging.info("Determining time slide shifts and time delays")

    time_slides, time_delay_idx = get_time_delay_indices(
        num_slides,
        SLIDE_SHIFT_SECONDS,
        LENSED_INSTRUMENTS,
        UNLENSED_INSTRUMENTS,
        sky_positions,
        TRIGGER_TIMES_SECONDS,
        sky_pos_indices,
        SAMPLE_RATE,
    )

    logging.info("Setting up MatchedFilterControl at each IFO")
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

    logging.info("Initializing signal-based vetoes: power")
    power_chisq = vetoes.SingleDetPowerChisq(CHISQ_BINS)

    logging.info("Overwhitening frequency-domain data segments")
    for ifo in INSTRUMENTS:
        for seg in segments[ifo]:
            seg /= seg.psd

    logging.info("Setting up event manager")
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

    logging.info("Full template bank size: %d", len(bank))

    logging.info("Calculating antenna pattern functions at every sky position")
    antenna_pattern = calculate_antenna_pattern(
        sky_pos_indices, INSTRUMENTS, sky_positions, TRIGGER_TIMES_SECONDS
    )

    logging.info("Starting the filtering...")
    for t_num, template in enumerate(bank):
        logging.info("Filtering template %d/%d", t_num + 1, len(bank))
        filter_template(
            segments,
            INSTRUMENTS,
            template,
            event_mgr,
            matched_filter,
            num_slides,
            sky_pos_indices,
            time_delay_idx,
            LENSED_INSTRUMENTS,
            COINC_THRESHOLD,
            antenna_pattern,
            DO_NULL_CUT,
            NULL_MIN,
            NULL_GRAD,
            NULL_STEP,
            power_chisq,
            CHISQ_INDEX,
            CHISQ_NHIGH,
            ifo_out_vals,
            network_out_vals,
            ifo_names,
            sky_positions,
            CLUSTER_WINDOW,
            SAMPLE_RATE,
            network_names,
        )

    logging.info("Filtering completed")

    logging.info("Writing output")
    event_mgr.write_to_hdf(
        OUTPUT,
        SAMPLE_RATE,
        GPS_START_SECONDS,
        TRIG_START_TIME_SECONDS,
        TRIG_END_TIME_SECONDS,
    )

    logging.info("Finished")


if __name__ == "__main__":
    main()
