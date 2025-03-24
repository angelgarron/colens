import logging
from functools import partial

import numpy as np
from pycbc import init_logging, vetoes, waveform
from pycbc.filter import MatchedFilterControl
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens.background import slide_limiter
from colens.brute_force_filter import brute_force_filter_template
from colens.detector import MyDetector
from colens.fstatistic import get_two_f
from colens.injection import get_strain_list_from_bilby_simulation
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import (
    Output,
    PerDetectorOutput,
    create_filter_bank,
    get_strain_dict_from_files,
)
from colens.psd import associate_psd_to_segments
from colens.sky import get_circular_sky_patch, get_sky_grid_for_three_detectors
from colens.strain import process_strain_dict

FRAME_FILES = {
    "H1": "/home/angel/Documents/pycbc_checks/H-H1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "L1": "/home/angel/Documents/pycbc_checks/L-L1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "H1_lensed": "/home/angel/Documents/pycbc_checks/H-H1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
    "L1_lensed": "/home/angel/Documents/pycbc_checks/L-L1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
}
CHANNELS = {
    "H1": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1": "L1:GWOSC-4KHZ_R1_STRAIN",
    "H1_lensed": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1_lensed": "L1:GWOSC-4KHZ_R1_STRAIN",
}
TIME_GPS_PAST_SECONDS = 1185389807.298705
TIME_GPS_FUTURE_SECONDS = 1185437144.7875977
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
REFERENCE_FREQUENCY = 50.0
SNGL_SNR_THRESHOLD = 0.0
DOWNSAMPLE_FACTOR = 1
UPSAMPLE_THRESHOLD = None
UPSAMPLE_METHOD = "pruned_fft"
BANK_FILE = "gw170729_single_template.hdf"
ORDER = "-1"
TAPER_TEMPLATE = None
APPROXIMANT = "IMRPhenomXAS"
COINC_THRESHOLD = 0.0
DO_NULL_CUT = False
NULL_MIN = 5.25
NULL_GRAD = 0.2
NULL_STEP = 20.0
CLUSTER_WINDOW = 0.1
OUTPUT_FILE_NAME = "results.json"
PAD_SECONDS = 8
GPS_START_SECONDS = dict()
GPS_END_SECONDS = dict()
for ifo in INSTRUMENTS:
    GPS_START_SECONDS[ifo] = int(TRIGGER_TIMES_SECONDS[ifo]) - 192 - PAD_SECONDS
    GPS_END_SECONDS[ifo] = int(TRIGGER_TIMES_SECONDS[ifo]) + 192 + PAD_SECONDS
SEGMENT_START_PAD_SECONDS = (
    111  # time in seconds to ignore at the beginning of each segment
)
SEGMENT_END_PAD_SECONDS = 17  # time in seconds to ignore at the end of each segment
TRIG_START_TIME_SECONDS = dict()  # gps time to start recording triggers
TRIG_END_TIME_SECONDS = dict()  # gps time to stop recording triggers
for ifo in INSTRUMENTS:
    TRIG_START_TIME_SECONDS[ifo] = GPS_START_SECONDS[ifo] + SEGMENT_START_PAD_SECONDS
    TRIG_END_TIME_SECONDS[ifo] = GPS_END_SECONDS[ifo] - SEGMENT_END_PAD_SECONDS
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
PSD_SEGMENT_STRIDE_SECONDS = 8.0  # separation (in seconds) between consecutive segments
PSD_SEGMENT_LENGTH_SECONDS = 32.0  # segment length (in seconds) for PSD estimation
PSD_NUM_SEGMENTS = 29  # PSD estimated using only this number of segments
STRAIN_HIGH_PASS_HERTZ = 25.0
ANGULAR_SPACING = 1.8 * np.pi / 180  # radians
SKY_ERROR = 0.1 * np.pi / 180  # radians


def create_injections(injection_parameters: dict[str, float]):
    # The extra padding we are adding here is going to get removed after highpassing
    return_value = get_strain_list_from_bilby_simulation(
        injection_parameters,
        ["H1", "L1"],
        start_time=GPS_START_SECONDS["H1"] - conf.injection.pad_seconds,
        end_time=GPS_END_SECONDS["H1"] + conf.injection.pad_seconds,
        low_frequency_cutoff=conf.injection.low_frequency_cutoff,
        reference_frequency=conf.injection.reference_frequency,
        sampling_frequency=conf.injection.sample_rate,
        seed=1,
        approximant=conf.injection.approximant,
        is_zero_noise=False,
        # is_real_noise=True,
    )
    strain_dict = dict(zip(["H1", "L1"], return_value))
    # the lensed image
    injection_parameters["geocent_time"] = conf.injection.time_gps_future_seconds
    return_value = get_strain_list_from_bilby_simulation(
        injection_parameters,
        ["H1", "L1"],
        start_time=GPS_START_SECONDS["H1_lensed"] - conf.injection.pad_seconds,
        end_time=GPS_END_SECONDS["H1_lensed"] + conf.injection.pad_seconds,
        low_frequency_cutoff=conf.injection.low_frequency_cutoff,
        reference_frequency=conf.injection.reference_frequency,
        sampling_frequency=conf.injection.sample_rate,
        seed=2,
        approximant=conf.injection.approximant,
        is_zero_noise=False,
        # is_real_noise=True,
        suffix="_lensed",
    )
    strain_dict.update(
        dict(
            zip(
                ["H1_lensed", "L1_lensed"],
                return_value,
            )
        )
    )

    return strain_dict


def main():
    init_logging(True)
    output_data = Output(
        H1=PerDetectorOutput(),
        H1_lensed=PerDetectorOutput(),
        L1=PerDetectorOutput(),
        L1_lensed=PerDetectorOutput(),
    )

    lensed_detectors = {
        ifo: MyDetector(ifo) for ifo in conf.injection.lensed_instruments
    }
    unlensed_detectors = {
        ifo: MyDetector(ifo) for ifo in conf.injection.unlensed_instruments
    }

    logging.info("Creating template bank")
    create_filter_bank(
        79.45,
        48.50,
        0.60,
        0.05,
        conf.injection.bank_file,
        conf.injection.approximant,
        conf.injection.low_frequency_cutoff,
        conf.injection.reference_frequency,
    )

    # logging.info("Injecting simulated signals on gaussian noise")
    # injection_parameters = dict(
    #     mass_1=79.45,
    #     mass_2=48.5,
    #     a_1=0.6,
    #     a_2=0.05,
    #     tilt_1=0.0,
    #     tilt_2=0.0,
    #     phi_12=0.0,
    #     phi_jl=0.0,
    #     luminosity_distance=2000.0,
    #     theta_jn=0.0,
    #     psi=0.0,
    #     phase=0.0,
    #     geocent_time=TIME_GPS_PAST_SECONDS,
    #     ra=6.0,
    #     dec=-1.2,
    # )
    # strain_dict = create_injections(injection_parameters)

    process_strain_dict(
        strain_dict,
        conf.psd.strain_high_pass_hertz,
        conf.injection.sample_rate,
        conf.injection.pad_seconds,
    )

    num_slides = slide_limiter(
        conf.injection.segment_length_seconds,
        conf.injection.slide_shift_seconds,
        len(conf.injection.lensed_instruments),
    )
    num_slides = 1

    # Create a dictionary of Python slice objects that indicate where the segments
    # start and end for each detector timeseries.
    segments = dict()
    for ifo in INSTRUMENTS:
        segments[ifo] = StrainSegments(
            strain_dict[ifo],
            segment_length=conf.injection.segment_length_seconds,
            segment_start_pad=conf.injection.segment_start_pad_seconds,
            segment_end_pad=conf.injection.segment_end_pad_seconds,
            trigger_start=TRIG_START_TIME_SECONDS[ifo],
            trigger_end=TRIG_END_TIME_SECONDS[ifo],
            filter_inj_only=False,
            allow_zero_padding=False,
        ).fourier_segments()

    sky_grid = get_circular_sky_patch(
        ra=RA, dec=DEC, sky_error=SKY_ERROR, angular_spacing=ANGULAR_SPACING
    )

    delta_f = (
        1.0 / SEGMENT_LENGTH_SECONDS
    )  # frequency step of the fourier transform of each segment
    segment_length = int(
        SEGMENT_LENGTH_SECONDS * SAMPLE_RATE
    )  # number of samples of each segment
    frequency_length = int(
        segment_length // 2 + 1
    )  # number of samples of the fourier transform of each segment

    logging.info("Associating PSDs to the fourier segments")
    for ifo in INSTRUMENTS:
        associate_psd_to_segments(
            strain_dict[ifo],
            segments[ifo],
            PSD_SEGMENT_STRIDE_SECONDS,
            SAMPLE_RATE,
            PSD_SEGMENT_LENGTH_SECONDS,
            PSD_NUM_SEGMENTS,
            frequency_length,
            delta_f,
        )

    logging.info("Setting up MatchedFilterControl at each IFO")
    template_mem = zeros(segment_length, dtype=complex64)

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
            segment_length,
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

    logging.info("Read in template bank")
    bank = waveform.FilterBank(
        BANK_FILE,
        frequency_length,
        delta_f,
        complex64,
        low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
        phase_order=ORDER,
        taper=TAPER_TEMPLATE,
        approximant=APPROXIMANT,
        out=template_mem,
    )

    logging.info("Full template bank size: %d", len(bank))

    logging.info("Starting the filtering...")
    for t_num, template in enumerate(bank):
        logging.info("Filtering template %d/%d", t_num + 1, len(bank))
        brute_force_filter_template(
            lensed_detectors,
            unlensed_detectors,
            segments,
            INSTRUMENTS,
            template,
            matched_filter,
            num_slides,
            COINC_THRESHOLD,
            NULL_MIN,
            NULL_GRAD,
            NULL_STEP,
            power_chisq,
            CHISQ_INDEX,
            CHISQ_NHIGH,
            CLUSTER_WINDOW,
            SLIDE_SHIFT_SECONDS,
            SAMPLE_RATE,
            GPS_START_SECONDS,
            TIME_GPS_PAST_SECONDS,
            TIME_GPS_FUTURE_SECONDS,
            get_two_f,
            output_data,
            get_snr_interpolated,
        )

    logging.info("Filtering completed")
    logging.info(f"Saving results to {OUTPUT_FILE_NAME}")
    output_data.write_to_json(OUTPUT_FILE_NAME)


if __name__ == "__main__":
    main()
