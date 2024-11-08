from ligo import segments as ligo_segments
from pycbc.psd.estimate import interpolate, welch
from pycbc.types import FrequencySeries, TimeSeries, float32


def associate_psd_to_segments(
    strain: TimeSeries,
    segments: list[FrequencySeries],
    psd_segment_stride_seconds: int | float,
    sample_rate: int,
    psd_segment_length_seconds: int | float,
    psd_num_segments: int,
    flen: int,
    delta_f: float,
) -> None:
    """Generate a set of overlapping PSDs covering the data in `strain`.
    Then associate these PSDs with the appropriate segment in `strain`.

    Args:
        strain (TimeSeries): Time series containing the data from which the PSD should be measured.
        segments (list[FrequencySeries]): A list of frequencyseries corresponding to each segment
        in which the strain of the science block is divided.
        psd_segment_stride_seconds (int | float): The separation (in seconds) of the consecutive
        sub-segments inside each segment of the science block used for computing the PSD.
        sample_rate (int): The sample rate (in Hertz).
        psd_segment_length_seconds (int | float): The duration (in seconds) of each sub-segment
        used for the estimation of the PSD.
        psd_num_segments (int): PSDs will be estimated using only this number of segments.
        flen (int): The length (in samples) of the output PSD.
        delta_f (float): The frequency step of the output PSD.
    """
    psd_seg_stride = int(psd_segment_stride_seconds * sample_rate)
    psd_seg_len = int(psd_segment_length_seconds * sample_rate)
    input_data_len = len(strain)
    psd_data_len = (psd_num_segments - 1) * psd_seg_stride + psd_seg_len
    num_psd_measurements = int(2 * (input_data_len - 1) / psd_data_len)
    psd_stride = int((input_data_len - psd_data_len) / num_psd_measurements)
    psds_and_times = []

    for idx in range(num_psd_measurements):
        if idx == (num_psd_measurements - 1):
            start = input_data_len - psd_data_len
            end = input_data_len
        else:
            start = psd_stride * idx
            end = psd_data_len + psd_stride * idx
        strain_part = strain[start:end]
        sample_rate = (flen - 1) * 2 * delta_f
        psd = welch(
            strain_part,
            avg_method="median",
            seg_len=int(psd_segment_length_seconds * sample_rate + 0.5),
            seg_stride=int(psd_segment_stride_seconds * sample_rate + 0.5),
            num_segments=psd_num_segments,
            require_exact_data_fit=False,
        )

        if delta_f != psd.delta_f:
            psd = interpolate(psd, delta_f, flen)
        psd = psd.astype(float32)
        psds_and_times.append((start, end, psd))

    for fd_segment in segments:
        best_psd = None
        psd_overlap = 0
        inp_seg = ligo_segments.segment(
            fd_segment.seg_slice.start, fd_segment.seg_slice.stop
        )
        for start, end, psd in psds_and_times:
            psd_seg = ligo_segments.segment(start, end)
            if psd_seg.intersects(inp_seg):
                curr_overlap = abs(inp_seg & psd_seg)
                if curr_overlap > psd_overlap:
                    psd_overlap = curr_overlap
                    best_psd = psd
        fd_segment.psd = best_psd
