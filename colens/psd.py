from ligo import segments as ligo_segments
from pycbc.psd.estimate import interpolate, welch
from pycbc.types import float32


def associate_psd_to_segments(
    strain,
    segments,
    psd_segment_stride,
    sample_rate,
    psd_segment_length,
    psd_num_segments,
    flen,
    delta_f,
):
    seg_stride = int(psd_segment_stride * sample_rate)
    seg_len = int(psd_segment_length * sample_rate)
    input_data_len = len(strain)
    num_segments = int(psd_num_segments)
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
        strain_part = strain[start_idx:end_idx]
        sample_rate = (flen - 1) * 2 * delta_f
        _psd = welch(
            strain_part,
            avg_method="median",
            seg_len=int(psd_segment_length * sample_rate + 0.5),
            seg_stride=int(psd_segment_stride * sample_rate + 0.5),
            num_segments=psd_num_segments,
            require_exact_data_fit=False,
        )

        if delta_f != _psd.delta_f:
            _psd = interpolate(_psd, delta_f, flen)
        _psd = _psd.astype(float32)
        psds_and_times.append((start_idx, end_idx, _psd))
    for fd_segment in segments:
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
