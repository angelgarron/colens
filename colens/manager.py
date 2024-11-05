"""Definition of the modified event manager."""

import numpy as np
from pycbc.events import EventManagerCoherent
from pycbc.events.eventmgr import H5FileSyntSugar


class MyEventManagerCoherent(EventManagerCoherent):
    def write_to_hdf(
        self, outname, sample_rate, gps_start, trig_start_time, trig_end_time
    ):
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
                    network_events[col] / float(sample_rate) + gps_start[self.ifos[0]]
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
                f["coa_phase"] = np.angle(ifo_events["snr"])
                f["chisq"] = ifo_events["chisq"]
                f["end_time"] = (
                    ifo_events["time_index"] / float(sample_rate) + gps_start[ifo]
                )
                f["time_index"] = ifo_events["time_index"]
                f["slide_id"] = ifo_events["slide_id"]
                template_sigmasq = np.array(
                    [t["sigmasq"][ifo] for t in self.template_params],
                    dtype=np.float32,
                )
                f["sigmasq"] = template_sigmasq[tid]

                if "chisq_dof" in ifo_events.dtype.names:
                    f["chisq_dof"] = ifo_events["chisq_dof"] / 2 + 1
                else:
                    f["chisq_dof"] = np.zeros(len(ifo_events))

                f["template_hash"] = th[tid]
            f["search/time_slides"] = np.array(self.time_slides[ifo])
            f["search/start_time"] = np.array([trig_start_time[ifo]], dtype=np.int32)
            f["search/end_time"] = np.array([trig_end_time[ifo]], dtype=np.int32)
