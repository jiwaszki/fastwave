import fastwave as f
import torchaudio  # reference library


def ref_info(fp):
    info = {}
    si = torchaudio.info(str(fp))
    info["sampling_rate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["sampling_rate"]
    return info


