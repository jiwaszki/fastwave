import pytest

import numpy as np
import torchaudio  # reference library

import fastwave as f


FILES = (
    "./test_files/0.wav",
    "./test_files/1.wav",
    "./test_files/2.wav",
    "./test_files/3.wav",
)


def ref_read(file_path):
    info = {}
    _info = torchaudio.info(file_path)
    info["duration"] = _info["samples"] / _info["sampling_rate"]
    info["num_samples"] = _info.num_frames
    info["num_channels"] = _info.num_channels
    info["sample_rate"] = _info.sample_rate
    return info


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
def test_info_direct(file_name):
    ref_info = ref_read(file_name)
    audio_info = f.info(file_name)
    assert ref_info["duration"] == audio_info.duration
    assert ref_info["num_samples"] == audio_info.num_samples
    assert ref_info["num_channels"] == audio_info.num_channels
    assert ref_info["sample_rate"] == audio_info.sample_rate
    # assert ref_info["bit_depth"] == audio_info.bit_depth


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
@pytest.mark.parametrize(
    ("mode"),
    (
        f.ReadMode.DEFAULT,
        f.ReadMode.THREADS,
        f.ReadMode.MMAP_PRIVATE,
        f.ReadMode.MMAP_SHARED,
    ),
)
def test_info_read(file_name, mode):
    ref_info = ref_read(file_name)
    audio_info = f.read(file_name, mode=mode)
    assert ref_info["duration"] == audio_info.duration
    assert ref_info["num_samples"] == audio_info.num_samples
    assert ref_info["num_channels"] == audio_info.num_channels
    assert ref_info["sample_rate"] == audio_info.sample_rate
    # assert ref_info["bit_depth"] == audio_info.bit_depth
