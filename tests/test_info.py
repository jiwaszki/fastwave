import pytest

import numpy as np
import wave  # reference library -- native Python

import fastwave as f


FILES = (
    "./test_files/0.wav",
    "./test_files/1.wav",
    "./test_files/2.wav",
    "./test_files/3.wav",
)


def ref_read(file_path):
    info = {}
    with wave.open(file_path, mode="rb") as audio:
        # Need to calculate duration by hand:
        info["duration"] = audio.getnframes() / audio.getframerate()
        info["num_samples"] = audio.getnframes()
        info["num_channels"] = audio.getnchannels()
        info["sample_rate"] = audio.getframerate()
        # wave returns width in bytes, multiply it to get bits:
        info["bit_depth"] = audio.getsampwidth() * 8
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
    assert ref_info["bit_depth"] == audio_info.bit_depth


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
    audio_info = f.read(file_name, mode=mode).info
    assert ref_info["duration"] == audio_info.duration
    assert ref_info["num_samples"] == audio_info.num_samples
    assert ref_info["num_channels"] == audio_info.num_channels
    assert ref_info["sample_rate"] == audio_info.sample_rate
    assert ref_info["bit_depth"] == audio_info.bit_depth
