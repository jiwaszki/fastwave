import pytest

import numpy as np
from scipy.io import wavfile  # reference library

import fastwave as f


FILES = (
    "./test_files/0.wav",
    "./test_files/1.wav",
    "./test_files/2.wav",
    "./test_files/3.wav",
)


def ref_read(file_path):
    # audio, _ = torchaudio.load(file_path, normalize=False)
    # return audio.numpy().flatten()
    return wavfile.read(file_path)[1]


@pytest.mark.parametrize(
    ("cache_size"),
    (-2137, 0),
)
def test_read_threads_fails_cache(cache_size):
    with pytest.raises(ValueError) as error:
        _ = f.read(FILES[0], mode=f.ReadMode.THREADS, cache_size=cache_size)
    assert str(error.value) in "cache_size must be more than 0!"


@pytest.mark.parametrize(
    ("num_threads"),
    (-1, 0),
)
def test_read_threads_fails_num_threads(num_threads):
    with pytest.raises(ValueError) as error:
        _ = f.read(FILES[0], mode=f.ReadMode.THREADS, num_threads=num_threads)
    assert str(error.value) in "num_threads must be more than 0!"


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
def test_read_default(file_name):
    ref_data = ref_read(file_name)
    audio_data = f.read(file_name, mode=f.ReadMode.DEFAULT)
    assert np.array_equal(ref_data, audio_data.data)


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
@pytest.mark.parametrize(
    ("cache_size"),
    (1, 21, 420, 512, 2137, 8192, 32770, 131072),
)
@pytest.mark.parametrize(
    ("num_threads"),
    (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32),
)
def test_read_threads(file_name, cache_size, num_threads):
    ref_data = ref_read(file_name)
    audio_data = f.read(
        file_name,
        mode=f.ReadMode.THREADS,
        cache_size=cache_size,
        num_threads=num_threads,
    )
    assert np.array_equal(ref_data, audio_data.data)


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
@pytest.mark.parametrize(
    ("mode"),
    (f.ReadMode.MMAP_PRIVATE, f.ReadMode.MMAP_SHARED),
)
def test_read_mmap(file_name, mode):
    ref_data = ref_read(file_name)
    audio_data = f.read(file_name, mode=mode)
    assert np.array_equal(ref_data, audio_data.data)
