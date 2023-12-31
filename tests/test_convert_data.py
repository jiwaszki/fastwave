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


def ref_read(file_path, mono, dtype):
    audio = wavfile.read(file_path)[1]
    if dtype is not None:
        audio = audio.astype(dtype)
        if dtype == np.float32:
            audio = audio / 32767.0
    if mono and audio.ndim != 1:
        # SciPy data is saved as samples-first, convert to channel-first:
        audio = audio.T
        audio = np.mean(audio, axis=tuple(range(audio.ndim - 1)))
    return audio


def test_convert_data_fails_input():
    # audio_data = f.read(FILES[0])
    with pytest.raises(NotImplementedError) as error:
        _ = f.convert_data("./path_to_a_file")
    assert str(error.value) in "Unknown dispatch has been used!"


@pytest.mark.parametrize(
    ("failing_dtype"),
    (
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float64,
    ),
)
def test_convert_data_fails_dtype(failing_dtype):
    audio_data = f.read(FILES[0])
    # Direct numpy
    with pytest.raises(NotImplementedError) as error:
        _ = f.convert_data(audio_data.data, dtype=failing_dtype)
    assert str(error.value) in f"Conversion for {failing_dtype} is not supported!"
    # Using AudioData class
    with pytest.raises(NotImplementedError) as error:
        _ = f.convert_data(audio_data, dtype=failing_dtype)
    assert str(error.value) in f"Conversion for {failing_dtype} is not supported!"


@pytest.mark.parametrize(
    ("file_name"),
    (FILES),
)
@pytest.mark.parametrize(
    ("mono"),
    (True, False),
)
@pytest.mark.parametrize(
    ("dtype"),
    (np.float32,),
)
def test_convert_data(file_name, mono, dtype):
    ref_data = ref_read(file_name, mono, dtype)
    audio_data = f.read(file_name)
    # Direct numpy
    result = f.convert_data(audio_data.data, mono=mono, dtype=dtype)
    assert result.ndim == ref_data.ndim
    assert result.shape == ref_data.shape
    assert np.array_equal(result, ref_data)
    # Using AudioData class
    result = f.convert_data(audio_data, mono=mono, dtype=dtype)
    assert result.ndim == ref_data.ndim
    assert result.shape == ref_data.shape
    assert np.array_equal(result, ref_data)
