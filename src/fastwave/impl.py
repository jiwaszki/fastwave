from functools import singledispatch
from typing import Union

import numpy as np

from ._fastwave import AudioInfo
from ._fastwave import AudioData
from ._fastwave import ReadMode
from ._fastwave import info
from ._fastwave import read as _read


# Helpers to work with data convertions
# TODO: how to include it into "read" function and keep RO on "data"?
# Considerations:
# - should AudioInfo change accordingly?
# - is it better to move optimizations to numpy?
# - is overhead of Python a real problem here?
@singledispatch
def convert_data(audio: Union[AudioData, np.ndarray], *, mono: bool = False, dtype = None):
    raise NotImplemented("Unknown dispatch has been used!")


@convert_data.register(AudioData)
def _(audio: AudioData, *, mono: bool = False, dtype = None) -> np.ndarray:
    return convert_data(audio.data, mono=mono, dtype=dtype)


@convert_data.register(np.ndarray)
def _(audio: np.ndarray, *, mono: bool = False, dtype = None) -> np.ndarray:

    _data = audio.data

    if dtype is not None:
        # TODO: check if dtype is correct
        _data = _data.astype(dtype)
        if dtype == np.float32:
            _data = _data / 32767.
        else:
            raise NotImplemented(f"Conversion for {dtype} is not supported!")
    # Currently only two channels are supported:
    if mono and _data.shape[0] == 2:
        # Originally samples first, change to channels first
        _data = _data.T
        _data = (_data[0] + _data[1]) / 2

    return _data


def read(
    file_path,
    *,
    mode: ReadMode = ReadMode.DEFAULT,
    cache_size: int = 131072,  # in bytes
    num_threads: int = 8,  # TODO: make it more generic
) -> AudioData:
    # TODO: add one extra mode of "PYTHON"/"NATIVE"
    # This will not release the GIL but give native wrapper
    # Downside is that it needs some kind of heuristic first
    # to determine when this method is better.
    # import wave
    # import numpy as np
    # w = wave.open("./test.wav", "rb")
    # info = ... gather all info
    # data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # w.rewind() ?
    # w.close()
    # return data (+ info? in some structure)

    if cache_size < 1:
        raise ValueError("cache_size must be more than 0!")

    if num_threads < 1:
        raise ValueError("num_threads must be more than 0!")

    return _read(file_path, mode, cache_size, num_threads)
