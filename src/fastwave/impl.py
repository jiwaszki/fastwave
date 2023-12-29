import numpy as np

from ._fastwave import AudioInfo
from ._fastwave import AudioData
from ._fastwave import ReadMode
from ._fastwave import info
from ._fastwave import read as _read


def read(
    file_path,
    mono: bool = True,
    dtype = None,
    mode: ReadMode = ReadMode.DEFAULT,
    *, 
    cache_size: int = 131072,  # in bytes
    num_threads: int = 8,  # TODO: make it more generic
) -> AudioData:
    # TODO: add one extra mode of "PYTHON"/"NATIVE"
    # This will not release the GIL but give native wrapper
    # import wave
    # import numpy as np
    # w = wave.open("./test.wav", "rb")
    # info = ... gather all info
    # data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # w.rewind() ?
    # w.close()
    # return data (+ info? in some structure)

    # return always 1D and reshape it and transpose accordingly?
    # TODO: benchmark, this can work better as numpy can speed-up comp. with multiple threads
    # data, info = _read(file_path, mode, num_threads)
    
    # # Originally samples first, change to channels first
    # if data.shape[0] != 1:
    #     data = data.T

    # if dtype is not None:
    #     # TODO: check if dtype is correct
    #     data = data.astype(dtype)
    #     if dtype == np.float32:
    #         data = data / 32767.

    # # Currently only two channels are supported:
    # if mono and data.shape[0] == 2:
    #     data = (data[0] + data[1]) / 2

    return _read(file_path, mode, cache_size, num_threads)
