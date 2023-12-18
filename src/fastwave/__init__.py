# If used like when impl is also loading _fastwave:
# from ._fastwave import AudioInfo
# Throw errors on import:
# <frozen importlib._bootstrap>:241: RuntimeWarning: nanobind: type 'AudioInfo' was already registered!
# ...
# python(40176,0x1e3d85300) malloc: *** error for object 0x10441c458: pointer being freed was not allocated
# python(40176,0x1e3d85300) malloc: *** set a breakpoint in malloc_error_break to debug
# TODO: Create issue for it?
# pybind is trying to solve it with py::module_local()
# https://github.com/pybind/pybind11/blob/bf88e29c95d20a94528637c1827df757821b5a88/include/pybind11/pybind11.h#L749-L753

from .impl import AudioInfo
from .impl import AudioData
from .impl import ReadMode
from .impl import info
from .impl import read
