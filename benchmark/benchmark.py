import os
import numpy as np
import datetime
import random
import string
import functools
import timeit

import fastwave
import torchaudio
from scipy.io import wavfile
import pydub
import wave


class AudioGenerator:
    # Duration is in seconds!
    def __init__(
        self, sample_rate=44100, duration=5, channels=2, prefix="random_audio_"
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.file_name = self.generate_random_name(prefix)
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        # Run generation at init!
        self.generate_scipy_audio()

    def generate_scipy_audio(self):
        if self.channels not in [1, 2]:
            raise RuntimeError("Unsupported number of channels!")

        noises = [
            np.random.normal(0, 1, int(self.sample_rate * self.duration))
            for _ in range(self.channels)
        ]
        audio_data = np.column_stack(noises) if self.channels == 2 else noises[0]
        scaled_audio_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(self.file_path, self.sample_rate, scaled_audio_data)

    def delete_generated_file(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            # print(f"Deleted file: {self.file_path}")

    def generate_random_name(self, prefix):
        current_datetime = datetime.datetime.now()
        random_suffix = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=3)
        )
        return (
            f"{prefix}{current_datetime.strftime('%Y%m%d%H%M%S')}_{random_suffix}.wav"
        )


def benchmark_fastwave_default(audio_generator):
    audio = fastwave.read(audio_generator.file_path, mode=fastwave.ReadMode.DEFAULT)
    # audio_data = fastwave.convert_data(audio.data, dtype=np.float32)
    audio_data = audio.data.astype("float32") / 32767.0
    return audio_data


def benchmark_fastwave_threads(audio_generator):
    audio = fastwave.read(
        audio_generator.file_path, mode=fastwave.ReadMode.THREADS, num_threads=6
    )
    # audio_data = fastwave.convert_data(audio.data, dtype=np.float32)
    audio_data = audio.data.astype("float32") / 32767.0
    return audio_data


def benchmark_fastwave_mmap_private(audio_generator):
    audio = fastwave.read(
        audio_generator.file_path, mode=fastwave.ReadMode.MMAP_PRIVATE
    )
    # audio_data = fastwave.convert_data(audio.data, dtype=np.float32)
    audio_data = audio.data.astype("float32") / 32767.0
    return audio_data


def benchmark_fastwave_mmap_shared(audio_generator):
    audio = fastwave.read(audio_generator.file_path, mode=fastwave.ReadMode.MMAP_SHARED)
    # audio_data = fastwave.convert_data(audio.data, dtype=np.float32)
    audio_data = audio.data.astype("float32") / 32767.0
    return audio_data


def benchmark_native_python(audio_generator):
    w = wave.open(audio_generator.file_path, "rb")
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    audio_data = audio.astype("float32") / 32767.0
    return audio_data


def benchmark_pydub(audio_generator):
    song = pydub.AudioSegment.from_file(audio_generator.file_path)
    sig = np.asarray(song.get_array_of_samples(), dtype="float32")
    sig = sig.reshape(song.channels, -1) / 32767.0
    return sig


def benchmark_torchaudio(audio_generator):
    sig, _ = torchaudio.load(
        audio_generator.file_path, normalize=True, channels_first=False
    )
    return sig


def benchmark_scipy_default(audio_generator):
    _, sig = wavfile.read(audio_generator.file_path)
    sig = sig.astype("float32") / 32767.0
    return sig


def benchmark_scipy_mmap(audio_generator):
    _, sig = wavfile.read(audio_generator.file_path, mmap=True)
    sig = sig.astype("float32") / 32767.0
    return sig


if __name__ == "__main__":
    audio_generator = AudioGenerator(sample_rate=44100, duration=30 * 10)
    # print(f"Generated file: {audio_generator.file_path}")

    ITERATIONS = 10
    REPS = 5

    methods = {
        "fastwave_DEFAULT": benchmark_fastwave_default,
        "fastwave_THREADS": benchmark_fastwave_threads,
        "fastwave_MMAP_PRIVATE": benchmark_fastwave_mmap_private,
        "fastwave_MMAP_SHARED": benchmark_fastwave_mmap_shared,
        "native_python": benchmark_native_python,
        "pydub": benchmark_pydub,
        "torchaudio": benchmark_torchaudio,
        "scipy_default": benchmark_scipy_default,
        "scipy_mmap": benchmark_scipy_mmap,
    }

    ITERATIONS = 10
    REPS = 5

    for method_name, method_func in methods.items():
        execution_time = timeit.repeat(
            functools.partial(method_func, audio_generator),
            number=ITERATIONS,
            repeat=REPS,
        )
        print(f"{method_name}: {min(execution_time)} seconds")

    audio_generator.delete_generated_file()