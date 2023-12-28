# fastwave

**fastwave** is a lightweight and efficient tool designed to accelerate the loading of audio files, with a primary focus on a specific type of WAVE files in PCM format. The project is implemented as a hybrid of C++ and Python, with a primary emphasis on Python bindings for ease of use.

## Features

### 1. Info Function

The `info` function provides detailed information about the loaded audio file. It returns an instance of the `AudioInfo` class, which includes essential details such as:

- Duration
- Sample rate
- Number of samples
- Number of channels
- Bit depth

### 2. Read Function

The `read` function loads the audio file and returns an instance of the `AudioData` class. This class has two properties:

- `data`: Contains the loaded audio file in NumPy format.
- `info`: Returns the previously mentioned `AudioInfo` class.

## Usage

Here's a simple example of how to use **fastwave** in Python:

```python
import fastwave

# Load audio file information
audio_info = fastwave.info("path/to/your/audio/file.wav")

# Print audio information
print("Duration:", audio_info.duration)
print("Sample Rate:", audio_info.sample_rate)
print("Number of Samples:", audio_info.num_samples)
print("Number of Channels:", audio_info.num_channels)
print("Bit Depth:", audio_info.bit_depth)

# Load audio data
audio_data = fastwave.read("path/to/your/audio/file.wav")

# Access loaded data and information
print("Loaded audio:", audio_data.data)
print("Audio duration:", audio_data.info.duration)
# ...
```

## Requirements

* Python (>=3.9)
* PDM

## Installation

To install **fastwave**, use the following commands:

```bash
# First clone the project from the repository.
git submodule update --init
pdm install --no-lock
```

## Installation

Contributions to fastwave are welcome! If you have any suggestions, feature requests, or bug reports, please create an issue or submit a pull request.

## License

*TBA*
