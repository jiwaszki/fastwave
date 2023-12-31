#include <array>
#include <chrono>
#include <cstring> // std::memcpy
#include <iterator>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h> // include for mmap support
// includes for open and it's flags
#include <sys/stat.h>
#include <fcntl.h>
// #elif _WIN32
//     // windows code goes here
// #else
#endif

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
// #include "nanobind/stl/shared_ptr.h"

namespace nb = nanobind;

namespace fastwave
{
    // TODO: Code itself to be moved to different header file
    enum Endianness
    {
        ENDIAN_UNKNOWN,
        LittleEndian,
        BigEndian
    };

    enum ReadMode
    {
        // Modes using C++ file access:
        DEFAULT = 0,
        // Modes using C file access:
        THREADS = 1,
        MMAP_PRIVATE = 2,
        MMAP_SHARED = 3,
        // Last mode which is only used for info:
        INFO_ONLY = 4,
    };

    enum Format
    {
        FORMAT_UNKNOWN,
        PCM = 0x0001,
        IEEEFloat = 0x0003,
        ALaw = 0x0006,
        MULaw = 0x0007,
        Extensible = 0xFFFE
    };

    // TODO: make a template based on type and size
    inline std::array<uint8_t, 2> _read_chunk_2(std::ifstream &file)
    {
        std::array<uint8_t, 2> chunk;
        file.read((char *)&chunk[0], chunk.size());

        return chunk;
    }

    // TODO: make a template based on type and size
    inline std::array<uint8_t, 4> _read_chunk_4(std::ifstream &file)
    {
        std::array<uint8_t, 4> chunk;
        file.read((char *)&chunk[0], chunk.size());

        return chunk;
    }

    // TODO: make it a template based on type and size
    inline bool _compare_chunk_str(const std::array<uint8_t, 4> &chunk, const std::string_view &str)
    {
        return std::equal(chunk.begin(), chunk.end(), str.begin());
    }

    int32_t _unpack4(const std::array<uint8_t, 4> &chunk, Endianness endianness)
    {
        int32_t unpacked_value;

        if (endianness == Endianness::LittleEndian)
        {
            unpacked_value = (chunk[3] << 24) | (chunk[2] << 16) | (chunk[1] << 8) | chunk[0];
        }
        else
        { // endianness == Endianness::BigEndian
            unpacked_value = (chunk[0] << 24) | (chunk[1] << 16) | (chunk[2] << 8) | chunk[3];
        }

        return unpacked_value;
    }

    int16_t _unpack2(const std::array<uint8_t, 2> &chunk, Endianness endianness)
    {
        int16_t unpacked_value;

        if (endianness == Endianness::LittleEndian)
        {
            unpacked_value = (chunk[1] << 8) | chunk[0];
        }
        else
        { // endianness == Endianness::BigEndian
            unpacked_value = (chunk[0] << 8) | chunk[1];
        }

        return unpacked_value;
    }

    class AudioInfo
    {
    public:
        AudioInfo() = default;

        // Copy constructor
        AudioInfo(const AudioInfo& other)
            : duration(other.duration),
            num_samples(other.num_samples),
            num_channels(other.num_channels),
            sample_rate(other.sample_rate),
            bit_depth(other.bit_depth),
            endianness(other.endianness),
            format(other.format),
            bytes_per_sample(other.bytes_per_sample),
            data_offset(other.data_offset) {}

        // Copy assignment operator
        AudioInfo& operator=(const AudioInfo& other) {
            if (this != &other) {
                // Copy member variables
                duration = other.duration;
                num_samples = other.num_samples;
                num_channels = other.num_channels;
                sample_rate = other.sample_rate;
                bit_depth = other.bit_depth;
                endianness = other.endianness;
                format = other.format;
                bytes_per_sample = other.bytes_per_sample;
                data_offset = other.data_offset;
            }
            return *this;
        }

        AudioInfo(AudioInfo&& other) noexcept
                : duration(std::exchange(other.duration, 0.0)),
                num_samples(std::exchange(other.num_samples, 0)),
                num_channels(std::exchange(other.num_channels, 0)),
                sample_rate(std::exchange(other.sample_rate, 0)),
                bit_depth(std::exchange(other.bit_depth, 0)),
                endianness(std::exchange(other.endianness, Endianness::ENDIAN_UNKNOWN)),
                format(std::exchange(other.format, Format::FORMAT_UNKNOWN)),
                bytes_per_sample(std::exchange(other.bytes_per_sample, 0)),
                data_offset(std::exchange(other.data_offset, 0)) {}

        // Move assignment operator
        AudioInfo& operator=(AudioInfo&& other) noexcept {
            if (this != &other) {
                // Move member variables
                duration = std::exchange(other.duration, 0.0);
                num_samples = std::exchange(other.num_samples, 0);
                num_channels = std::exchange(other.num_channels, 0);
                sample_rate = std::exchange(other.sample_rate, 0);
                bit_depth = std::exchange(other.bit_depth, 0);
                endianness = std::exchange(other.endianness, Endianness::ENDIAN_UNKNOWN);
                format = std::exchange(other.format, Format::FORMAT_UNKNOWN);
                bytes_per_sample = std::exchange(other.bytes_per_sample, 0);
                data_offset = std::exchange(other.data_offset, 0);
            }
            return *this;
        }

        AudioInfo(std::ifstream &file)
        {
            read(file);
        }
    
        // To expose in Python:
        double duration;
        size_t num_samples;  // TODO: change to size_t for example?
        size_t num_channels; // TODO: change to size_t for example?
        size_t sample_rate;
        size_t bit_depth;
        // TODO: consider exposing in Python: if not hide from C++ "users" of this class?
        Endianness endianness;
        Format format;
        size_t bytes_per_sample;
        size_t data_offset = 0; // data offest in bytes

    private:
        void read(std::ifstream &file)
        {
            // Read RIFF chunk - first 4 bytes
            auto signature = _read_chunk_4(file);

            if (_compare_chunk_str(signature, "RIFF"))
            {
                endianness = Endianness::LittleEndian;
            }
            else if (_compare_chunk_str(signature, "RIFX"))
            {
                endianness = Endianness::BigEndian;
            }
            else
            {
                throw std::runtime_error("Invalid file format! Only 'RIFF' and 'RIFX' supported.");
            }

            // Read size of the file
            // TODO: Is it needed? or can it just std::advance(it, 4)? Apply to all similar is_PCM etc.
            auto file_size = _unpack4(_read_chunk_4(file), endianness) + 8;

            // Check if WAVE file
            auto is_wave = _read_chunk_4(file);
            if (!_compare_chunk_str(is_wave, "WAVE"))
            {
                throw std::runtime_error("Invalid file format! Not a WAV file!");
            }

            data_offset += 12; // increment data_offset for mmap reads

            while (file.good())
            {
                auto chunk = _read_chunk_4(file);
                data_offset += 4; // increment data_offset for mmap reads

                if (_compare_chunk_str(chunk, "fmt "))
                {
                    auto fmt_size = _unpack4(_read_chunk_4(file), endianness);
                    if (fmt_size != 16)
                    {
                        throw std::runtime_error("Invalid fmt_size!");
                    }
                    format = static_cast<Format>(_unpack2(_read_chunk_2(file), endianness)); // if equal to 1 it is PCM
                    if (format != Format::PCM)
                    {
                        throw std::runtime_error("Only PCM format is currently supported.");
                    }
                    num_channels = _unpack2(_read_chunk_2(file), endianness);          // channels
                    sample_rate = _unpack4(_read_chunk_4(file), endianness);           // sampleRate
                    auto bytes_per_second = _unpack4(_read_chunk_4(file), endianness); // bytesPerSecond
                    bytes_per_sample = _unpack2(_read_chunk_2(file), endianness);      // bytesPerSample
                    bit_depth = _unpack2(_read_chunk_2(file), endianness);             // bitsPerSample
                    // Are these two always after the
                    auto data_marker = _read_chunk_4(file); // "data"
                    if (!_compare_chunk_str(data_marker, "data"))
                    {
                        throw std::runtime_error("Cannot find 'data' marker!");
                    }
                    auto data_size = _unpack4(_read_chunk_4(file), endianness); // dataSize

                    // Calculate number of samples and duration:
                    num_samples = data_size / (num_channels * bit_depth / 8);
                    duration = (double)num_samples / (double)sample_rate;
                    data_offset += 28; // increment data_offset for mmap reads
                    break;
                }
                // Should skip other chunks like scipy: _compare_chunk_str(chunk, "JUNK") and move move?
            }

            return;
        }
    };

    class AudioData
    {
    public:
        AudioData() {
            _buffer = reinterpret_cast<char *>(malloc(0));
            is_mmap = false;
        };

        // All dispatches using ifstream:
        AudioData(
            std::ifstream &file,
            AudioInfo info,
            fastwave::ReadMode read_mode) :
            info(std::move(info)),
            is_mmap(false)
        {
            allocate_buffer();

            switch (read_mode)
            {
            case fastwave::ReadMode::DEFAULT:
                read_ifstream(file);
                break;
            default:
                throw std::runtime_error("Unknown read mode!");
            }
        }

        // All dispatches using low-level C file access:
        AudioData(
            const std::string &file_path,
            AudioInfo info,
            fastwave::ReadMode read_mode,
            size_t cache_size,
            size_t num_threads) :
            info(std::move(info)),
            is_mmap(false)
        {
            allocate_buffer();
            // Dispatch reading of the data:
            switch (read_mode)
            {
            case fastwave::ReadMode::THREADS:
                read_threads(file_path, cache_size, num_threads);
                break;
            case fastwave::ReadMode::MMAP_PRIVATE:
                read_mmap(file_path, false);
                break;
            case fastwave::ReadMode::MMAP_SHARED:
                read_mmap(file_path, true);
                break;
            default:
                throw std::runtime_error("Unknown read mode!");
            }
        }

        AudioData(AudioData&& other) noexcept
            : info(std::move(other.info)),
            _buffer(std::exchange(other._buffer, nullptr)),
            _buffer_size(std::exchange(other._buffer_size, 0)),
            is_mmap(std::exchange(other.is_mmap, false)) {}

        ~AudioData() {
            deallocate_buffer();
        };

        AudioData& operator=(const AudioData& other) {
            if (this != &other) {
                // Release resources held by the current object
                deallocate_buffer();

                info = other.info;
                _buffer_size = other._buffer_size;
                is_mmap = other.is_mmap;

                allocate_buffer();
                std::memcpy(_buffer, other._buffer, _buffer_size);
            }
            return *this;
        }

        // Move assignment operator
        AudioData& operator=(AudioData&& other) noexcept {
            if (this != &other) {
                // Release resources held by the current object
                deallocate_buffer();

                // Transfer ownership
                info = std::move(other.info);
                _buffer = std::exchange(other._buffer, nullptr);
                _buffer_size = std::exchange(other._buffer_size, 0);
                is_mmap = std::exchange(other.is_mmap, false);
            }
            return *this;
        }

        AudioInfo info;  // TODO: refactor, to not make a copy, reference already existing one
        char* _buffer;
        size_t _buffer_size;
        bool is_mmap;

    private:
        inline void allocate_buffer()
        {
            _buffer_size = info.num_channels * info.num_samples * info.bytes_per_sample;
            _buffer = reinterpret_cast<char *>(malloc(info.num_channels * info.num_samples * info.bytes_per_sample));
        }

        inline void deallocate_buffer()
        {
            if (_buffer != nullptr) {
                if (is_mmap) {
#if defined(__linux__) || defined(__APPLE__)
                    munmap(reinterpret_cast<void *>(_buffer), _buffer_size);
#else
                    throw std::runtime_error("MUNMAP is not supported on this platform!");
#endif
                }
                else {
                    free(_buffer);
                }
                _buffer = nullptr;
            }
        }

        void read_mmap(const std::string &file_path, bool is_shared)
        {
#if defined(__linux__) || defined(__APPLE__)
            // Allow MAP_SHARED and MAP_PRIVATE
            // https://man7.org/linux/man-pages/man2/mmap.2.html
            // Unmap the memory if it was previously mapped
            if (is_mmap && _buffer != nullptr) {
                munmap(reinterpret_cast<void *>(_buffer), _buffer_size);
                _buffer = nullptr;
                is_mmap = false;
            }
            int file = open(file_path.c_str(), O_RDONLY);
#ifdef __linux__
            _buffer = reinterpret_cast<char *>(reinterpret_cast<uint8_t *>(mmap(NULL, _buffer_size, PROT_READ, ((is_shared) ? MAP_SHARED : MAP_PRIVATE) | MAP_POPULATE, file, 0)) + info.data_offset);
#else
            _buffer = reinterpret_cast<char *>(reinterpret_cast<uint8_t *>(mmap(NULL, _buffer_size, PROT_READ, (is_shared) ? MAP_SHARED : MAP_PRIVATE, file, 0)) + info.data_offset);
#endif
            close(file);
            // Update mmap flag:
            is_mmap = true;
            return;
#else
            throw std::runtime_error("MMAP is not supported on this platform!");
#endif
        }

        void read_threads(const std::string &file_path, const size_t cache_size, const size_t num_threads)
        {
            // Calculate the number of chunks based on chunk_size
            size_t chunk_size = cache_size << 7;
            size_t num_chunks = (_buffer_size + chunk_size - 1) / chunk_size;

            // Calculate the number of chunks per thread
            size_t chunks_per_thread = num_chunks / num_threads;
            size_t remaining_chunks = num_chunks % num_threads;

            // Vector to store threads
            std::vector<std::thread> threads;
            threads.reserve(num_threads);

            // Function to load a range of chunks in a thread
            auto load_range = [&](std::string path, size_t start_chunk, size_t end_chunk) {
                // Calculate the starting position for this thread's range
                size_t start_position = start_chunk * chunk_size;

                // Calculate the size of this thread's range
                size_t range_size = (end_chunk - start_chunk) * chunk_size;

                // Allocate a local buffer for each thread
                char* mybuffer = reinterpret_cast<char*>(malloc(cache_size));

                // Open a local file stream for each thread
                std::ifstream local_file;
                local_file.rdbuf()->pubsetbuf(mybuffer, cache_size);
                local_file.open(path, std::ios::binary | std::ios::in);

                if (!local_file.is_open()) {
                    std::cerr << "Error opening file: " << path << std::endl;
                }

                local_file.seekg(info.data_offset + start_position);
                local_file.read(_buffer + start_position, range_size);
                free(mybuffer);
            };

            // Launch threads to load data ranges in parallel
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start_chunk = i * chunks_per_thread;
                size_t end_chunk = start_chunk + chunks_per_thread + ((i == num_threads - 1) ? remaining_chunks : 0);
                threads.emplace_back(load_range, file_path, start_chunk, end_chunk);
            }

            // Wait for all threads to finish
            for (auto& thread : threads) {
                thread.join();
            }
        }

        inline void read_ifstream(std::ifstream &file)
        {
            file.read(_buffer, _buffer_size);
        }
    };
} // namespace fastwave

// Bindings

NB_MODULE(_fastwave, m)
{
    nb::class_<fastwave::AudioInfo> audioinfo(m, "AudioInfo");

    audioinfo.def_ro("duration", &fastwave::AudioInfo::duration);
    audioinfo.def_ro("num_samples", &fastwave::AudioInfo::num_samples);
    audioinfo.def_ro("num_channels", &fastwave::AudioInfo::num_channels);
    audioinfo.def_ro("sample_rate", &fastwave::AudioInfo::sample_rate);
    audioinfo.def_ro("bit_depth", &fastwave::AudioInfo::bit_depth);

    nb::class_<fastwave::AudioData> audiodata(m, "AudioData");

    audiodata.def_prop_ro("info", [](fastwave::AudioData& self) {
        return self.info;
    });

    audiodata.def_prop_ro("data", [](fastwave::AudioData& self) {
        const size_t channels = self.info.num_channels;
        const size_t samples = self.info.num_samples;

        // Early return if data is mono:
        if (channels == 1) {
            // TODO: is std::move ok?
            // return nb::ndarray<nb::numpy, int16_t>(std::move(self._buffer), {samples});
            return nb::ndarray<nb::numpy, int16_t>(self._buffer, {samples});
        }
        // Only other supported case - 2 channels:
        // TODO: is std::move ok?
        // return nb::ndarray<nb::numpy, int16_t>(std::move(self._buffer), {samples, channels});
        return nb::ndarray<nb::numpy, int16_t>(self._buffer, {samples, channels});
    }); // , nb::rv_policy::reference_internal); ??

    nb::enum_<fastwave::ReadMode>(m, "ReadMode")
        .value("DEFAULT", fastwave::ReadMode::DEFAULT)
        .value("THREADS", fastwave::ReadMode::THREADS)
        .value("MMAP_PRIVATE", fastwave::ReadMode::MMAP_PRIVATE)
        .value("MMAP_SHARED", fastwave::ReadMode::MMAP_SHARED)
        .export_values();

    // Return info only:
    m.def(
        "info", [](const std::string &file_path)
        {
            std::ifstream file;
            file.open(file_path, std::ios::binary | std::ios::in);
            auto info = fastwave::AudioInfo(file);
            file.close();
            return info;
        },
        nb::call_guard<nb::gil_scoped_release>());

    // TODO: think if read-->load is better name
    // TODO: refactor num_threads to different overload?
    m.def(
        "read", [](
            const std::string &file_path,
            fastwave::ReadMode read_mode,
            size_t cache_size,
            size_t num_threads)
        {
            // std::ios::sync_with_stdio(false);  // ???
            // std::cin.tie(0);  // ???

            // Open the file:
            std::ifstream file;

            // Adjust internal buffer of the stream:
            char* mybuffer = reinterpret_cast<char *>(malloc(cache_size));
            file.rdbuf()->pubsetbuf(mybuffer, cache_size);

            file.open(file_path, std::ios::binary | std::ios::in);

            // Always read a header section of the file:
            auto info = fastwave::AudioInfo(file);
            auto data = fastwave::AudioData();

            // Return only information
            if (read_mode == fastwave::ReadMode::INFO_ONLY)
            {
                file.close();
            }
            else {
                // Continue reading the rest of the file:
                switch (read_mode)
                {
                case fastwave::ReadMode::DEFAULT:
                    // Pass down already opened stream, close it afterwards.
                    data = fastwave::AudioData(file, info, read_mode);
                    file.close();
                    break;
                case fastwave::ReadMode::THREADS:
                case fastwave::ReadMode::MMAP_PRIVATE:
                case fastwave::ReadMode::MMAP_SHARED:
                    // Close the file before accessing it in C methods.
                    file.close();
                    data = fastwave::AudioData(file_path, info, read_mode, cache_size, num_threads);
                    break;
                default:
                    file.close();
                    throw std::runtime_error("Unknown read mode!");
                }
            }

            // std::ios::sync_with_stdio(true);  // ???
            std::free(mybuffer);

            // std::cout << "Endian:" << audio.info.endianness << "\n";
            const size_t channels = info.num_channels;

            if (channels == 0 || channels > 2) {
                throw std::runtime_error("Invalid number of channels!");
            }

            return data;
        }, nb::call_guard<nb::gil_scoped_release>());
}
