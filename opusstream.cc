#include <iostream>
#include "opusenc.h"
#include "opusfile.h"
#include "opus.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <memory>
#include <algorithm>
#include <mutex>

#ifdef _WIN32
    #include <winsock2.h>
#else
    #include <arpa/inet.h>
#endif

namespace py = pybind11;

// Supported Opus sample rates
const std::vector<int> OPUS_VALID_RATES = {8000, 12000, 16000, 24000, 48000};

template<typename T>
py::array_t<T> MakeNpArray(std::vector<ssize_t> shape, T* data) {
    // Fix: Use C-contiguous order (row-major) instead of Fortran order
    std::vector<ssize_t> strides(shape.size());
    if (shape.size() == 2) {
        // For 2D arrays: [time, channels]
        strides[1] = sizeof(T);                    // stride for channels
        strides[0] = shape[1] * sizeof(T);         // stride for time
    } else if (shape.size() == 1) {
        strides[0] = sizeof(T);
    } else {
        // General case for N-dimensional arrays (C-contiguous)
        size_t v = sizeof(T);
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = v;
            v *= shape[i];
        }
    }
    
    py::capsule free_when_done(data, [](void* f) {
        auto* foo = reinterpret_cast<T*>(f);
        delete[] foo;
    });
    return py::array_t<T>(shape, strides, data, free_when_done);
}

struct MemoryBuffer {
    std::vector<uint8_t> data;
    size_t position;
    
    MemoryBuffer() : position(0) {}
    
    void reserve(size_t size) {
        data.reserve(size);
    }
};

static int memory_write(void *user_data, const unsigned char *ptr, opus_int32 len) {
    MemoryBuffer* buffer = static_cast<MemoryBuffer*>(user_data);
    buffer->data.insert(buffer->data.end(), ptr, ptr + len);
    return 0; // Success
}

static int memory_close(void *user_data) {
    return 0;
}

// opus decoding from memory
static int memory_read(void *stream, unsigned char *ptr, int nbytes) {
    MemoryBuffer* buffer = static_cast<MemoryBuffer*>(stream);
    size_t available = buffer->data.size() - buffer->position;
    size_t to_read = std::min(static_cast<size_t>(nbytes), available);
    
    if (to_read > 0) {
        std::memcpy(ptr, buffer->data.data() + buffer->position, to_read);
        buffer->position += to_read;
    }
    
    return static_cast<int>(to_read);
}

static int memory_seek(void *stream, opus_int64 offset, int whence) {
    MemoryBuffer* buffer = static_cast<MemoryBuffer*>(stream);
    
    size_t new_position;
    switch (whence) {
        case SEEK_SET:
            new_position = offset;
            break;
        case SEEK_CUR:
            new_position = buffer->position + offset;
            break;
        case SEEK_END:
            new_position = buffer->data.size() + offset;
            break;
        default:
            return -1;
    }
    
    if (new_position > buffer->data.size()) {
        return -1;
    }
    
    buffer->position = new_position;
    return 0;
}

static opus_int64 memory_tell(void *stream) {
    MemoryBuffer* buffer = static_cast<MemoryBuffer*>(stream);
    return buffer->position;
}

py::array_t<opus_int16> float32_to_int16(const py::array_t<float>& input) {
    auto* out_data = new opus_int16[input.size()];
    const float* in_ptr = input.data();
    
    for (ssize_t i = 0; i < input.size(); ++i) {
        // Fix: Manual clamp for C++11/14 compatibility
        float sample = in_ptr[i];
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        // Scale to int16 range
        sample *= 32767.0f;
        out_data[i] = static_cast<opus_int16>(sample);
    }
    
    return MakeNpArray<opus_int16>({input.shape(0), input.shape(1)}, out_data);
}

py::array_t<float> int16_to_float32(const py::array_t<opus_int16>& input) {
    auto* out_data = new float[input.size()];
    const opus_int16* in_ptr = input.data();
    
    for (ssize_t i = 0; i < input.size(); ++i) {
        // Proper scaling to [-1.0, 1.0] range
        out_data[i] = in_ptr[i] / 32768.0f;
    }
    
    return MakeNpArray<float>({input.shape(0), input.shape(1)}, out_data);
}

// PCM to Opus encoding in memory with Ogg container
py::bytes OpusEncodeMemory(const py::array_t<int16_t>& waveform_tc, 
                          const int sample_rate, 
                          const int bitrate = OPUS_AUTO, 
                          const int signal_type = 0, 
                          const int encoder_complexity = 10) {
    
    if (waveform_tc.ndim() != 2) {
        throw py::value_error("waveform_tc must have exactly 2 dimensions: [time, channels].");
    }
    if (waveform_tc.shape(1) > 8 || waveform_tc.shape(1) < 1) {
        throw py::value_error("waveform_tc must have at least 1 channel, and no more than 8.");
    }
    if ((bitrate < 500 || bitrate > 512000) && bitrate != OPUS_BITRATE_MAX && bitrate != OPUS_AUTO) {
        throw py::value_error("Invalid bitrate, must be at least 512 and at most 512k bits/s.");
    }
    if (sample_rate < 8000 || sample_rate > 48000) {
        throw py::value_error("Invalid sample_rate, must be at least 8k and at most 48k.");
    }
    if (encoder_complexity > 10 || encoder_complexity < 0) {
        throw py::value_error("Invalid encoder_complexity, must be in range [0, 10] inclusive.");
    }
    
    opus_int32 opus_signal_type;
    switch (signal_type) {
        case 0:
            opus_signal_type = OPUS_AUTO;
            break;
        case 1:
            opus_signal_type = OPUS_SIGNAL_MUSIC;
            break;
        case 2:
            opus_signal_type = OPUS_SIGNAL_VOICE;
            break;
        default:
            throw py::value_error("Invalid signal type, must be 0 (auto), 1 (music) or 2 (voice).");
    }
    
    // pre allocated
    MemoryBuffer buffer;
    buffer.reserve(waveform_tc.size() * sizeof(int16_t) / 10);
    
    OpusEncCallbacks callbacks;
    callbacks.write = memory_write;
    callbacks.close = memory_close;
    
    OggOpusComments* comments = ope_comments_create();
    
    int error;
    OggOpusEnc* enc = ope_encoder_create_callbacks(&callbacks, &buffer, comments, 
                                                   sample_rate, waveform_tc.shape(1), 0, &error);
    if (error != 0) {
        ope_comments_destroy(comments);
        throw py::value_error("Failed to create encoder.");
    }
    
    if (ope_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate) != 0) {
        ope_encoder_destroy(enc);
        ope_comments_destroy(comments);
        throw py::value_error("Could not set bitrate.");
    }
    
    if (ope_encoder_ctl(enc, OPUS_SET_SIGNAL_REQUEST, opus_signal_type) != 0) {
        ope_encoder_destroy(enc);
        ope_comments_destroy(comments);
        throw py::value_error("Could not set signal type.");
    }
    
    if (ope_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, encoder_complexity) != 0) {
        ope_encoder_destroy(enc);
        ope_comments_destroy(comments);
        throw py::value_error("Could not set encoder complexity.");
    }
    
    // PCM data
    if (ope_encoder_write(enc, waveform_tc.data(), waveform_tc.shape(0)) != 0) {
        ope_encoder_destroy(enc);
        ope_comments_destroy(comments);
        throw py::value_error("Could not encode audio data.");
    }
    
    if (ope_encoder_drain(enc) != 0) {
        ope_encoder_destroy(enc);
        ope_comments_destroy(comments);
        throw py::value_error("Could not finalize encoding.");
    }
    
    // Cleaning
    ope_encoder_destroy(enc);
    ope_comments_destroy(comments);
    
    return py::bytes(reinterpret_cast<char*>(buffer.data.data()), buffer.data.size());
}

// Decode Opus to PCM from memory
std::tuple<py::array_t<opus_int16>, int> OpusDecodeMemory(const py::bytes& opus_data) {
    std::string data_str = opus_data;
    
    // memory buffer
    MemoryBuffer buffer;
    buffer.data.assign(data_str.begin(), data_str.end());
    buffer.position = 0;
    
    OpusFileCallbacks callbacks;
    callbacks.read = memory_read;
    callbacks.seek = memory_seek;
    callbacks.tell = memory_tell;
    callbacks.close = nullptr;
    
    int error;
    OggOpusFile* file = op_open_callbacks(&buffer, &callbacks, nullptr, 0, &error);
    if (error != 0) {
        throw py::value_error("Could not open opus data from memory.");
    }
    
    const ssize_t num_chans = op_channel_count(file, -1);
    const ssize_t num_samples = op_pcm_total(file, -1);
    
    const OpusHead* meta = op_head(file, -1);
    const int sample_rate = meta->input_sample_rate;
    
    auto* data = new opus_int16[num_samples * num_chans];
    std::fill(data, data + num_samples * num_chans, 0);
    
    auto waveform_tc = MakeNpArray<opus_int16>({num_samples, num_chans}, data);
    
    size_t num_read = 0;
    
    while (true) {
        int chunk = op_read(file, data + num_read * num_chans, 
                           (num_samples - num_read) * num_chans, nullptr);
        if (chunk < 0) {
            op_free(file);
            throw py::value_error("Could not decode opus data.");
        }
        if (chunk == 0) {
            break;
        }
        num_read += chunk;
    }
    
    op_free(file);
    
    return std::make_tuple(std::move(waveform_tc), sample_rate);
}

// Direct Opus API
// Packet format: [2-byte length big-endian][opus data][2-byte length][opus data]...
py::bytes OpusEncodeRaw(const py::array_t<int16_t>& waveform_tc,
                        const int sample_rate,
                        const int bitrate = OPUS_AUTO,
                        const int signal_type = 0,
                        const int encoder_complexity = 10,
                        const int frame_duration_ms = 20) {
    
    if (waveform_tc.ndim() != 2) {
        throw py::value_error("waveform_tc must have exactly 2 dimensions: [time, channels].");
    }
    
    int channels = waveform_tc.shape(1);
    if (channels != 1 && channels != 2) {
        throw py::value_error("Raw Opus API only supports 1 or 2 channels.");
    }
    
    // Opus supports 2.5, 5, 10, 20, 40, 60ms frame times
    if (frame_duration_ms != 5 && frame_duration_ms != 10 && 
        frame_duration_ms != 20 && frame_duration_ms != 40 && frame_duration_ms != 60) {
        throw py::value_error("Frame duration must be 5, 10, 20, 40, or 60 ms.");
    }
    
    int error;
    OpusEncoder* encoder = opus_encoder_create(sample_rate, channels, 
                                              signal_type == 2 ? OPUS_APPLICATION_VOIP : 
                                              signal_type == 1 ? OPUS_APPLICATION_AUDIO : 
                                              OPUS_APPLICATION_AUDIO, &error);
    if (error != OPUS_OK) {
        throw py::value_error("Failed to create Opus encoder.");
    }
    
    opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
    opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(encoder_complexity));
    
    // pre allocate output buffer
    std::vector<uint8_t> output;
    output.reserve(waveform_tc.size() * sizeof(int16_t) / 10);
    
    // frame processing
    int frame_size = sample_rate * frame_duration_ms / 1000;
    size_t max_packet_size = 4000; // Maximum Opus packet size
    std::vector<unsigned char> packet(max_packet_size);
    
    const int16_t* input_ptr = waveform_tc.data();
    size_t total_samples = waveform_tc.shape(0);
    
    for (size_t i = 0; i < total_samples; i += frame_size) {
        size_t remaining = total_samples - i;
        size_t current_frame_size = std::min(static_cast<size_t>(frame_size), remaining);
        
        // frame encoding
        int packet_len = opus_encode(encoder, input_ptr + i * channels, 
                                   current_frame_size, packet.data(), max_packet_size);
        
        if (packet_len < 0) {
            opus_encoder_destroy(encoder);
            throw py::value_error("Encoding failed.");
        }
        
        uint16_t be_len = htons(static_cast<uint16_t>(packet_len));
        output.push_back((be_len >> 8) & 0xFF);
        output.push_back(be_len & 0xFF);
        output.insert(output.end(), packet.begin(), packet.begin() + packet_len);
    }
    
    opus_encoder_destroy(encoder);
    
    return py::bytes(reinterpret_cast<char*>(output.data()), output.size());
}

std::tuple<py::array_t<opus_int16>, int> OpusDecodeRaw(const py::bytes& opus_data,
                                                       int sample_rate,
                                                       int channels) {
    if (channels != 1 && channels != 2) {
        throw py::value_error("Raw Opus API only supports 1 or 2 channels.");
    }
    
    // Sample rate validation in one of opus supperted rates
    if (std::find(OPUS_VALID_RATES.begin(), OPUS_VALID_RATES.end(), sample_rate) == OPUS_VALID_RATES.end()) {
        throw py::value_error("Sample rate must be 8000, 12000, 16000, 24000, or 48000 Hz");
    }
    
    std::string data_str = opus_data;
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data_str.data());
    size_t data_size = data_str.size();
    
    int error;
    OpusDecoder* decoder = opus_decoder_create(sample_rate, channels, &error);
    if (error != OPUS_OK) {
        throw py::value_error("Failed to create Opus decoder.");
    }
    
    std::vector<opus_int16> output;
    output.reserve(data_size * 10); // Rough estimate
    
    int max_frame_size = 5760; // 120ms at 48kHz
    std::vector<opus_int16> pcm(max_frame_size * channels);
    
    size_t pos = 0;
    while (pos + 2 <= data_size) {
        // Read packet length (big-endian)
        uint16_t be_len = (data_ptr[pos] << 8) | data_ptr[pos + 1];
        uint16_t packet_len = ntohs(be_len);
        pos += 2;
        
        if (pos + packet_len > data_size) {
            opus_decoder_destroy(decoder);
            throw py::value_error("Invalid packet structure: packet extends beyond data.");
        }
        
        int samples = opus_decode(decoder, data_ptr + pos, packet_len, 
                                pcm.data(), max_frame_size, 0);
        
        if (samples < 0) {
            opus_decoder_destroy(decoder);
            throw py::value_error("Decoding failed.");
        }
        
        output.insert(output.end(), pcm.begin(), pcm.begin() + samples * channels);
        pos += packet_len;
    }
    
    opus_decoder_destroy(decoder);
    
    size_t num_samples = output.size() / channels;
    auto* data = new opus_int16[output.size()];
    std::copy(output.begin(), output.end(), data);
    
    auto waveform_tc = MakeNpArray<opus_int16>({static_cast<ssize_t>(num_samples), 
                                               static_cast<ssize_t>(channels)}, data);
    
    return std::make_tuple(std::move(waveform_tc), sample_rate);
}

// Main streaming encoder class
class OpusStreamEncoder {
private:
    OpusEncoder* encoder;
    int sample_rate;
    int channels;
    int frame_size;
    std::vector<opus_int16> buffer;
    size_t buffer_pos;
    std::vector<uint8_t> output_buffer;
    mutable std::mutex encoder_mutex;  // Thread safety
    
public:
    OpusStreamEncoder(int sample_rate, int channels, int bitrate = OPUS_AUTO,
                     int signal_type = 0, int complexity = 10, int frame_duration_ms = 20)
        : sample_rate(sample_rate), channels(channels), buffer_pos(0) {
        
        if (channels != 1 && channels != 2) {
            throw py::value_error("Only 1 or 2 channels supported for streaming.");
        }
        
        if (std::find(OPUS_VALID_RATES.begin(), OPUS_VALID_RATES.end(), sample_rate) == OPUS_VALID_RATES.end()) {
            throw py::value_error("Invalid sample rate for Opus.");
        }
        
        int error;
        int application = (signal_type == 2) ? OPUS_APPLICATION_VOIP :
                         (signal_type == 1) ? OPUS_APPLICATION_AUDIO :
                         OPUS_APPLICATION_AUDIO;
        
        encoder = opus_encoder_create(sample_rate, channels, application, &error);
        if (error != OPUS_OK) {
            throw py::value_error("Failed to create encoder.");
        }
        
        opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
        opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(complexity));
        opus_encoder_ctl(encoder, OPUS_SET_INBAND_FEC(1));
        
        frame_size = sample_rate * frame_duration_ms / 1000;
        buffer.resize(frame_size * channels);
        output_buffer.resize(4000); // Max Opus packet size
    }
    
    ~OpusStreamEncoder() {
        if (encoder) {
            opus_encoder_destroy(encoder);
        }
    }
    
    py::bytes encode(const py::array_t<opus_int16>& samples) {
        std::lock_guard<std::mutex> lock(encoder_mutex);
        
        if (samples.ndim() != 2 || samples.shape(1) != channels) {
            throw py::value_error("Input must have shape [samples, channels]");
        }
        
        const opus_int16* input = samples.data();
        size_t input_samples = samples.shape(0);
        size_t input_pos = 0;
        
        std::vector<uint8_t> encoded_packets;
        
        while (input_pos < input_samples) {
            size_t to_copy = std::min(frame_size - buffer_pos, input_samples - input_pos);
            std::memcpy(buffer.data() + buffer_pos * channels,
                       input + input_pos * channels,
                       to_copy * channels * sizeof(opus_int16));
            
            buffer_pos += to_copy;
            input_pos += to_copy;
            
            // If buffer is full we encode
            if (buffer_pos >= static_cast<size_t>(frame_size)) {
                int packet_len = opus_encode(encoder, buffer.data(), frame_size,
                                           output_buffer.data(), output_buffer.size());
                
                if (packet_len < 0) {
                    throw py::value_error("Encoding failed");
                }
                
                // add length prefix
                uint16_t be_len = htons(static_cast<uint16_t>(packet_len));
                encoded_packets.push_back((be_len >> 8) & 0xFF);
                encoded_packets.push_back(be_len & 0xFF);
                encoded_packets.insert(encoded_packets.end(),
                                     output_buffer.begin(),
                                     output_buffer.begin() + packet_len);
                
                buffer_pos = 0;
            }
        }
        
        return py::bytes(reinterpret_cast<char*>(encoded_packets.data()),
                        encoded_packets.size());
    }
    
    // Optionally we flush remaining samples (padding)
    py::bytes flush(bool encode_silence = true) {
        std::lock_guard<std::mutex> lock(encoder_mutex);
        
        if (buffer_pos == 0 || !encode_silence) {
            buffer_pos = 0;  // Reset buffer even if not encoding
            return py::bytes();
        }
        
        // zero padding
        std::fill(buffer.begin() + buffer_pos * channels, buffer.end(), 0);
        
        int packet_len = opus_encode(encoder, buffer.data(), frame_size,
                                   output_buffer.data(), output_buffer.size());
        
        if (packet_len < 0) {
            throw py::value_error("Encoding failed");
        }
        
        std::vector<uint8_t> result;
        uint16_t be_len = htons(static_cast<uint16_t>(packet_len));
        result.push_back((be_len >> 8) & 0xFF);
        result.push_back(be_len & 0xFF);
        result.insert(result.end(), output_buffer.begin(),
                     output_buffer.begin() + packet_len);
        
        buffer_pos = 0;
        
        return py::bytes(reinterpret_cast<char*>(result.data()), result.size());
    }
    
    // Reset encoder state
    void reset() {
        std::lock_guard<std::mutex> lock(encoder_mutex);
        opus_encoder_ctl(encoder, OPUS_RESET_STATE);
        buffer_pos = 0;
    }
    
    int get_frame_size() const { return frame_size; }
};

// Main streaming decoder class
class OpusStreamDecoder {
private:
    OpusDecoder* decoder;
    int sample_rate;
    int channels;
    std::vector<uint8_t> packet_buffer;
    std::vector<opus_int16> pcm_buffer;
    mutable std::mutex decoder_mutex;  // Thread safety
    
public:
    OpusStreamDecoder(int sample_rate, int channels)
        : sample_rate(sample_rate), channels(channels) {
        
        if (channels != 1 && channels != 2) {
            throw py::value_error("Only 1 or 2 channels supported for streaming.");
        }
        
        if (std::find(OPUS_VALID_RATES.begin(), OPUS_VALID_RATES.end(), sample_rate) == OPUS_VALID_RATES.end()) {
            throw py::value_error("Invalid sample rate for Opus.");
        }
        
        int error;
        decoder = opus_decoder_create(sample_rate, channels, &error);
        if (error != OPUS_OK) {
            throw py::value_error("Failed to create decoder.");
        }
        
        pcm_buffer.resize(5760 * channels); // Max frame size
    }
    
    ~OpusStreamDecoder() {
        if (decoder) {
            opus_decoder_destroy(decoder);
        }
    }
    
    py::array_t<opus_int16> decode(const py::bytes& data) {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        
        std::string data_str = data;
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data_str.data());
        size_t data_size = data_str.size();
        
        packet_buffer.insert(packet_buffer.end(), data_ptr, data_ptr + data_size);
        
        std::vector<opus_int16> output;
        size_t pos = 0;
        
        while (pos + 2 <= packet_buffer.size()) {
            uint16_t be_len = (packet_buffer[pos] << 8) | packet_buffer[pos + 1];
            uint16_t packet_len = ntohs(be_len);
            // wait for more data
            if (pos + 2 + packet_len > packet_buffer.size()) {
                break;
            }
            
            pos += 2;
            
            int samples = opus_decode(decoder, packet_buffer.data() + pos, packet_len,
                                    pcm_buffer.data(), pcm_buffer.size() / channels, 0);
            
            if (samples < 0) {
                throw py::value_error("Decoding failed");
            }
            
            output.insert(output.end(), pcm_buffer.begin(),
                         pcm_buffer.begin() + samples * channels);
            
            pos += packet_len;
        }
        
        // clean buffer
        if (pos > 0) {
            packet_buffer.erase(packet_buffer.begin(), packet_buffer.begin() + pos);
        }
        
        if (output.empty()) {
            // Fix: Create empty array properly
            std::vector<ssize_t> empty_shape = {0, static_cast<ssize_t>(channels)};
            auto* empty_data = new opus_int16[0];
            return MakeNpArray<opus_int16>(empty_shape, empty_data);
        }
        
        size_t num_samples = output.size() / channels;
        auto* data_out = new opus_int16[output.size()];
        std::copy(output.begin(), output.end(), data_out);
        
        return MakeNpArray<opus_int16>({static_cast<ssize_t>(num_samples),
                                       static_cast<ssize_t>(channels)}, data_out);
    }
    
    py::array_t<opus_int16> decode_with_fec() {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        
        int samples = opus_decode(decoder, nullptr, 0,
                                pcm_buffer.data(), pcm_buffer.size() / channels, 1);
        
        if (samples <= 0) {
            std::vector<ssize_t> empty_shape = {0, static_cast<ssize_t>(channels)};
            auto* empty_data = new opus_int16[0];
            return MakeNpArray<opus_int16>(empty_shape, empty_data);
        }
        
        auto* data_out = new opus_int16[samples * channels];
        std::copy(pcm_buffer.begin(), pcm_buffer.begin() + samples * channels, data_out);
        
        return MakeNpArray<opus_int16>({static_cast<ssize_t>(samples),
                                       static_cast<ssize_t>(channels)}, data_out);
    }
    
    // Reset decoder state
    void reset() {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        opus_decoder_ctl(decoder, OPUS_RESET_STATE);
        packet_buffer.clear();
    }
};

PYBIND11_MODULE(opusstream, m) {
    m.doc() = "Opus audio codec bindings with streaming support.\n\n"
              "THREAD SAFETY WARNING: Encoder/decoder instances are NOT thread-safe.\n"
              "Each thread should use its own encoder/decoder instance, or use external\n"
              "synchronization when sharing instances between threads.";
    
    m.def("encode_memory", &OpusEncodeMemory, py::arg("waveform_tc"), 
          py::arg("sample_rate"), py::arg("bitrate")=OPUS_AUTO, 
          py::arg("signal_type")=0, py::arg("encoder_complexity")=10,
          "Encodes PCM waveform to Opus format in memory (with Ogg container). "
          "Returns bytes object containing the encoded data.");
    
    m.def("decode_memory", &OpusDecodeMemory, py::arg("opus_data"),
          "Decodes Opus data from memory (with Ogg container). "
          "Returns tuple of (waveform_tc, sample_rate).");
    
    // Raw Opus API (without Ogg container)
    m.def("encode_raw", &OpusEncodeRaw, py::arg("waveform_tc"), 
          py::arg("sample_rate"), py::arg("bitrate")=OPUS_AUTO, 
          py::arg("signal_type")=0, py::arg("encoder_complexity")=10,
          py::arg("frame_duration_ms")=20,
          "Encodes PCM to raw Opus packets (no Ogg container) for maximum efficiency. "
          "Limited to 1 or 2 channels. Returns bytes with custom packet format "
          "(2-byte big-endian length prefix per packet).");
    
    m.def("decode_raw", &OpusDecodeRaw, py::arg("opus_data"), 
          py::arg("sample_rate"), py::arg("channels"),
          "Decodes raw Opus packets (no Ogg container). "
          "Requires sample_rate and channels to be specified. "
          "Returns tuple of (waveform_tc, sample_rate).");
    
    m.def("float32_to_int16", &float32_to_int16, py::arg("input"),
          "Convert float32 audio (-1.0 to 1.0) to int16 format.");
    
    m.def("int16_to_float32", &int16_to_float32, py::arg("input"),
          "Convert int16 audio to float32 format (-1.0 to 1.0).");
    
    py::class_<OpusStreamEncoder>(m, "StreamEncoder")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("sample_rate"),
             py::arg("channels"),
             py::arg("bitrate") = OPUS_AUTO,
             py::arg("signal_type") = 0,
             py::arg("complexity") = 10,
             py::arg("frame_duration_ms") = 20,
             "Create a streaming Opus encoder for real-time encoding.\n"
             "WARNING: Not thread-safe. Use one instance per thread.\n"
             "Args:\n"
             "    sample_rate: Sample rate (8000, 12000, 16000, 24000, or 48000)\n"
             "    channels: Number of channels (1 or 2)\n"
             "    bitrate: Target bitrate (default: OPUS_AUTO)\n"
             "    signal_type: 0=auto, 1=music, 2=voice\n"
             "    complexity: Encoder complexity (0-10)\n"
             "    frame_duration_ms: Frame duration (5, 10, 20, 40, or 60)")
        .def("encode", &OpusStreamEncoder::encode,
             py::arg("samples"),
             "Encode audio samples. Returns encoded packets as bytes.\n"
             "Input shape must be [samples, channels].")
        .def("flush", &OpusStreamEncoder::flush,
             py::arg("encode_silence") = true,
             "Flush remaining samples.\n"
             "Args:\n"
             "    encode_silence: If set to True, pad with zeros and encode, else discard buffered samples.")
        .def("reset", &OpusStreamEncoder::reset,
             "Reset encoder state.")
        .def("get_frame_size", &OpusStreamEncoder::get_frame_size,
             "Get the frame size in samples.");
    
    py::class_<OpusStreamDecoder>(m, "StreamDecoder")
        .def(py::init<int, int>(),
             py::arg("sample_rate"),
             py::arg("channels"),
             "Create a streaming Opus decoder for real-time decoding.\n"
             "WARNING: Not thread-safe. Use one instance per thread.\n"
             "Args:\n"
             "    sample_rate: Sample rate -must match encoder-\n"
             "    channels: Number of channels -must match encoder-")
        .def("decode", &OpusStreamDecoder::decode,
             py::arg("data"),
             "Decodes Opus packets. Returns PCM samples as numpy array.\n"
             "Handles partial packets by buffering.")
        .def("decode_with_fec", &OpusStreamDecoder::decode_with_fec,
             "Decodes a lost packet using Forward Error Correction.\n"
             "Call this when a packet is lost to generate substitute audio.")
        .def("reset", &OpusStreamDecoder::reset,
             "Resets decoder state and clears buffers.");

    m.attr("AUTO") = py::int_(OPUS_AUTO);
    m.attr("BITRATE_MAX") = py::int_(OPUS_BITRATE_MAX);
    
    m.attr("SIGNAL_AUTO") = py::int_(0);
    m.attr("SIGNAL_MUSIC") = py::int_(1);
    m.attr("SIGNAL_VOICE") = py::int_(2);
    
    // Valid Opus sample rates
    m.attr("VALID_RATES") = py::make_tuple(8000, 12000, 16000, 24000, 48000);
}
