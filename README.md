# opusstream
encodes pcm to opus format and decodes opus to pcm live 

# install opus and required dependencies :
```bash
git clone https://github.com/Saga9103/opusstream.git
cd opusstream
sudo ./install_opus.sh
```

# install the package : <br>
```bash
cd opusstream
# either
pip install -e .
# or
python3 setup.py build_ext --inplace
```

# In your python script :
```bash
import opusstream
import numpy as np

# for realtime encoding
encoder = opusstream.StreamEncoder(16000, 1, bitrate=24000, signal_type=opuspy.SIGNAL_VOICE)
decoder = opusstream.StreamDecoder(16000, 1)

# realtime audio chunks simulation
chunk_size = 320  # 20ms at 16kHz
audio_chunk = np.random.randint(-32768, 32767, (chunk_size, 1), dtype=np.int16)

# Encoding
encoded = encoder.encode(audio_chunk)
# or with parameters :
encoder = opusstream.StreamEncoder(
    sample_rate=16000,
    channels=1,               # Mono
    bitrate=24000,           # 24 kbps for voice
    signal_type=2,           # VOICE mode (optimized algorithm), 0 is AUTO, 1 is MUSIC
    complexity=3,            # important for low power device : Lowest complexity, from 0 to 10, 5-6 is a good balance, 3-4 for embedded but even 0 is good
    frame_duration_ms=20     # Standard frame size. possible frame sizes are : 2.5, 5, 10, 20 (good lattency but more calls, more CPU encoding), 40, 60
)

# Continuous streaming - no flush needed
while streaming:
    audio_chunk = get_audio_chunk()  # e.g., 320 samples (20ms at 16kHz)
    encoded_audio = encoder.encode(audio_chunk)
    send_encoded_data(encoded_audio)

# Only at the very end, needed if it receives incomplete frames
final_packet = encoder.flush()  # Gets any remaining samples
# or use : encoder.flush(encode_silence=False) to discard any remaining samples

# Decoding
decoded = decoder.decode(encoded)

# Handling packet loss
recovered = decoder.decode_with_fec()
