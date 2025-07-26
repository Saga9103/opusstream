# for now it is tested on jabra but it can be adapted in the script
import time
import pyaudio
import numpy as np
import opusstream

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = 320  # 20ms 16kHz mono
DURATION_SECONDS = 5 # recorded duration
QUALITY = 1 # from 0 to 10, higher is better but more cpu intensive (the difference is really minimal)

pa = pyaudio.PyAudio()
jabra_index = None

print("Available input devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"{i}: {info['name']}")
    if "jabra" in info["name"].lower():
        jabra_index = i

if jabra_index is None:
    raise RuntimeError("recording device wa not found")

print(f"Using Jabra input index: {jabra_index}")

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=jabra_index,
    frames_per_buffer=FRAME_SIZE,
)

out = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    output=True,
)

encoder = opusstream.StreamEncoder(
    sample_rate=SAMPLE_RATE,
    channels=CHANNELS,
    bitrate=24000,
    signal_type=2,
    complexity=QUALITY,
    frame_duration_ms=20,
)

decoder = opusstream.StreamDecoder(SAMPLE_RATE, CHANNELS)

decoded_chunks = []

print("Recording and encoding 20ms chunks")

start_time = time.time()
while time.time() - start_time < DURATION_SECONDS:
    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
    pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)

    encoded = encoder.encode(pcm)
    decoded = decoder.decode(encoded)
    decoded_chunks.append(decoded)

print("encoding/decoding done")

stream.stop_stream()
stream.close()
out.stop_stream()
out.close()
pa.terminate()

final_audio = np.concatenate(decoded_chunks, axis=0).astype(np.int16)

print("Playing back decoded audio :")
pa = pyaudio.PyAudio()
play = pa.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)
play.write(final_audio.tobytes())
play.stop_stream()
play.close()
pa.terminate()
