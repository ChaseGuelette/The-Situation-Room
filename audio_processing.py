# audio_processing.py

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import io

def reduce_noise(audio_data, sample_rate, channels):
    # Convert raw audio data to AudioSegment
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=channels
    )

    # Apply high-pass filter to remove low frequency noise
    filtered_audio = audio_segment.high_pass_filter(300)

    # Normalize audio volume
    normalized_audio = normalize(filtered_audio)

    # Convert back to raw audio data
    buffer = io.BytesIO()
    normalized_audio.export(buffer, format="raw")
    return buffer.getvalue()

def process_audio_chunk(chunk, sample_rate, channels):
    # Convert chunk to numpy array
    audio_data = np.frombuffer(chunk, dtype=np.int16)

    # Apply noise reduction
    processed_data = reduce_noise(audio_data.tobytes(), sample_rate, channels)

    return processed_data