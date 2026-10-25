"""Gemini two-speaker podcast speech, returned as a browser-playable WAV."""
import io
import os
import re
import wave

from google import genai
from google.genai import types

TTS_MODEL = "gemini-3.1-flash-tts-preview"
HOST_VOICES = {"Joe": "Puck", "Jane": "Kore"}


def require_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before generating a Gemini podcast.")
    return key


def _to_wav(data: bytes, mime_type: str) -> bytes:
    if not data:
        raise RuntimeError("Gemini TTS returned empty audio.")
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return data
    if not mime_type.lower().startswith(("audio/l16", "audio/pcm")):
        raise RuntimeError(f"Gemini TTS returned unsupported audio format: {mime_type}")
    if len(data) % 2:
        raise RuntimeError("Gemini TTS returned incomplete 16-bit PCM audio.")
    match = re.search(r"rate=(\d+)", mime_type, re.IGNORECASE)
    sample_rate = int(match.group(1)) if match else 24_000
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)
    return output.getvalue()


def generate_podcast_audio(segments: list[dict]) -> bytes:
    if not segments or {seg["speaker"] for seg in segments} != set(HOST_VOICES):
        raise ValueError("Podcast script must contain dialogue for both Joe and Jane.")
    transcript = "\n".join(f'{seg["speaker"]}: {seg["text"]}' for seg in segments)
    with genai.Client(api_key=require_gemini_key()) as client:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_TTS_MODEL", TTS_MODEL),
            contents="Read this podcast dialogue naturally, using the two configured speakers:\n" + transcript,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker=name,
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                                ),
                            )
                            for name, voice in HOST_VOICES.items()
                        ]
                    )
                ),
            ),
        )
    for candidate in response.candidates or []:
        for part in (candidate.content.parts or []) if candidate.content else []:
            if part.inline_data and part.inline_data.data:
                return _to_wav(part.inline_data.data, part.inline_data.mime_type or "")
    raise RuntimeError("Gemini TTS returned no audio. Try a shorter podcast script.")
