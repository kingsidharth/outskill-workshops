"""Gemini two-speaker podcast speech, returned as a browser-playable WAV."""
import io
import os
import re
import time
import wave

from google import genai
from google.genai import errors, types

TTS_MODEL = "gemini-3.1-flash-tts-preview"
HOST_VOICES = {"Joe": "Puck", "Jane": "Kore"}
MAX_BATCH_BYTES = 4_000


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


def _transcript_batches(segments: list[dict]) -> list[str]:
    batches = []
    current = ""
    for segment in segments:
        prefix = f'{segment["speaker"]}: '
        remaining = segment["text"]
        turn_bytes = len((prefix + remaining + "\n").encode())
        if current and turn_bytes <= MAX_BATCH_BYTES and len(current.encode()) + turn_bytes > MAX_BATCH_BYTES:
            batches.append(current)
            current = ""
        while remaining:
            budget = MAX_BATCH_BYTES - len((current + prefix + "\n").encode())
            if budget <= 0:
                batches.append(current)
                current = ""
                continue
            piece = remaining.encode()[:budget].decode("utf-8", errors="ignore")
            if not piece:
                batches.append(current)
                current = ""
                continue
            if len(piece) < len(remaining):
                boundary = piece.rfind(" ")
                if boundary > len(piece) // 2:
                    piece = piece[:boundary + 1]
            current += prefix + piece + "\n"
            remaining = remaining[len(piece):]
            if remaining:
                batches.append(current)
                current = ""
    if current:
        batches.append(current)
    return batches


def _join_wavs(chunks: list[bytes]) -> bytes:
    output = io.BytesIO()
    audio_format = None
    with wave.open(output, "wb") as combined:
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as audio:
                chunk_format = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
                if audio_format is None:
                    audio_format = chunk_format
                    combined.setnchannels(chunk_format[0])
                    combined.setsampwidth(chunk_format[1])
                    combined.setframerate(chunk_format[2])
                elif chunk_format != audio_format:
                    raise RuntimeError("Gemini TTS returned inconsistent audio formats across batches.")
                combined.writeframes(audio.readframes(audio.getnframes()))
    return output.getvalue()


class _MissingAudioError(RuntimeError):
    pass


def generate_podcast_audio(segments: list[dict]) -> bytes:
    if not segments or {seg["speaker"] for seg in segments} != set(HOST_VOICES):
        raise ValueError("Podcast script must contain dialogue for both Joe and Jane.")
    config = types.GenerateContentConfig(
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
    )
    chunks = []
    with genai.Client(api_key=require_gemini_key()) as client:
        for transcript in _transcript_batches(segments):
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=os.getenv("GEMINI_TTS_MODEL", TTS_MODEL),
                        contents="Read this podcast dialogue naturally, using the two configured speakers:\n" + transcript,
                        config=config,
                    )
                    candidate = next(iter(response.candidates or []), None)
                    if candidate and "MAX_TOKENS" in str(getattr(candidate, "finish_reason", "")):
                        raise RuntimeError("Gemini TTS reached its audio limit; shorten the podcast dialogue.")
                    parts = (candidate.content.parts or []) if candidate and candidate.content else []
                    audio = next((part.inline_data for part in parts if part.inline_data and part.inline_data.data), None)
                    if audio is None:
                        raise _MissingAudioError("Gemini TTS returned no audio after repeated attempts.")
                    chunks.append(_to_wav(audio.data, audio.mime_type or ""))
                    break
                except (errors.ServerError, _MissingAudioError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
    return _join_wavs(chunks)
