import asyncio
import base64
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import main
from app_agents import podcast_voice


class PodcastVoiceTests(unittest.TestCase):
    def test_gemini_sends_both_voices_and_wraps_pcm(self):
        pcm = b"\x00\x00" * 240
        part = SimpleNamespace(inline_data=SimpleNamespace(data=pcm, mime_type="audio/L16;rate=24000"))
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(podcast_voice.genai, "Client") as client_class,
        ):
            client = client_class.return_value.__enter__.return_value
            client.models.generate_content.return_value = response
            audio = podcast_voice.generate_podcast_audio([
                {"speaker": "Joe", "text": "Hello Jane"},
                {"speaker": "Jane", "text": "Hello Joe"},
            ])
            request = client.models.generate_content.call_args.kwargs
            voices = request["config"].speech_config.multi_speaker_voice_config.speaker_voice_configs
            self.assertEqual([v.speaker for v in voices], ["Joe", "Jane"])
            self.assertIn("Joe: Hello Jane", request["contents"])
            self.assertIn("Jane: Hello Joe", request["contents"])
            with wave.open(io.BytesIO(audio)) as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.readframes(240), pcm)

    def test_long_unicode_dialogue_is_bounded_without_text_loss(self):
        segments = [{"speaker": "Joe", "text": "é" * 5000}, {"speaker": "Jane", "text": "sunshine " * 1500}]
        batches = podcast_voice._transcript_batches(segments)
        self.assertGreater(len(batches), 1)
        recovered = {"Joe": "", "Jane": ""}
        for batch in batches:
            self.assertLessEqual(len(batch.encode()), podcast_voice.MAX_BATCH_BYTES)
            for line in batch.splitlines():
                speaker, text = line.split(": ", 1)
                recovered[speaker] += text
        self.assertEqual(recovered, {segment["speaker"]: segment["text"] for segment in segments})

    def test_wav_batches_are_joined_in_order(self):
        chunks = [podcast_voice._to_wav(data, "audio/L16;rate=24000") for data in [b"\x01\x00" * 10, b"\x02\x00" * 20]]
        with wave.open(io.BytesIO(podcast_voice._join_wavs(chunks))) as audio:
            self.assertEqual(audio.getnframes(), 30)
            self.assertEqual(audio.readframes(30), b"\x01\x00" * 10 + b"\x02\x00" * 20)
        with self.assertRaises(RuntimeError):
            podcast_voice._join_wavs([chunks[0], podcast_voice._to_wav(b"\x00\x00", "audio/L16;rate=16000")])

    def test_rejects_empty_or_invalid_audio(self):
        for data, mime in [(b"", "audio/L16"), (b"x", "audio/L16"), (b"text", "text/plain")]:
            with self.subTest(mime=mime), self.assertRaises(RuntimeError):
                podcast_voice._to_wav(data, mime)

    def test_parser_keeps_short_dialogue(self):
        self.assertEqual(main._parse_script_segments("Joe: Yes.\nJane: No."), [
            {"speaker": "Joe", "text": "Yes."}, {"speaker": "Jane", "text": "No."}
        ])

    def test_full_pipeline_keeps_more_than_fifty_turns(self):
        script = "\n".join(f'{"Joe" if i % 2 == 0 else "Jane"}: Workshop dialogue number {i}' for i in range(60))
        job_id = "gemini-test"
        main._podcaster_jobs[job_id] = {"steps": {}}
        try:
            with (
                patch.object(main, "require_gemini_key"),
                patch.object(main, "run_podcast", return_value={"script": script, "topic": "solar"}),
                patch.object(main, "generate_podcast_audio", return_value=b"wav-audio") as generate,
            ):
                asyncio.run(main._run_podcaster_job(job_id, "solar"))
                job = main._podcaster_jobs[job_id]
                self.assertEqual(job["status"], "completed")
                self.assertEqual(len(generate.call_args.args[0]), 60)
                self.assertEqual(job["result"]["audio_mime_type"], "audio/wav")
                self.assertEqual(base64.b64decode(job["result"]["audio"]), b"wav-audio")
        finally:
            main._podcaster_jobs.pop(job_id, None)
