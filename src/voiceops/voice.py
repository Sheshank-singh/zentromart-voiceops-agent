"""Step 4: Gnani speech-to-text in, Gnani text-to-speech out. The agent is unchanged."""
from __future__ import annotations

import io
import logging
import re
import struct
import wave
from datetime import date
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 55  # Gnani REST STT accepts up to 60 s per request
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
ISO_DATE_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
RUPEE_RE = re.compile(r"₹\s?([\d,]+(?:\.\d+)?)")
LONG_NUM_RE = re.compile(r"\b\d{7,}\b")
TICKET_RE = re.compile(r"\bTKT[0-9a-f]{8,}\b", re.I)

UNCLEAR_REPLY = {
    "en": "Sorry, I didn't catch that. Could you please say it again?",
    "hi": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। क्या आप दोबारा बता सकते हैं?",
}

AGENT_ERROR_REPLY = ("Sorry, I'm having trouble looking that up right now. "
                     "Please try again in a moment, or I can connect you with a colleague.")


def speech_text(text: str) -> str:
    """Rewrite a reply for the ear (STY-3, Gnani text-normalization guide)."""
    hindi = bool(DEVANAGARI_RE.search(text))
    t = re.sub(r"[*_`#>|]+", "", text)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = TICKET_RE.sub("your ticket" if not hindi else "आपका टिकट", t)
    t = ISO_DATE_RE.sub(lambda m: _spoken_date(m), t)
    t = RUPEE_RE.sub(lambda m: f"{m.group(1).replace(',', '')} {'रुपये' if hindi else 'rupees'}", t)
    # Order IDs and reference numbers: groups of four digits.
    t = LONG_NUM_RE.sub(lambda m: " ".join(m.group(0)[i:i + 4] for i in range(0, len(m.group(0)), 4)), t)
    return re.sub(r"\s+", " ", t).strip()


def _spoken_date(m: re.Match) -> str:
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return m.group(0)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def tts_language(text: str) -> str:
    return "hi-IN" if DEVANAGARI_RE.search(text) else config.TTS_LANGUAGE


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY
        self.tts = GnaniTTSClient()

    # ---- ears ----------------------------------------------------------------------
    def transcribe(self, audio_path: str | Path, language: str = config.STT_LANGUAGE) -> str:
        audio_path = Path(audio_path)
        parts = _split_wav(audio_path) if audio_path.suffix.lower() == ".wav" else None
        if not parts:
            return self.stt.transcribe(str(audio_path), language_code=language).get("transcript", "").strip()
        texts = []
        for i, chunk in enumerate(parts):
            res = self.stt.transcribe_bytes(chunk, filename=f"part{i}.wav", language_code=language)
            texts.append(res.get("transcript", "").strip())
        return " ".join(t for t in texts if t)

    # ---- voice ---------------------------------------------------------------------
    def speak(self, text: str, out_path: str | Path) -> Path:
        spoken = speech_text(text)
        self.tts.synthesize(spoken, voice=config.TTS_VOICE, model=config.TTS_MODEL,
                            language=tts_language(spoken), output_file=str(out_path))
        return Path(out_path)


def _split_wav(path: Path) -> list[bytes] | None:
    """Split a WAV longer than the STT limit into <=55 s pieces; None if it fits as-is."""
    try:
        with wave.open(str(path), "rb") as w:
            # Streamed WAVs carry a placeholder frame count; trust the file size instead.
            frame_bytes = w.getsampwidth() * w.getnchannels()
            rate = w.getframerate()
            frames = min(w.getnframes(), path.stat().st_size // frame_bytes)
            if frames / rate <= MAX_CLIP_SECONDS:
                return None
            params = w.getparams()._replace(nframes=0)
            step, parts = rate * MAX_CLIP_SECONDS, []
            while True:
                data = w.readframes(step)
                if not data:
                    break
                buf = io.BytesIO()
                with wave.open(buf, "wb") as out:
                    out.setparams(params)
                    out.writeframes(data)
                parts.append(buf.getvalue())
            return parts
    except (wave.Error, EOFError, struct.error):
        return None  # not plain PCM; let the API decide


def is_unclear(transcript: str) -> bool:
    """Empty or near-empty transcripts (noise, silence) get a 'please repeat' reply."""
    words = re.findall(r"\w+", transcript)
    return len(words) == 0 or (len(words) == 1 and len(words[0]) <= 2)


class VoiceSession:
    """One phone call: STT -> agent -> TTS, with recovery from unclear audio."""

    def __init__(self, agent, voice: Voice | None = None, stt_language: str = config.STT_LANGUAGE):
        self.agent, self.voice, self.stt_language = agent, voice or Voice(), stt_language

    def turn(self, audio_path: str | Path, reply_wav: str | Path) -> dict:
        record: dict = {"audio": str(audio_path)}
        try:
            heard = self.voice.transcribe(audio_path, self.stt_language)
        except Exception as exc:
            log.error("STT failed for %s: %s", audio_path, exc)
            heard, record["stt_error"] = "", str(exc)
        record["transcript"] = heard
        if is_unclear(heard):
            reply = UNCLEAR_REPLY["hi" if self.stt_language.startswith("hi") else "en"]
            record.update(reply=reply, decision="clarify", sources=[], chunks_used=[], retrieved=[], tool_calls=[],
                          ticket_created=False, unclear_audio=True)
        else:
            try:
                record.update(self.agent.respond(heard).as_dict())
            except Exception as exc:  # keep the call alive; the customer still hears something
                log.error("Agent failed on %s: %s", audio_path, exc)
                record.update(reply=AGENT_ERROR_REPLY, decision="escalate", sources=[], chunks_used=[],
                              retrieved=[], tool_calls=[], ticket_created=False, agent_error=str(exc))
            reply = record["reply"]
        record["spoken_text"] = speech_text(reply)
        try:
            self.voice.speak(reply, reply_wav)
            record["reply_audio"] = str(reply_wav)
        except Exception as exc:
            log.error("TTS failed: %s", exc)
            record["tts_error"] = str(exc)
        return record
