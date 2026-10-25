"""Web demo: talk to the ZentroMart agent from the browser.

    python app.py            # then open http://127.0.0.1:7860

Record with the mic (or upload a clip, or type). Each turn goes Gnani STT -> agent ->
Gnani TTS, and the panel on the right shows the chunks and tool calls behind the reply.
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from voiceops import config  # noqa: E402
from voiceops.agent import SupportAgent  # noqa: E402
from voiceops.retriever import get_retriever  # noqa: E402
from voiceops.voice import Voice, VoiceSession  # noqa: E402

RETRIEVER = get_retriever()
VOICE = Voice()
SESSIONS: dict[str, VoiceSession] = {}
AUDIO_DIR = Path(tempfile.gettempdir()) / "voiceops_replies"
AUDIO_DIR.mkdir(exist_ok=True)


def _session(sid: str, lang: str) -> VoiceSession:
    if sid not in SESSIONS:
        SESSIONS[sid] = VoiceSession(SupportAgent("full", RETRIEVER), VOICE, lang)
    SESSIONS[sid].stt_language = lang
    return SESSIONS[sid]


def _details(rec: dict) -> dict:
    return {"heard": rec.get("transcript", rec.get("user_text")), "decision": rec.get("decision"),
            "sources": rec.get("sources"), "tool_calls": rec.get("tool_calls"),
            "ticket_created": rec.get("ticket_created"), "latency_s": rec.get("latency_s"),
            "chunks_used": rec.get("chunks_used")}


def on_audio(audio_path, history, sid, lang):
    if not audio_path:
        return history, None, gr.update(), None
    sess = _session(sid, lang)
    rec = sess.turn(audio_path, AUDIO_DIR / f"{sid}_{sess.agent.turn + 1}.wav")
    history = history + [{"role": "user", "content": f"🎙️ {rec['transcript'] or '(unclear audio)'}"},
                         {"role": "assistant", "content": rec["reply"]}]
    return history, rec.get("reply_audio"), _details(rec), None


def on_text(text, history, sid, lang, speak):
    if not text.strip():
        return history, None, gr.update(), ""
    sess = _session(sid, lang)
    rec = sess.agent.respond(text).as_dict()
    audio = None
    if speak:
        audio = str(VOICE.speak(rec["reply"], AUDIO_DIR / f"{sid}_{sess.agent.turn}.wav"))
    history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": rec["reply"]}]
    return history, audio, _details(rec), ""


def new_call(sid):
    SESSIONS.pop(sid, None)
    return [], None, {}, str(uuid.uuid4())


with gr.Blocks(title="ZentroMart Voice Support") as demo:
    sid = gr.State(lambda: str(uuid.uuid4()))
    gr.Markdown(f"## ZentroMart voice support agent\nGnani STT → {config.LLM_MODEL} + RAG + tools → "
                f"Gnani TTS ({config.TTS_VOICE}). Dataset date: {config.TODAY:%d %B %Y}.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=460, label="Conversation")
            reply_audio = gr.Audio(label="Agent reply", autoplay=True, interactive=False)
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak (max 60 s)")
            with gr.Row():
                text = gr.Textbox(placeholder="…or type a message", scale=4, show_label=False)
                speak = gr.Checkbox(value=True, label="Speak reply", scale=1)
            with gr.Row():
                lang = gr.Dropdown(["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN", "bn-IN",
                                    "gu-IN", "pa-IN"], value=config.STT_LANGUAGE, label="STT language")
                reset = gr.Button("New call")
        with gr.Column(scale=2):
            details = gr.JSON(label="Behind this reply: sources, tool calls, decision")

    mic.stop_recording(on_audio, [mic, chat, sid, lang], [chat, reply_audio, details, mic])
    mic.upload(on_audio, [mic, chat, sid, lang], [chat, reply_audio, details, mic])
    text.submit(on_text, [text, chat, sid, lang, speak], [chat, reply_audio, details, text])
    reset.click(new_call, [sid], [chat, reply_audio, details, sid])

if __name__ == "__main__":
    demo.launch()
