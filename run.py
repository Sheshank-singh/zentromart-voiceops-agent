"""Batch runner: a folder of customer audio in, replies (text + audio + sources) out.

    python run.py --input path/to/audio --output outputs

Each audio file is one customer turn. Files that belong to the same call are run in one
conversation, so the agent remembers order IDs and context across them:
  * files inside a sub-folder form one conversation (folder = call), or
  * files named <call>_turn<N>.<ext> / <call>_t<N>.<ext> are grouped by <call>.
Every other file is treated as a standalone call. --single-session forces one call.

Outputs, per audio file <name>:
  <output>/<name>.reply.wav   spoken reply (Gnani TTS)
  <output>/<name>.json        transcript, reply text, decision, chunks used, tool calls
and <output>/results.jsonl + results.csv summarising every turn.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voiceops import config  # noqa: E402
from voiceops.agent import SupportAgent  # noqa: E402
from voiceops.retriever import get_retriever  # noqa: E402
from voiceops.voice import Voice, VoiceSession  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
TURN_RE = re.compile(r"^(?P<call>.+?)[_\-. ]+(?:turn|t)[_\-]?(?P<n>\d+)$", re.I)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def group_calls(folder: Path, single: bool) -> "OrderedDict[str, list[Path]]":
    files = sorted((p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT), key=natural_key)
    calls: OrderedDict[str, list[Path]] = OrderedDict()
    for f in files:
        if single:
            key = "session"
        elif f.parent != folder:
            key = str(f.parent.relative_to(folder))
        else:
            m = TURN_RE.match(f.stem)
            key = m.group("call") if m else f.stem
        calls.setdefault(key, []).append(f)
    return calls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path, help="folder of customer audio files")
    ap.add_argument("--output", "-o", default=Path("outputs"), type=Path)
    ap.add_argument("--single-session", action="store_true", help="treat all files as one call")
    ap.add_argument("--stt-language", default=config.STT_LANGUAGE, help="Gnani STT language code")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    calls = group_calls(args.input, args.single_session)
    if not calls:
        sys.exit(f"No audio files ({', '.join(sorted(AUDIO_EXT))}) found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    retriever, voice = get_retriever(), Voice()
    rows = []
    with open(args.output / "results.jsonl", "w", encoding="utf-8") as jl:
        for call, files in calls.items():
            session = VoiceSession(SupportAgent("full", retriever), voice, args.stt_language)
            for turn, f in enumerate(files, 1):
                rel = f.relative_to(args.input).with_suffix("")
                stem = "__".join(rel.parts)
                rec = session.turn(f, args.output / f"{stem}.reply.wav")
                rec.update(call=call, turn=turn, file=str(f.relative_to(args.input)))
                (args.output / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jl.flush()
                rows.append(rec)
                print(f"[{call} #{turn}] heard: {rec['transcript']!r}\n    reply: {rec['reply']}\n"
                      f"    sources: {', '.join(rec.get('sources', []))}", flush=True)

    with open(args.output / "results.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "call", "turn", "transcript", "reply", "decision", "sources", "tools",
                    "ticket_created", "reply_audio", "latency_s", "error"])
        for r in rows:
            w.writerow([r["file"], r["call"], r["turn"], r["transcript"], r["reply"], r.get("decision"),
                        " ; ".join(r.get("sources", [])),
                        " ; ".join(c["name"] for c in r.get("tool_calls", [])), r.get("ticket_created"),
                        r.get("reply_audio", ""), r.get("latency_s", ""),
                        r.get("stt_error") or r.get("agent_error") or r.get("tts_error") or ""])
    print(f"\nDone: {len(rows)} turns in {len(calls)} calls -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
