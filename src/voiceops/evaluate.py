"""Evaluate the agent on the practice set.

    python -m voiceops.evaluate questions      # Step 1 vs Step 2 (vs full agent)
    python -m voiceops.evaluate conversations  # Step 3 multi-turn behaviour

Scoring follows DATA_DICTIONARY.md: every ` ; `-separated must_include fact is required,
` / ` separates acceptable alternatives, case is ignored and equivalent dates / native
digits are accepted. An LLM judge also compares each reply with the reference answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import config, llm
from .agent import SupportAgent
from .kb import always_on_rules
from .retriever import get_retriever

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ISO_RE = re.compile(r"^(20\d\d)-(\d\d)-(\d\d)$")
NUM_WORDS = {str(i): w for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"30": "thirty", "48": "forty-eight", "49": "forty-nine", "79": "seventy-nine"})
EVAL_DIR = config.ROOT / "eval_results"


# ---------------------------------------------------------------------------------
# Fact matching
# ---------------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(DEV_DIGITS).lower().replace("’", "'"))


def _variants(alt: str) -> list[str]:
    alt = alt.strip()
    out = [alt]
    m = ISO_RE.match(alt)
    if m:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        mon, abbr = d.strftime("%B"), d.strftime("%b")
        out += [f"{d.day} {mon}", f"{mon} {d.day}", f"{d.day} {abbr}", f"{d.day}th {mon}",
                f"{d.day}st {mon}", f"{d.day}nd {mon}", f"{d.day}rd {mon}", f"{d.day}th of {mon}"]
    if alt in NUM_WORDS:
        out.append(NUM_WORDS[alt])
    return out


def _contains(reply: str, phrase: str) -> bool:
    p = _norm(phrase)
    if not p:
        return True
    if re.fullmatch(r"[\d.,]+", p):  # bare numbers must not match inside other numbers
        return re.search(rf"(?<![\d.]){re.escape(p)}(?![\d])", reply) is not None
    return p in reply


def fact_check(reply: str, must_include: str, must_not_include: str) -> dict:
    r = _norm(reply)
    facts = [f for f in must_include.split(" ; ") if f.strip()]
    missing = [f for f in facts if not any(_contains(r, v) for a in f.split(" / ") for v in _variants(a))]
    banned = [b for b in must_not_include.split(" ; ") if b.strip() and _norm(b.strip()) in r]
    return {"facts_ok": not missing and not banned, "missing": missing, "forbidden_found": banned}


def split_ids(cell: str) -> list[str]:
    return [x.strip() for x in str(cell).split(";") if x.strip()]


def context_ids(agent: SupportAgent, res) -> list[str]:
    """What the model could see: retrieved sections plus the always-on agent-guide rules."""
    if agent.mode == "baseline":
        return []
    return [r["id"] for r in res.retrieved] + [c.id for c in always_on_rules(agent.retriever.chunks)]


def retrieval_scores(gold: list[str], retrieved: list[str], cited: list[str]) -> dict:
    if not gold:
        return {"recall": None, "cited_recall": None, "cited_precision": None}
    g = set(gold)
    return {"recall": len(g & set(retrieved)) / len(g),
            "cited_recall": len(g & set(cited)) / len(g),
            "cited_precision": (len(g & set(cited)) / len(set(cited))) if cited else 0.0}


# ---------------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------------
JUDGE_SYSTEM = """You grade replies from ZentroMart's customer support assistant against a reference.
A reply is correct when it gives the same decision and the same key facts (numbers, dates, yes/no) as the
reference answer and the required facts, and claims nothing the notes or forbidden phrases rule out.
Wording, extra politeness, an AI-assistant introduction and language (English / Hindi / Hinglish) do not
matter. For unanswerable questions, a reply that says it doesn't have the information (optionally offering a
human) is correct; an invented policy is wrong."""

JUDGE_FORMAT = {"type": "json_schema", "schema": {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["correct", "reason"], "additionalProperties": False}}


def judge(question: str, reply: str, row: dict) -> dict:
    prompt = (f"Customer message: {question}\n\nReference answer: {row.get('reference_answer', '')}\n"
              f"Expected decision: {row.get('expected_decision', '')}\n"
              f"Required facts: {row.get('must_include', '') or '-'}\n"
              f"Forbidden claims: {row.get('must_not_include', '') or '-'}\n"
              f"Notes: {row.get('notes', '') or '-'}\n\nAssistant reply to grade:\n{reply}")
    try:
        resp = llm.create(JUDGE_SYSTEM, [{"role": "user", "content": prompt}], model=config.JUDGE_MODEL,
                          effort="low", max_tokens=4000, output_format=JUDGE_FORMAT)
        return json.loads(llm.text_of(resp))
    except Exception as exc:
        return {"correct": None, "reason": f"judge error: {exc}"}


# ---------------------------------------------------------------------------------
# Questions: Step 1 vs Step 2 (vs Step 3 agent)
# ---------------------------------------------------------------------------------
def run_question(row: dict, mode: str, use_asr: bool, use_judge: bool) -> dict:
    text = row["asr_text"] if use_asr and row.get("asr_text") else row["question"]
    agent = SupportAgent(mode, None if mode == "baseline" else get_retriever())
    try:
        res = agent.respond(text)
    except Exception as exc:
        return {"id": row["id"], "mode": mode, "error": str(exc)}
    acceptable = split_ids(row.get("acceptable_decisions") or row["expected_decision"])
    fc = fact_check(res.reply, row.get("must_include", ""), row.get("must_not_include", ""))
    gold = split_ids(row.get("gold_evidence", ""))
    out = {"id": row["id"], "mode": mode, "language": row["language"], "input": text, "reply": res.reply,
           "decision": res.decision, "expected_decision": row["expected_decision"],
           "decision_ok": res.decision in acceptable, **fc,
           **retrieval_scores(gold, context_ids(agent, res), res.sources),
           "gold_evidence": " ; ".join(gold), "sources": " ; ".join(res.sources),
           "tools": " ; ".join(c["name"] for c in res.tool_calls), "latency_s": res.latency_s}
    if use_judge:
        j = judge(text, res.reply, row)
        out.update(judge_correct=j.get("correct"), judge_reason=j.get("reason"))
    # Headline metric: right decision, required facts present, judge agrees (when used).
    out["correct"] = bool(out["decision_ok"] and fc["facts_ok"] and (out.get("judge_correct") is not False))
    return out


def eval_questions(modes, use_asr, use_judge, workers, limit):
    df = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False)
    rows = df.to_dict("records")[:limit] if limit else df.to_dict("records")
    if any(m != "baseline" for m in modes):
        get_retriever()  # build the index once before threads start
    jobs = [(r, m) for m in modes for r in rows]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: _progress(run_question(j[0], j[1], use_asr, use_judge)), jobs))
    return pd.DataFrame(results)


def _progress(r: dict) -> dict:
    mark = "ERR" if r.get("error") else ("ok " if r.get("correct") else "MISS")
    print(f"  {mark} {r['mode']:<8} {r['id']}", flush=True)
    return r


# ---------------------------------------------------------------------------------
# Conversations: Step 3
# ---------------------------------------------------------------------------------
TOOL_RE = re.compile(r"(\w+)\((.*?)\)")


def parse_expected_tools(cell: str) -> list[tuple[str, dict]]:
    out = []
    for name, args in TOOL_RE.findall(cell or ""):
        kv = dict(a.split("=", 1) for a in [x.strip() for x in args.split(",")] if "=" in a)
        out.append((name, {k.strip(): v.strip() for k, v in kv.items()}))
    return out


def tools_ok(expected: list[tuple[str, dict]], actual: list) -> bool:
    """Each expected call must appear with matching order_id / verification_value."""
    for name, args in expected:
        def match(c):
            if c.name != name:
                return False
            if name == "lookup_order":
                oid = "".join(ch for ch in str(c.input.get("order_id", "")) if ch.isdigit())
                if args.get("order_id") and oid != args["order_id"]:
                    return False
                want_v = args.get("verification_value")
                got_v = "".join(ch for ch in str(c.input.get("verification_value") or "") if ch.isdigit())
                return got_v == want_v if want_v else True
            return name != "create_ticket" or bool(c.output.get("created"))
        if not any(match(c) for c in actual):
            return False
    return True


def run_conversation(cid: str, turns: pd.DataFrame, use_judge: bool) -> list[dict]:
    agent = SupportAgent("full", get_retriever())
    out, history = [], []
    for _, row in turns.iterrows():
        row = row.to_dict()
        try:
            res = agent.respond(row["customer_message"])
        except Exception as exc:
            out.append({"conversation_id": cid, "turn": row["turn"], "error": str(exc)})
            break
        actual = [c for c in agent.runner.calls if c.turn == agent.turn]
        expected = parse_expected_tools(row.get("expected_tool_calls", ""))
        fc = fact_check(res.reply, row.get("must_include", ""), row.get("must_not_include", ""))
        gold = split_ids(row.get("gold_evidence", ""))
        rec = {"conversation_id": cid, "turn": int(row["turn"]), "scenario": row["scenario"],
               "language": row["language"], "customer_message": row["customer_message"], "reply": res.reply,
               "decision": res.decision, "expected_decision": row["expected_decision"],
               "decision_ok": res.decision == row["expected_decision"],
               "expected_tools": row.get("expected_tool_calls", ""),
               "actual_tools": " ; ".join(f"{c.name}({', '.join(k for k in c.input)})" for c in actual),
               "tools_ok": tools_ok(expected, actual), **fc,
               **retrieval_scores(gold, context_ids(agent, res), res.sources),
               "gold_evidence": " ; ".join(gold), "sources": " ; ".join(res.sources),
               "latency_s": res.latency_s}
        if use_judge:
            convo = "\n".join(history + [f"customer: {row['customer_message']}"])
            j = judge(convo, res.reply, {**row, "reference_answer": row.get("notes", "")})
            rec.update(judge_correct=j.get("correct"), judge_reason=j.get("reason"))
        rec["correct"] = bool(rec["decision_ok"] and rec["tools_ok"] and fc["facts_ok"]
                              and rec.get("judge_correct") is not False)
        history += [f"customer: {row['customer_message']}", f"assistant: {res.reply}"]
        print(f"  {'ok ' if rec['correct'] else 'MISS'} {cid} t{rec['turn']}", flush=True)
        out.append(rec)
    return out


def eval_conversations(use_judge, workers, limit):
    df = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)
    df["_t"] = df["turn"].astype(int)
    groups = [(cid, g.sort_values("_t")) for cid, g in df.groupby("conversation_id", sort=True)]
    groups = groups[:limit] if limit else groups
    get_retriever()
    with ThreadPoolExecutor(workers) as pool:
        nested = list(pool.map(lambda g: run_conversation(g[0], g[1], use_judge), groups))
    return pd.DataFrame([r for conv in nested for r in conv])


# ---------------------------------------------------------------------------------
def _rate(series) -> str:
    s = series.dropna()
    return f"{s.mean() * 100:.1f}%" if len(s) else "-"


def summarise_questions(df: pd.DataFrame) -> str:
    lines = ["| Mode | Correct | Decision OK | Facts OK | Judge OK | Retrieval recall | Cited recall | n |",
             "|---|---|---|---|---|---|---|---|"]
    names = {"baseline": "Step 1 (LLM only)", "rag": "Step 2 (RAG)", "full": "Step 3 agent"}
    for mode, g in df.groupby("mode", sort=False):
        ok = g["correct"].sum()
        lines.append(f"| {names.get(mode, mode)} | {ok}/{len(g)} ({ok / len(g) * 100:.1f}%) | "
                     f"{_rate(g['decision_ok'])} | {_rate(g['facts_ok'])} | "
                     f"{_rate(g.get('judge_correct', pd.Series(dtype=float)))} | "
                     f"{_rate(g['recall'])} | {_rate(g['cited_recall'])} | {len(g)} |")
    return "\n".join(lines)


def summarise_conversations(df: pd.DataFrame) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| Turns correct | {int(df['correct'].sum())}/{len(df)} ({df['correct'].mean() * 100:.1f}%) |",
              f"| Decision match | {_rate(df['decision_ok'])} |",
              f"| Expected tool calls made | {_rate(df['tools_ok'])} |",
              f"| Required facts present | {_rate(df['facts_ok'])} |",
              f"| Judge agrees | {_rate(df.get('judge_correct', pd.Series(dtype=float)))} |",
              f"| Retrieval recall (gold sections) | {_rate(df['recall'])} |",
              f"| Conversations fully correct | "
              f"{int(df.groupby('conversation_id')['correct'].all().sum())}/{df['conversation_id'].nunique()} |"]
    by = df.groupby("scenario")["correct"].mean().sort_values()
    lines += ["", "| Scenario | Turns correct |", "|---|---|"]
    lines += [f"| {s} | {v * 100:.0f}% |" for s, v in by.items()]
    return "\n".join(lines)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["questions", "conversations", "all"])
    ap.add_argument("--modes", default="baseline,rag,full", help="for questions")
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of the typed question")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    out = EVAL_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report = [f"# Evaluation {out.name}", f"model: {config.LLM_MODEL} (effort {config.LLM_EFFORT}), "
              f"embeddings: {config.EMBED_MODEL}, top_k: {config.TOP_K}"]
    t0 = time.time()
    if args.what in ("questions", "all"):
        df = eval_questions(args.modes.split(","), args.asr, not args.no_judge, args.workers, args.limit)
        df.to_csv(out / "questions.csv", index=False, encoding="utf-8-sig")
        report += ["", f"## Practice questions ({'asr_text' if args.asr else 'typed text'})", "",
                   summarise_questions(df)]
    if args.what in ("conversations", "all"):
        df = eval_conversations(not args.no_judge, args.workers, args.limit)
        df.to_csv(out / "conversations.csv", index=False, encoding="utf-8-sig")
        report += ["", "## Practice conversations", "", summarise_conversations(df)]
    report.append(f"\n_took {time.time() - t0:.0f}s_")
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report) + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
