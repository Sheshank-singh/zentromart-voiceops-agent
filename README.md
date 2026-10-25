# ZentroMart VoiceOps agent

A customer support agent for ZentroMart (VoiceOps 2026 capstone). It answers from the company's
documents, holds a multi-turn conversation with caller verification, order lookup and human
handover, and takes voice in and out through Gnani's speech models.

```
customer audio ─► Gnani STT ─► agent (Sarvam LLM + hybrid RAG + tools) ─► Gnani TTS ─► spoken reply
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Fill in `.env`:

| Variable | What |
|---|---|
| `GNANI_API_KEY` | From [app.gnani.ai/voice](https://app.gnani.ai/voice). Used for STT and TTS only. |
| `SARVAM_API_KEY` | From [dashboard.sarvam.ai](https://dashboard.sarvam.ai). The LLM is `sarvam-105b` by default (`LLM_MODEL`). Gemini (`GEMINI_API_KEY`) or Claude (`LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`) also work. Sarvam is used only as the LLM; all speech goes through Gnani. |
| `DATASET_DIR` | Path to the extracted `Capstone_Project_Dataset_FTTT_2026` folder. Optional if it sits in the project root or in `./dataset`. |

The dataset is not included in this repository. Extract `Capstone_Project_Dataset_FTTT_2026.zip`
into the project root (or point `DATASET_DIR` at it).

The first run downloads the multilingual embedding model (about 220 MB) into `.cache/` and builds
the vector index. Later runs load it from the cache.

## Run command (batch, used for testing)

```bash
python run.py --input path/to/audio_folder --output outputs
```

- Accepts `.wav .mp3 .flac .ogg .m4a .aac`.
- Each file is one customer turn. Files in the same sub-folder, or named `<call>_turn<N>.wav`,
  run as one conversation in turn order, so the agent remembers the order ID and context. Every
  other file is a separate call. `--single-session` treats the whole folder as one call.
- `--stt-language hi-IN` (or any Gnani code) changes the STT language. The default `en-IN` also
  handles mixed English and Hindi and returns Latin script.

For every input `<name>` it writes:

| File | Contents |
|---|---|
| `<name>.reply.wav` | The spoken reply (Gnani TTS) |
| `<name>.json` | Transcript, reply text, the text sent to TTS, decision (answer / clarify / escalate / refuse), `sources` (section IDs the reply relies on), `chunks_used` (their text), `retrieved` (everything retrieval returned), tool calls (verification values redacted), latency |
| `results.jsonl`, `results.csv` | All turns in one place |

Unclear or empty audio gets a "sorry, could you say that again?" reply. If STT, the LLM or TTS
fails on one file, the run logs the error in that file's record and carries on.

## Live demo

```bash
python app.py        # http://127.0.0.1:7860
```

Record from the mic, upload a clip or type. The right-hand panel shows the sections, tool calls
and decision behind each reply.

## Evaluation

```bash
python evaluate.py questions          # Step 1 vs Step 2 vs full agent, typed questions
python evaluate.py questions --asr    # same, on the ASR-style transcripts
python evaluate.py conversations      # Step 3: 23 multi-turn conversations
```

Results go to `eval_results/<timestamp>/` as per-item CSVs
plus a `summary.md` table.

Scoring follows the dataset's DATA_DICTIONARY.md:

- **Facts:** every `must_include` fact must appear (` / ` alternatives, equivalent date formats,
  Devanagari digits and spoken numbers accepted), and no `must_not_include` claim may appear.
- **Decision:** must be in `acceptable_decisions` (questions) or equal `expected_decision`
  (conversations).
- **Tool calls (conversations):** every expected `lookup_order` call is made with the same order
  ID and verification value, and every expected `create_ticket` returns `created=True`.
- **LLM judge:** compares the reply with `reference_answer` / `notes` (switch off with
  `--no-judge`).
- **Retrieval:** recall of `gold_evidence` IDs in the retrieved context, and recall and precision
  of the IDs the agent cites.

A reply counts as correct when the decision, facts, tools (if any) and judge all pass.

## How it works

| Step | Where | What |
|---|---|---|
| 1. LLM only | `SupportAgent("baseline")` | System prompt with role and limits; introduces itself as an AI assistant. No documents. |
| 2. RAG | `SupportAgent("rag")` | Section-level chunks of `kb/*.md` plus one chunk per catalog row, hybrid retrieval, answers only from retrieved sections, cites section IDs. |
| 3. Conversation | `SupportAgent("full")` | Adds memory across turns, `search_kb`, `lookup_order` (identity-checked), `create_ticket`, `check_pincode` and `date_calculator`. |
| 4. Voice | `voiceops/voice.py`, `run.py`, `app.py` | Gnani STT in, Gnani TTS out; the agent is unchanged. |

**Chunking.** Each `## [SECTION-ID] Title` section is one chunk, carrying the document's `doc_type`,
`status`, `effective_from` and `superseded_by`. A note above the first section of a document
(for example "this version applies only to electronics delivered before 1 August 2026") is copied
into each of its chunks. Product rows become chunks with IDs like `products.csv:ELE-0012`, which
is how the gold evidence cites them.

**Retrieval.** Multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, so Hindi
and Hinglish questions match English policy text) and BM25, fused with reciprocal rank fusion.
On top of that:

- Superseded and marketing documents are down-weighted (KB-00.1 precedence).
- Spoken numbers in ASR text are normalised ("ele zero zero one to" → `ELE-0012`).
- A product named in the question pulls in the policies for its category, Assured status and
  warranty.
- The model can call `search_kb` itself with an English, policy-worded query when the first
  retrieval misses.

**Always-on rules.** The agent guides (conversation style, identity verification, escalation) and
the "which document wins" rules are in the system prompt every turn, since they govern how the
agent behaves rather than what it looks up.

**Deterministic helpers.** Business-day and calendar-day arithmetic uses `date_calculator`, which
counts Monday to Saturday and skips `holidays.csv`, so timelines are exact rather than estimated
by the LLM. Today is fixed at 20 October 2026.

**Safety rails in code, not just the prompt:**

- `lookup_order` comes from the dataset's own `tools/` package and withholds details until
  verified.
- The wrapper allows at most two verification attempts per order (IDV-3).
- Verification values are redacted from every log and output file.
- A ticket counts as created only when the tool returns `created=True`.
- The agent has no tool that cancels, refunds or credits anything.

**Voice.** Replies are written for the ear (two or three sentences, no markdown). Before TTS:

- ISO dates are spoken as "22 October 2026".
- Rupee amounts are read as rupees.
- Order IDs and RRNs are read in groups of four digits.
- Ticket hashes are never read out.

Devanagari replies are synthesised with `hi-IN`. WAV clips over the 55-second STT limit are split
and transcribed in parts.

## Layout

```
run.py                 batch command (audio folder -> replies)
app.py                 Gradio voice demo
evaluate.py            practice-set evaluation
src/voiceops/
  config.py            settings (.env)
  kb.py                KB + catalog chunking
  retriever.py         hybrid dense + BM25 retrieval
  toolkit.py           tool schemas, dataset tools bridge, date calculator
  agent.py             Steps 1-3 (prompting, RAG, conversation + tools)
  llm.py               LLM wrapper (Sarvam, Gemini or Claude)
  voice.py             Gnani STT/TTS, speech normalisation
  evaluate.py          scoring (fact checks, judge, retrieval metrics)
report/REPORT.md       report draft
sample_outputs/        example run output
```

## Rules compliance

- Gnani is the only speech-to-text and text-to-speech service, with no other STT or TTS even as
  a fallback.
- Only the synthetic dataset is used.
- Answers come from retrieved documents at run time; nothing is hard-coded per question.
- No keys are in the code or the repository.
