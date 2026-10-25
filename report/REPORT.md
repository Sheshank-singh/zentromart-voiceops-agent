# ZentroMart VoiceOps Agent: Capstone Report

VoiceOps 2026 · LLM & Generative AI Course · Consulting & Analytics Club, IIT Guwahati × Gnani.ai

**Repository:** `<your public GitHub URL>` · **Run command:** `python run.py --input <audio_folder> --output outputs`

---

## 1. What we built

A customer support agent for ZentroMart that answers from the company's documents, holds a
multi-turn conversation with caller verification, order lookup and human handover, and takes
phone-style voice input and output through Gnani.

```
customer audio → Gnani STT → agent (Sarvam LLM + hybrid RAG + tools) → Gnani TTS → spoken reply
```

| Component | Choice |
|---|---|
| Speech to text | Gnani STT (`en-IN`, accepts Hinglish; `hi-IN` and other languages selectable) |
| Text to speech | Gnani TTS (`timbre-v2.5`, voice Kaveri; `hi-IN` for Devanagari replies) |
| LLM | Sarvam `sarvam-105b`, low reasoning effort |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local, via fastembed) |
| Keyword search | BM25 |
| Framework | Plain Python, Gradio for the demo UI |

### Step 1: Answer a question with an LLM
A system prompt makes the model ZentroMart's support assistant. It sets the role, the topics it
may help with, and what it cannot do (cancel, refund, compensate). Its first reply says it is an AI
assistant. It has no documents and no tools, so policy answers are guesses.

### Step 2: Ground it in the documents
- **Chunking.** Each `## [SECTION-ID] Title` section of the 21 KB files is one chunk (84 sections).
  Each chunk keeps its document's `doc_type`, `status`, `effective_from` and `superseded_by`. A note
  above a document's first section (e.g. "returns v3 applies only to electronics delivered before
  1 August 2026") is copied into each of its chunks. Each of the 70 product rows is also a chunk
  (`products.csv:ELE-0012`), because gold evidence cites catalog rows.
- **Hybrid retrieval.** Dense multilingual embeddings and BM25, fused with reciprocal rank fusion
  (top 8). Hindi and Hinglish questions match English policy text through the multilingual model.
- **Precedence (KB-00.1).** Superseded documents and the marketing page are down-weighted, and each
  chunk shows its status to the LLM.
- **ASR robustness.** Spoken numbers are normalised before search ("ele zero zero one to" → `ELE-0012`).
- **Product awareness.** When a product is named, its catalog row is pinned and the policy sections for
  its return category, Assured badge and warranty are pulled in.
- **Grounded answers.** The model answers only from the supplied sections, says "I don't have that
  information" and offers a colleague when nothing covers the question, and lists the section IDs it
  used. Those IDs are saved with every answer.
- **Exact dates.** A `date_calculator` tool counts business days (Mon–Sat, skipping `holidays.csv`)
  and calendar days, with today fixed at 20 October 2026, so refund and return deadlines are computed,
  not estimated.

### Step 3: Hold a conversation
- Full message history is kept, so follow-ups ("how long will that take?") resolve against the order
  and item already discussed.
- The agent guides (STY, IDV, ESC) and the KB-00 rules are always in the system prompt, because they
  govern behaviour on every turn.
- Tools: `search_kb`, `lookup_product`, `date_calculator`, `check_pincode`, and the dataset's own
  `lookup_order` and `create_ticket`.
- **Rules enforced in code, not only in the prompt:**
  - `lookup_order` withholds order details until a phone suffix or pincode matches.
  - Our wrapper allows two verification attempts per order (IDV-3).
  - Verification values are redacted from every log and output file (IDV-4).
  - A handover counts only when `create_ticket` returns `created=True`.
  - There is no tool that cancels, refunds or credits anything, so the agent cannot claim to have done so.
- Each turn records a decision (answer, clarify, escalate or refuse), the sources used, and every tool
  call.

### Step 4: Give it a voice
The Step 3 agent is unchanged. `run.py` takes a folder of audio files; files in one sub-folder or named
`<call>_turn<N>` run as a single conversation. For each file it saves the transcript, the reply text,
the reply audio, the section IDs used with their text, and the tool calls.
- **Written for the ear:** replies are two or three short sentences without markdown. Before TTS, dates
  become "22 October 2026", rupee amounts are spoken as rupees, order IDs and refund references are
  read in groups of four digits, and ticket hashes are never read out.
- **Recovery:** empty or unclear transcripts get "Sorry, I didn't catch that, could you say it again?".
  WAV clips over the 60-second STT limit are split and transcribed in parts. A failure in STT, the LLM
  or TTS on one file is logged and the batch continues.
- **Measured latency:** Gnani TTS about 1.4 s and STT about 1.0 s per short clip; a Sarvam call about
  1.2 s.

## 2. Evaluation: Step 1 vs Step 2

**Method.** `python evaluate.py questions` runs all 52 practice questions through each mode. Scoring
follows the dataset's DATA_DICTIONARY:
- every `must_include` fact must appear (alternatives, equivalent date formats and Devanagari digits
  accepted) and no `must_not_include` claim may appear;
- the decision must be one of `acceptable_decisions`;
- an LLM judge compares the reply with `reference_answer`.

An answer is **correct** only when all three pass.

| Practice questions (n = 52) | Step 1: LLM only | Step 2: RAG | Full agent (Step 3) |
|---|---|---|---|
| **Correct** | **___ / 52** | **___ / 52** | ___ / 52 |
| Decision matches | ___ | ___ | ___ |
| Required facts present | ___ | ___ | ___ |
| Judge agrees | ___ | ___ | ___ |
| Gold sections retrieved | n/a | ___ | ___ |
| Correct on `asr_text` input | ___ | ___ | ___ |

**Retrieval alone** (top 8 plus the always-on agent-guide rules) finds **87.8%** of the gold evidence
IDs, on both the typed questions and the ASR-style transcripts. Number normalisation and product
awareness raised ASR recall from 73.3% to 87.8%.

**What we learned**
- **Without documents the LLM guesses plausibly and wrongly.** In the first run, Step 1 said wallet refunds
  take "3 to 5 working days"; the policy says instant, within 2 hours. With retrieval, Step 2 answered
  correctly and cited REF-2.
- **Retrieval is necessary but not sufficient.** Step 2 found the right Assured-warranty section but
  answered "yes, it's covered" without the 12-month figure. We added a rule to always state the
  concrete numbers a policy gives.
- **Date maths belongs in code.** Refund deadlines need business days that skip Sundays and the
  19 October warehouse holiday. The calculator tool removes that class of error.
- ___ _(add one or two observations from the full run's misses)_

## 3. Evaluation: conversations (Step 3)

`python evaluate.py conversations` replays the 23 practice conversations (96 customer turns) in turn
order. A turn passes when the decision matches, every expected tool call is made with the same order ID
and verification value, the required facts are present, and the judge agrees.

| Metric | Result |
|---|---|
| Turns correct | ___ / 96 |
| Decision matches | ___ |
| Expected tool calls made | ___ |
| Conversations fully correct | ___ / 23 |
| Weakest scenario | ___ |

## 4. What went wrong and what we would fix

- **Words vs documents.** Customers say "send it back" and "nobody came to collect it"; keyword search
  alone missed these. The multilingual dense model fixed most cases. A few sections are still missed
  (RET-4 for opened beauty products, DWM-3 for high-value damage); the agent's own `search_kb` call is
  the fallback. Next step: a small reranker over the top 20.
- **Hindi audio.** STT runs in one language per call. A Hindi-only caller on `en-IN` gets a romanised
  transcript and a Hinglish reply. Next step: detect the language on a first pass and re-run STT in that
  language.
- **Latency.** An earlier Gemini version took 5–14 s per turn, too slow for voice. Moving the LLM to
  Sarvam cut a call to about 1.2 s. Next step: stream TTS so speech starts before the full reply is ready.
- **Gnani API surprises.** WAV files from Gnani TTS carry a placeholder length header, which made our
  60-second check misread short clips as hours long; we now compute length from file size. The SDK also
  rejects `hi-en` for TTS despite the docs, so Hinglish is spoken with `en-IN`.
- ___ _(add the main failure from the full evaluation)_

## 5. Bonus work

- **Hindi and Hinglish end to end:** multilingual retrieval, replies in the caller's language and script,
  Devanagari replies spoken with Gnani's Hindi voice.
- **Live demo UI** (`python app.py`) with a panel showing the sections, tool calls and decision behind
  every reply.
- **Swappable LLM:** Sarvam by default, with Gemini or Claude selectable from `.env`.
- **Privacy:** verification values never appear in logs, outputs or ticket summaries.

## 6. Rules compliance

- Gnani is the only STT and TTS service, with no other speech service even as a fallback.
- Only the synthetic dataset is used.
- Answers are retrieved at run time; nothing is hard-coded per question.
- No API keys are in the repository (`.env` is git-ignored; `.env.example` has none).
