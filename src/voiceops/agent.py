"""The ZentroMart support agent.

One class, three modes, matching the capstone steps:
  baseline - Step 1: system prompt only, no documents, no tools.
  rag      - Step 2: retrieved KB sections injected each turn + date/catalog helpers.
  full     - Step 3: rag + conversation memory, search_kb, order lookup with caller
             verification, and ticket creation for human handover.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from . import config, llm
from .kb import always_on_rules
from .retriever import Hit, Retriever, get_retriever
from .toolkit import ToolRunner, compact, schemas_for

log = logging.getLogger(__name__)

META_RE = re.compile(r"<meta>(.*?)</meta>", re.S)
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
DECISIONS = ("answer", "clarify", "escalate", "refuse")
MAX_TOOL_ROUNDS = 10

ROLE = f"""You are {config.COMPANY}'s AI customer support assistant. {config.COMPANY} is an Indian online store \
selling groceries, electronics, fashion, home and beauty products. Customers reach you by chat or by \
voice; voice messages arrive as speech-to-text transcripts, so expect spelled-out numbers ("eight nine \
seven eight" = 8978), missing punctuation and misheard words. Interpret them sensibly; if a message is \
unintelligible, ask the customer to repeat it.

Today is {config.TODAY.strftime('%A, %d %B %Y')}. Use this date for every deadline and delay, never any other "today"."""

BASELINE_RULES = """Rules:
- In your first reply, say you are ZentroMart's AI support assistant.
- Help only with ZentroMart orders, deliveries, returns, refunds, cancellations, payments and warranty.
- You cannot cancel orders, issue refunds, give compensation or change anything yourself.
- Reply in the customer's language, in two or three short sentences of plain text."""

FULL_RULES = """How you work:
1. Ground every fact in the documents. Policy facts, numbers, fees, windows and timelines must come from \
the KNOWLEDGE sections supplied with the message{search_hint}. When two documents disagree, follow \
KB-00.1 precedence: current policy > agent guide > help > FAQ > marketing. A superseded document applies \
only to the cases its successor names (e.g. electronics delivered before 1 August 2026).
2. If the documents do not cover the question, say plainly that you don't have that information and offer \
to connect the customer with a human colleague. Never invent a policy, date, amount or service.
3. Use date_calculator for every date computation: refund timelines (business days from the initiation \
date), return deadlines (calendar days from delivery), delays. For a range such as "5 to 7 business days", \
give the latest date ("by 27 October 2026"), optionally the range. Always include the year in dates.
4. Product questions: use the catalog row (return_category, warranty_months, assured, fulfilled_by) together \
with the policy. Always state the concrete figures the documents give (e.g. "12-month Assured warranty", \
"within 2 hours", "₹49 fee", "7 days"), never just "yes, it's covered". If a needed detail is missing (which item, which payment method, delivery date), ask ONE \
short clarifying question instead of guessing.
5. Reply in the customer's language and script: Hindi in Devanagari -> Hindi in Devanagari; Hinglish in Latin \
script -> Hinglish in Latin script; English -> English. Switch if they switch.
6. Two or three short sentences, plain text only: no markdown, bullet points, emojis or URLs. Replies are \
also spoken aloud. Write amounts as "₹79".
7. Introduce yourself as ZentroMart's AI support assistant in your first reply of the conversation (STY-1).
8. Politely refuse anything a policy does not allow (compensation, vouchers, exceptions) and requests for \
another person's data. Never promise outcomes, amounts or times beyond what a policy states."""

ORDER_RULES = """Orders, verification and actions:
- General policy questions need no verification. Anything about a specific order does (IDV-1).
- When the customer gives an order ID, call lookup_order(order_id) right away, then ask for ONE of: the last \
4 digits of the registered phone number or the delivery pincode (IDV-2). Never say which value you expect.
- When they give it, call lookup_order(order_id, verification_value). If it fails, tell them it doesn't \
match and ask them to try once more. After a second failure share no order details: say you can't share \
them, suggest contacting support from the registered phone number, and offer general policy help (IDV-3).
- Once verified, remember the order for the rest of the conversation and resolve follow-ups ("how long \
will that take?", "that item") against it without asking again. Never read out the full phone number, \
email or address (IDV-4), and never repeat the verification value.
- You have no tool to cancel, refund, approve, credit or change an order. Explain how the customer can do \
it (e.g. My Orders in the app) and never claim to have done it.
- Escalate per ESC-1 (e.g. damaged/wrong item over ₹10,000, unrecognised order or payment, legal threats, \
contacts_last_7_days >= 2 on the same issue, refund more than 3 business days past its maximum timeline, \
customer asks for a human, safety risk). To escalate, call create_ticket FIRST with an ESC-3 summary, then \
tell the customer a human colleague will contact them. Say a ticket was created only if the tool returned \
created=true. Do not read out the ticket ID."""

OUTPUT_FORMAT = """Output format: write the reply to the customer, then on a final separate line exactly
<meta>{"decision": "<answer|clarify|escalate|refuse>", "sources": ["<SECTION-ID>", ...]}</meta>
decision: answer = you answered; clarify = you asked the customer for a detail (including verification); \
escalate = you handed over to a human; refuse = you declined a request policy doesn't allow.
sources: the IDs of the sections and catalog rows (e.g. "RET-2", "products.csv:ELE-0012", "STY-1") your \
reply actually relies on. The meta line is removed before the customer sees the reply."""


def build_system_prompt(mode: str, retriever: Retriever | None) -> str:
    if mode == "baseline":
        return f"{ROLE}\n\n{BASELINE_RULES}"
    rules = "\n\n".join(c.render() for c in always_on_rules(retriever.chunks))
    parts = [ROLE,
             "# Operating rules from ZentroMart's agent guides (cite their IDs when you apply them)\n" + rules,
             FULL_RULES.format(search_hint=" or from search_kb / lookup_product results")]
    if mode == "full":
        parts.append(ORDER_RULES)
    parts.append(OUTPUT_FORMAT)
    return "\n\n".join(parts)


@dataclass
class TurnResult:
    user_text: str
    reply: str
    decision: str
    sources: list[str]           # IDs the reply relies on (cited by the model)
    chunks_used: list[dict]      # the text of those sections / catalog rows
    retrieved: list[dict]        # everything retrieval put in front of the model
    tool_calls: list[dict]
    ticket_created: bool
    latency_s: float
    raw: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


@dataclass
class SupportAgent:
    mode: str = "full"
    retriever: Retriever | None = None
    messages: list = field(default_factory=list)
    turn: int = 0

    def __post_init__(self):
        if self.mode != "baseline" and self.retriever is None:
            self.retriever = get_retriever()
        self.system = build_system_prompt(self.mode, self.retriever)
        self.tools = schemas_for(self.mode)
        self.runner = ToolRunner(self.retriever) if self.retriever else None
        self._last_user = ""

    def reset(self):
        self.messages, self.turn, self._last_user = [], 0, ""
        self.runner = ToolRunner(self.retriever) if self.retriever else None

    # ------------------------------------------------------------------------------
    def respond(self, user_text: str) -> TurnResult:
        t0 = time.time()
        self.turn += 1
        if self.runner:
            self.runner.turn = self.turn
            self.runner.searched_ids = []
        hits = self._retrieve(user_text)
        self.messages.append({"role": "user", "content": self._user_block(user_text, hits)})
        self._last_user = user_text

        raw = self._loop()
        reply, meta = _split_meta(raw)
        reply = self._ensure_intro(reply)

        known = self.retriever.by_id if self.retriever else {}
        sources = [s for s in meta.get("sources", []) if s in known]
        if not sources and self.mode != "baseline":
            sources = [h.chunk.id for h in hits[:3]]
        retrieved_ids = {h.chunk.id for h in hits}
        retrieved = [h.as_dict() for h in hits]
        if self.runner:
            for cid in self.runner.searched_ids:
                if cid not in retrieved_ids and cid in known:
                    retrieved_ids.add(cid)
                    retrieved.append(Hit(known[cid], 0.0).as_dict())
        calls = [c.public() for c in self.runner.calls if c.turn == self.turn] if self.runner else []
        decision = meta.get("decision") if meta.get("decision") in DECISIONS else _guess_decision(reply, calls)
        return TurnResult(
            user_text=user_text, reply=reply, decision=decision, sources=sources,
            chunks_used=[Hit(known[s], 0.0).as_dict() for s in sources], retrieved=retrieved,
            tool_calls=calls, ticket_created=bool(self.runner and self.runner.ticket_created(self.turn)),
            latency_s=round(time.time() - t0, 2), raw=raw)

    # ------------------------------------------------------------------------------
    def _retrieve(self, user_text: str) -> list[Hit]:
        if self.mode == "baseline":
            return []
        # Prior customer message helps follow-ups like "and how long will that take?".
        query = f"{self._last_user}\n{user_text}" if self._last_user else user_text
        hits = self.retriever.search(query, k=config.TOP_K)
        return [h for h in hits if h.chunk.doc_type not in ("agent_guide", "meta")]

    def _user_block(self, user_text: str, hits: list[Hit]) -> str:
        if self.mode == "baseline":
            return user_text
        knowledge = "\n\n".join(h.chunk.render() for h in hits) or "(no matching sections)"
        return (f"<knowledge>\n{knowledge}\n</knowledge>\n\n"
                f"<customer_message>\n{user_text}\n</customer_message>")

    def _loop(self) -> str:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = llm.create(self.system, self.messages, self.tools or None)
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                return ("Sorry, I can't help with that here. I can connect you with a human colleague "
                        'if you like.\n<meta>{"decision": "refuse", "sources": []}</meta>')
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                return llm.text_of(resp)
            results = []
            for b in uses:
                out = self.runner.run(b.name, dict(b.input))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": compact(out),
                                "is_error": "error" in out})
            self.messages.append({"role": "user", "content": results})
        log.warning("Tool loop hit %d rounds", MAX_TOOL_ROUNDS)
        return "Sorry, that is taking me longer than expected. Let me connect you with a colleague."

    def _ensure_intro(self, reply: str) -> str:
        if self.turn != 1 or re.search(r"\bAI\b|एआई|artificial", reply, re.I):
            return reply
        intro = ("मैं ZentroMart की AI सहायक हूँ। " if DEVANAGARI_RE.search(self._last_user)
                 else "Hi, I'm ZentroMart's AI support assistant. ")
        return intro + reply


def _split_meta(raw: str) -> tuple[str, dict]:
    meta = {}
    m = META_RE.search(raw)
    if m:
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return META_RE.sub("", raw).strip(), meta


def _guess_decision(reply: str, calls: list[dict]) -> str:
    if any(c["name"] == "create_ticket" and c["output"].get("created") for c in calls):
        return "escalate"
    return "clarify" if reply.rstrip().endswith("?") else "answer"
