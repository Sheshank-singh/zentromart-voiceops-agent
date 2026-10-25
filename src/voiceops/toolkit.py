"""Tools the agent can call, and the code that runs them.

`lookup_order` and `create_ticket` are the dataset's own mock tools (imported from
<dataset>/tools); everything else here is deterministic helper code so the LLM never
does date arithmetic or table lookups in its head.
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import config
from .retriever import Retriever

# ---------------------------------------------------------------------------------
# Dataset mock tools
# ---------------------------------------------------------------------------------
_TOOLS_MODULE = None


def dataset_tools():
    """Import <dataset>/tools as a package so its relative data paths keep working."""
    global _TOOLS_MODULE
    if _TOOLS_MODULE is None:
        if not config.TOOLS_DIR.is_dir():
            raise FileNotFoundError(f"Dataset tools not found at {config.TOOLS_DIR}")
        sys.path.insert(0, str(config.DATASET_DIR))
        try:
            _TOOLS_MODULE = importlib.import_module("tools")
        finally:
            sys.path.remove(str(config.DATASET_DIR))
    return _TOOLS_MODULE


# ---------------------------------------------------------------------------------
# Calendar: business days are Mon-Sat, excluding data/holidays.csv (KB-00.3)
# ---------------------------------------------------------------------------------
def _load_holidays() -> dict[date, str]:
    path = config.DATA_DIR / "holidays.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {date.fromisoformat(r["date"]): r["name"] for r in csv.DictReader(fh)}


HOLIDAYS = _load_holidays()


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in HOLIDAYS


def add_business_days(start: date, n: int) -> tuple[date, list[str]]:
    d, counted, skipped = start, 0, []
    while counted < n:
        d += timedelta(days=1)
        if is_business_day(d):
            counted += 1
        else:
            skipped.append(f"{d.isoformat()} ({HOLIDAYS.get(d, 'Sunday')})")
    return d, skipped


def business_days_between(start: date, end: date) -> int:
    """Business days after `start` up to and including `end`."""
    sign = 1 if end >= start else -1
    lo, hi = sorted((start, end))
    n = sum(1 for i in range(1, (hi - lo).days + 1) if is_business_day(lo + timedelta(days=i)))
    return sign * n


def _fmt(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')}, {d.day} {d.strftime('%B %Y')})"


def date_calculator(operation: str, start_date: str, days: int | None = None,
                    days_max: int | None = None, end_date: str | None = None) -> dict:
    start = date.fromisoformat(start_date)
    if operation == "add_business_days":
        lo, skipped = add_business_days(start, int(days))
        out = {"start": _fmt(start), "result": _fmt(lo), "skipped_non_business_days": skipped}
        if days_max is not None:
            hi, skipped_hi = add_business_days(start, int(days_max))
            out = {"start": _fmt(start), "earliest": _fmt(lo), "latest": _fmt(hi),
                   "skipped_non_business_days": skipped_hi}
        return out
    if operation == "add_calendar_days":
        return {"start": _fmt(start), "result": _fmt(start + timedelta(days=int(days)))}
    if operation == "days_between":
        end = date.fromisoformat(end_date) if end_date else config.TODAY
        return {"start": _fmt(start), "end": _fmt(end), "calendar_days": (end - start).days,
                "business_days": business_days_between(start, end)}
    if operation == "day_info":
        return {"date": _fmt(start), "business_day": is_business_day(start),
                "holiday": HOLIDAYS.get(start)}
    return {"error": f"unknown operation {operation!r}"}


# ---------------------------------------------------------------------------------
# Pincodes
# ---------------------------------------------------------------------------------
def _load_pincodes() -> dict[str, dict]:
    path = config.DATA_DIR / "pincodes.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["pincode"]: r for r in csv.DictReader(fh)}


PINCODES = _load_pincodes()


def check_pincode(pincode: str) -> dict:
    pin = "".join(ch for ch in str(pincode) if ch.isdigit())
    row = PINCODES.get(pin)
    if not row:
        return {"pincode": pin, "listed": False,
                "note": "Not in the pincode table, so not serviceable (DEL-5)."}
    return {"listed": True, **row}


# ---------------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------------
TOOL_SCHEMAS = {
    "search_kb": {
        "name": "search_kb",
        "description": "Search ZentroMart's policy documents, help articles, FAQs and product catalog. "
                       "Returns sections with their section IDs. Write the query in English, using "
                       "policy vocabulary (e.g. 'reverse pickup fee', 'refund timeline card').",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"]},
    },
    "lookup_product": {
        "name": "lookup_product",
        "description": "Find catalog rows (SKU, return_category, price, seller, assured, warranty_months) "
                       "by SKU like ELE-0012 or by product name.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"]},
    },
    "date_calculator": {
        "name": "date_calculator",
        "description": "Exact date arithmetic on the ZentroMart calendar (business days = Mon-Sat, "
                       "excluding warehouse holidays). Always use this for deadlines and timelines. "
                       "Operations: add_business_days (start_date + days; pass days_max too for a range "
                       "such as '5 to 7 business days'), add_calendar_days (return windows), "
                       "days_between (start_date to end_date, default today), day_info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add_business_days", "add_calendar_days",
                                                         "days_between", "day_info"]},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "days": {"type": "integer"},
                "days_max": {"type": "integer"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["operation", "start_date"],
        },
    },
    "check_pincode": {
        "name": "check_pincode",
        "description": "Check whether a 6-digit pincode is serviceable and whether COD and Express are "
                       "available there.",
        "input_schema": {"type": "object", "properties": {"pincode": {"type": "string"}},
                         "required": ["pincode"]},
    },
    "lookup_order": {
        "name": "lookup_order",
        "description": "Look up an order. Call with order_id only as soon as the customer gives an order "
                       "ID (returns a verification prompt, no details). Then call again with "
                       "verification_value = what the customer said: last 4 digits of the registered "
                       "phone OR the 6-digit delivery pincode. Details are returned only when verified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "8 digits, spaces removed"},
                "verification_value": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
    "create_ticket": {
        "name": "create_ticket",
        "description": "Create a human-handover ticket (ESC-2). The summary must state the issue, intent, "
                       "order ID, what the customer said, what you checked and the escalation reason "
                       "(ESC-3). Never include verification values. Only tell the customer a ticket "
                       "exists if this returns created=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "order_id": {"type": "string"},
                "reason": {"type": "string", "description": "short code, e.g. high_value_damage, "
                                                            "human_requested, repeat_contact"},
                "intent": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
}

TOOLSETS = {
    "baseline": [],
    "rag": ["search_kb", "lookup_product", "date_calculator"],
    "full": ["search_kb", "lookup_product", "date_calculator", "check_pincode", "lookup_order",
             "create_ticket"],
}

MAX_VERIFICATION_ATTEMPTS = 2  # IDV-3: one retry


@dataclass
class ToolCall:
    name: str
    input: dict
    output: dict
    turn: int

    def public(self) -> dict:
        """Log-safe view: verification values never leave the process (README / IDV-4)."""
        inp = dict(self.input)
        if inp.get("verification_value"):
            inp["verification_value"] = "<redacted>"
        return {"name": self.name, "input": inp, "output": self.output, "turn": self.turn}


@dataclass
class ToolRunner:
    retriever: Retriever
    calls: list[ToolCall] = field(default_factory=list)
    failed_verifications: dict[str, int] = field(default_factory=dict)
    searched_ids: list[str] = field(default_factory=list)
    turn: int = 0

    def run(self, name: str, args: dict) -> dict:
        try:
            out = getattr(self, f"_t_{name}")(**args)
        except Exception as exc:  # bad args from the model: report, don't crash the call
            out = {"error": f"{type(exc).__name__}: {exc}"}
        self.calls.append(ToolCall(name, args, out, self.turn))
        return out

    def _t_search_kb(self, query: str) -> dict:
        hits = self.retriever.search(query, k=5)
        self.searched_ids += [h.chunk.id for h in hits]
        return {"results": [{"id": h.chunk.id, "section": h.chunk.render()} for h in hits]}

    def _t_lookup_product(self, query: str) -> dict:
        rows = self.retriever.find_products(query)
        self.searched_ids += [c.id for c in rows]
        return {"results": [{"id": c.id, "row": c.text} for c in rows] or "no matching product"}

    def _t_date_calculator(self, **kw) -> dict:
        return date_calculator(**kw)

    def _t_check_pincode(self, pincode: str) -> dict:
        return check_pincode(pincode)

    def _t_lookup_order(self, order_id: str, verification_value: str | None = None) -> dict:
        oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        if verification_value and self.failed_verifications.get(oid, 0) >= MAX_VERIFICATION_ATTEMPTS:
            return {"verified": False, "attempts_exhausted": True,
                    "message": "Verification already failed twice (IDV-3). Do not share order details; "
                               "suggest contacting support from the registered phone number."}
        result = dataset_tools().lookup_order(oid, verification_value=verification_value)
        if verification_value and not result.get("verified"):
            n = self.failed_verifications[oid] = self.failed_verifications.get(oid, 0) + 1
            result = {**result, "failed_attempts": n,
                      "retry_allowed": n < MAX_VERIFICATION_ATTEMPTS}
        if result.get("verified"):
            result = _enrich_order(result)
        return result

    def _t_create_ticket(self, summary: str, order_id: str | None = None, reason: str | None = None,
                         intent: str | None = None) -> dict:
        oid = "".join(ch for ch in str(order_id) if ch.isdigit()) if order_id else None
        return dataset_tools().create_ticket(summary, order_id=oid, reason=reason, intent=intent)

    # ---- helpers for callers -------------------------------------------------------
    def ticket_created(self, turn: int | None = None) -> bool:
        return any(c.name == "create_ticket" and c.output.get("created")
                   for c in self.calls if turn is None or c.turn == turn)


def _enrich_order(order: dict) -> dict:
    """Add policy-neutral date facts so the model doesn't miscount."""
    facts = {"today": _fmt(config.TODAY)}
    for key in ("delivered_on", "shipped_on", "promised_date"):
        if order.get(key):
            d = date.fromisoformat(str(order[key])[:10])
            facts[f"calendar_days_since_{key}"] = (config.TODAY - d).days
    if order.get("placed_at"):
        facts["placed_on"] = str(order["placed_at"])[:10]
    return {**order, "computed": facts}


def schemas_for(mode: str) -> list[dict]:
    return [TOOL_SCHEMAS[n] for n in TOOLSETS[mode]]


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
