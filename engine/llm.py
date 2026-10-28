"""The one LLM call. It reads the computed plan and writes Farah's morning brief plus flags
on listings where the arithmetic may be misleading. It never returns a number that gets
applied, and every number it writes is checked against the numbers it was given.

Anthropic only, through the official SDK (`anthropic`). Settings, from the environment or .env:
    ANTHROPIC_API_KEY   required for the real brief; without it the engine's own verdict is shown
    LLM_MODEL           default claude-sonnet-5-5
    LLM_EFFORT          low | medium | high, default medium (how much the model reasons before writing)
    LLM_TIMEOUT_S       default 120: how long to wait for one attempt
The reply's shape is enforced by structured outputs (a JSON schema generated from BriefOut), then
checked again against BriefOut, so the instruction and the check can't drift apart.
"""

import json
import logging
import re

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import Config
from .decide import Plan, verdict_text
from .schema import CHANNEL_LABELS
from .settings import get_settings

log = logging.getLogger(__name__)


class BriefFlag(BaseModel):
    """One thing in the data a human should check, tied to a listing."""

    model_config = ConfigDict(extra="ignore")
    listing_id: str = ""
    note: str = Field(min_length=1)


class BriefOut(BaseModel):
    """The shape the model's reply must have. The structured-output schema sent with the request is
    generated from this same class (wire_schema), so the instruction and the check can't drift apart."""

    model_config = ConfigDict(extra="ignore")
    brief: str = Field(min_length=1)
    flags: list[BriefFlag] = Field(default_factory=list, max_length=12)
    questions: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("brief")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("brief is empty")
        return v


# Keywords structured outputs doesn't accept: length and range limits are checked client-side by BriefOut instead.
_NOT_ON_THE_WIRE = {
    "minLength",
    "maxLength",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


def wire_schema(model: type[BaseModel]) -> dict:
    """The JSON schema sent as structured output, generated from the pydantic model so the two can't
    drift: every object closed (additionalProperties false) with all its properties required, and the
    keywords structured outputs doesn't support left out (BriefOut still enforces them on the reply)."""

    def clean(node):
        if isinstance(node, list):
            return [clean(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k in _NOT_ON_THE_WIRE or (k == "title" and isinstance(v, str)):
                continue
            if k == "minItems" and isinstance(v, int) and v > 1:
                continue
            out[k] = {name: clean(sub) for name, sub in v.items()} if k in ("properties", "$defs") else clean(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return clean(model.model_json_schema())


BRIEF_SCHEMA = wire_schema(BriefOut)

SYSTEM = """You write the morning brief for Farah, a brand manager clearing last season's sneakers on Amazon.ae and Noon in 60 days.
You are given a plan computed by a deterministic engine. Your job is to explain it, not to change it.

Rules:
- Verdict first. Answer "will we make it?" in the first sentence, plainly, with the numbers given.
- Use only numbers that appear in the input. Never estimate, extrapolate or invent a figure.
- Plain language a busy brand manager reads in 60 seconds. No jargon, no headers, no bullet points in the brief.
- 3 to 5 sentences for the brief.
- Then flag anything in the data the arithmetic might be misreading (a reporting gap, a spike, a listing whose conversion is implausible, a parity collision coming next week, a style whose sales come from sizes that are nearly gone). Each flag names a listing id and says what a human should check. Flags are observations, never instructions to change a number.
- Up to 3 questions Farah should put to her team today.
Reply in the structured format requested: brief, flags (listing_id and note), questions."""


def _payload(plan: Plan, cfg: Config) -> dict:
    v = plan.verdict
    t = plan.table
    rows = []
    for _, r in t.iterrows():
        rows.append(
            {
                "listing_id": r["listing_id"],
                "name": r["style_name"],
                "channel": CHANNEL_LABELS.get(r["channel"], r["channel"]),
                "status": r["status"],
                "stock": int(r["on_hand"]),
                "required_per_day": round(float(r["required_rate"]), 2),
                "actual_per_day_7d": round(float(r["rr7"]), 2),
                "price": float(r["price_now"]),
                "floor": float(r["floor"]) if r["floor"] == r["floor"] else None,
                "discount_vs_mrp": round(float(r["discount"]), 2) if r["discount"] == r["discount"] else None,
                "sessions_per_day": round(float(r["sessions_7"]), 0),
                "conversion": round(float(r["cvr_mean"]), 4),
                "ad_spend_per_day": round(float(r["ad_now"]), 0),
                "price_change_slots_left": int(r["slots"]),
                "notes": list(r["notes"]),
            }
        )
    actions = [
        {"rank": a.rank, "headline": a.headline(), "impact_aed": round(a.impact_aed), "reason": a.reason, "flags": a.flags}
        for a in plan.changes
    ]
    watch = [{"listing_id": a.listing_id, "reason": a.reason} for a in plan.holds if a.kind == "watch"]
    return {
        "plan_date": str(plan.plan_date),
        "days_left": plan.days_left,
        "verdict_numbers": {
            "units_on_hand": v["on_hand"],
            "required_per_day": round(v["required_rate"], 1),
            "actual_per_day": round(v["actual_rate"], 1),
            "sell_through_if_nothing_changes": round(v["sell_through_hold"], 2),
            "sell_through_with_plan": round(v["sell_through_plan"], 2),
            "sell_through_range_with_plan": [round(x, 2) for x in v["plan_range"]],
            "units_sold_with_plan": round(v["plan"]["sold"]),
            "units_exited_now": round(v["plan"]["exits"]),
            "units_left_at_deadline": round(v["plan"]["unsold"]),
            "net_recovery_with_plan_aed": round(v["plan"]["net"]),
            "net_recovery_if_nothing_changes_aed": round(v["hold"]["net"]),
            "jobber_everything_today_aed": round(v["jobber_today"]),
            "listings_on_track": v["n_on_track"],
            "listings_behind": v["n_behind"],
            "listings_critical": v["n_critical"],
        },
        "ad_spend_per_day": {
            "plan": {CHANNEL_LABELS.get(k, k): round(x) for k, x in plan.ad_totals["plan"].items()},
            "today": {CHANNEL_LABELS.get(k, k): round(x) for k, x in plan.ad_totals["now"].items()},
            "caps": {CHANNEL_LABELS.get(k, k): round(x) for k, x in cfg.ad_budget.items()},
        },
        "verdict_text": verdict_text(v),
        "actions": actions,
        "watch": watch,
        "data_notes": plan.notes,
        "listings": rows,
        "rule_conflicts_needing_a_human": [f"{v.listing_id}: {v.detail}" for v in plan.conflicts],
    }


def _numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def _check_numbers(brief: str, payload: dict) -> list:
    """Every number in the brief must appear somewhere in what the model was given."""
    allowed = _numbers_in(json.dumps(payload))
    # percentages: 0.72 in the payload may be written as 72
    for n in list(allowed):
        try:
            f = float(n)
            if 0 < f <= 1:
                allowed.add(str(int(round(f * 100))))
        except ValueError:
            pass
    allowed |= {str(x) for x in range(0, 11)}  # small counts ("3 listings"); everything larger must come from the plan
    unverified = []
    for n in _numbers_in(brief):
        base = n.rstrip(".")
        if base not in allowed and base.split(".")[0] not in allowed:
            unverified.append(n)
    return unverified


DEFAULT_MODEL = "claude-sonnet-5-5"  # judgement in plain language at a fraction of Opus's cost; see DECISIONS.md
DEFAULT_EFFORT = "medium"  # explaining a computed plan is not deep reasoning; medium keeps the wait short
MAX_TOKENS = 8000  # thinking counts toward this; the brief itself is well under 1,000 tokens
MAX_RETRIES = 1  # the SDK retries connection errors, 408, 409, 429 and 5xx; once is enough for a page someone is waiting on


def _ask_claude(system: str, user: str, model: str, key: str, effort: str, timeout: float) -> str:
    """One Messages API call through the SDK, with the reply's shape enforced by structured outputs.
    Returns the JSON text of the reply."""
    client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=MAX_RETRIES)
    msg = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": BRIEF_SCHEMA}},
    )
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(f"the reply hit the {MAX_TOKENS}-token limit before the brief was finished; try LLM_EFFORT=low")
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined to write this brief (stop_reason: refusal)")
    return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")


def _explain_api_error(e: Exception, model: str, timeout: int) -> str:
    """The SDK's exception, as a sentence that says what to do about it."""
    if isinstance(e, anthropic.APITimeoutError):
        return (
            f"Claude didn't answer within {timeout} s (after {MAX_RETRIES} retry). Try again, set LLM_EFFORT=low, "
            f"or raise LLM_TIMEOUT_S in .env"
        )
    if isinstance(e, anthropic.APIConnectionError):
        return f"couldn't reach the Anthropic API ({e.__cause__ or e}); check the network or VPN"
    if isinstance(e, anthropic.AuthenticationError):
        return "the API key was rejected (401); check ANTHROPIC_API_KEY in .env"
    if isinstance(e, anthropic.PermissionDeniedError):
        return f"this key isn't allowed to use {model} (403); check the key's workspace or LLM_MODEL"
    if isinstance(e, anthropic.NotFoundError):
        return f"model '{model}' wasn't found for this key (404); check LLM_MODEL in .env"
    if isinstance(e, anthropic.RateLimitError):
        return "rate limit reached (429) even after a retry; wait a minute and refresh the brief"
    if isinstance(e, anthropic.APIStatusError):  # 400 (e.g. credit balance too low), 529 overloaded, ...
        return f"the API answered HTTP {e.status_code}: {getattr(e, 'message', e)}"
    return f"call failed: {e}"


def _parse(text: str) -> BriefOut:
    """Pull the JSON object out of the reply (models sometimes wrap it in prose or a code
    fence) and check it against BriefOut."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return BriefOut.model_validate_json(m.group(0) if m else text)


def template_brief(plan: Plan) -> dict:
    """What Farah sees when there is no key or the call fails: the deterministic verdict and the
    engine's own flags, assembled without a model."""
    flags = []
    for a in plan.actions:
        for f in a.flags:
            if "missing sessions" not in f:
                flags.append({"listing_id": a.listing_id, "note": f})
    for n in plan.notes:
        flags.append({"listing_id": "data", "note": n})
    top = plan.changes[:3]
    first = " ".join(f"{a.headline()}." for a in top)
    return {
        "brief": verdict_text(plan.verdict) + " Start with: " + first,
        "flags": flags[:8],
        "questions": [],
        "provider": "template",
        "error": None,
        "unverified_numbers": [],
    }


def _fallback(plan: Plan, error: str) -> dict:
    log.warning("brief: %s", error)
    out = template_brief(plan)
    out["error"] = error
    return out


def generate_brief(plan: Plan, cfg: Config) -> dict:
    """Ask Claude for Farah's brief, flags and questions about the computed plan.

    Never raises. With no key, an API or network error, or a reply in the wrong shape, it returns the engine's own
    verdict with the reason in 'error'. Every number in the brief is checked against the plan; any not found are
    listed in 'unverified_numbers'.
    """
    s = get_settings()
    key = s.anthropic_api_key.get_secret_value()
    model, effort = s.llm_model or DEFAULT_MODEL, s.llm_effort or DEFAULT_EFFORT
    if not key:
        return _fallback(plan, "no ANTHROPIC_API_KEY in .env; showing the engine's own verdict")
    payload = _payload(plan, cfg)
    user = "PLAN AS COMPUTED (JSON):\n" + json.dumps(payload, default=str)
    try:
        out = _parse(_ask_claude(SYSTEM, user, model, key, effort, float(s.llm_timeout_s)))
    except ValidationError as e:
        first = e.errors()[0]
        return _fallback(
            plan, f"Claude's reply wasn't in the expected shape ({'.'.join(map(str, first['loc'])) or 'reply'}: {first['msg']})"
        )
    except anthropic.APIError as e:
        return _fallback(plan, _explain_api_error(e, model, s.llm_timeout_s))
    except Exception as e:  # anything else: the page must still render
        return _fallback(plan, f"call failed: {e}")
    return {
        "brief": out.brief,
        "flags": [f.model_dump() for f in out.flags],
        "questions": out.questions,
        "provider": f"anthropic:{model} (effort {effort})",
        "error": None,
        "unverified_numbers": _check_numbers(out.brief, payload),
    }
