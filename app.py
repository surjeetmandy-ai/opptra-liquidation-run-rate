"""Farah's morning page.

    streamlit run app.py

One screen: the verdict, the brief, the actions with Apply, and a sidebar to feed in a new
day's data or stretch the conditions (budget, deadline, jobber rate, fees).
"""

from datetime import timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from engine.actions import price_only
from engine.apply import apply_action, score_predictions
from engine.config import Config
from engine.decide import build_plan, verdict_text
from engine.llm import generate_brief
from engine.load import build_dataset, identify_kind, merge_new_history, read_uploaded, replace_constraints, replace_inventory
from engine.schema import AMAZON, CHANNEL_LABELS, NOON
from engine.session import check_upload_dates, clear_session, load_session, persist_session, session_data_dir
from engine.settings import DATA_DIR, STATE_PATH, configure_logging, get_settings
from engine.state import State

try:
    SETTINGS = get_settings()
except ValidationError as e:  # a typo in .env should say so, not show a stack trace
    st.error(
        "Your .env has a setting the app can't use:\n\n"
        + "\n".join(f"- {'.'.join(map(str, x['loc'])).upper()}: {x['msg']}" for x in e.errors())
    )
    st.stop()
configure_logging()
DATA = DATA_DIR

st.set_page_config(page_title="Liquidation Run-Rate", page_icon="👟", layout="wide")


# ------------------------------------------------------------------------------------------
# session
# ------------------------------------------------------------------------------------------
def boot() -> None:
    """Load the data, the saved state and the settings once per browser session, and build the first plan."""
    if "ds" in st.session_state:
        return
    try:
        ds, state = load_session(DATA, STATE_PATH)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    cfg = Config.from_dict(state.config_overrides)
    st.session_state.update(ds=ds, state=state, cfg=cfg, log=[], brief=None, brief_stale=False)
    rebuild(with_brief=True)


def rebuild(with_brief: bool = False) -> None:
    """Re-plan from the current data, state and config, and save the state.

    with_brief=True also asks the LLM for a fresh brief; otherwise the brief on screen is marked stale.
    """
    ss = st.session_state
    ss.plan = build_plan(ss.ds, ss.state, ss.cfg)
    ss.state.save(STATE_PATH)
    if with_brief:
        ss.brief = generate_brief(ss.plan, ss.cfg)
        ss.brief_stale = False
    else:
        ss.brief_stale = True


def do_apply(actions) -> None:
    """Apply the given actions (state change and stock moves), save, and re-plan."""
    ss = st.session_state
    actions = with_parity_partners(actions, ss.plan)
    for a in actions:
        done = apply_action(ss.ds, ss.state, a, ss.plan.plan_date, ss.plan.sizes[a.listing_id]["sizes"])
        ss.log.extend(done)
    ss.state.save(STATE_PATH)
    rebuild(with_brief=False)


def with_parity_partners(actions, plan) -> list:
    """A price move on one channel is never applied alone: the other channel's price move for the
    same style (made to keep the ±5% parity) goes with it, so the applied state never breaks the
    rule between two clicks. Only the partner's price comes along; its ad change or exit stays on
    the page for its own click."""
    chosen = {a.listing_id: a for a in actions}
    for a in list(actions):
        if a.price_to == a.price_from:
            continue
        for b in plan.actions:
            if (
                b.style_id == a.style_id
                and b.listing_id != a.listing_id
                and b.price_to != b.price_from
                and b.listing_id not in chosen
            ):
                chosen[b.listing_id] = price_only(b)
                st.session_state.log.append(f"{b.listing_id}: price applied with {a.listing_id} to keep parity")
    return list(chosen.values())


boot()
ss = st.session_state
plan, cfg, v = ss.plan, ss.cfg, ss.plan.verdict
day_n = cfg.horizon_days - plan.days_left + 1

# ------------------------------------------------------------------------------------------
# sidebar: new data, conditions, reset
# ------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### New day's data")
    st.caption(
        "Drop today's export (daily history rows for the new date). A fresh inventory or constraints file is also accepted and replaces the current one."
    )
    files = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True, label_visibility="collapsed"
    )
    fresh = st.checkbox("Start fresh from these files (new dataset; clears everything applied)", value=False)
    if files and st.button("Load", type="primary"):
        msgs = []
        if fresh:
            kinds = {}
            for f in files:
                df = read_uploaded(f.getvalue(), f.name)
                kinds[identify_kind(df)] = df
            if {"history", "inventory", "constraints"} <= set(kinds):
                clear_session(STATE_PATH)
                ss.ds = build_dataset(kinds["inventory"], kinds["history"], kinds["constraints"])
                ss.state = State(campaign_start=ss.ds.today + timedelta(days=1))
                ss.cfg = Config()
                persist_session(ss.ds, ss.state)
                ss.log = [f"fresh start from {len(files)} files; data through {ss.ds.today}"] + ss.ds.issues
                rebuild(with_brief=True)
                st.rerun()
            else:
                st.warning(
                    "A fresh start needs all three files (history, inventory, constraints). Loading what you gave as updates instead."
                )
        for f in files:
            df = read_uploaded(f.getvalue(), f.name)
            kind = identify_kind(df)
            try:
                if kind == "history":
                    check_upload_dates(df, ss.ds.today)
                    ss.ds, issues = merge_new_history(ss.ds, df)
                elif kind == "inventory":
                    ss.ds, issues = replace_inventory(ss.ds, df)
                elif kind == "constraints":
                    ss.ds, issues = replace_constraints(ss.ds, df)
                else:
                    issues = [
                        f"{f.name}: could not tell what this file is from its columns ({', '.join(map(str, df.columns[:6]))}…)"
                    ]
                    kind = "?"
                msgs.append(f"{f.name}: read as {kind}")
                msgs.extend(issues)
            except Exception as e:
                msgs.append(f"{f.name}: not loaded — {e}")
        score_predictions(ss.ds, ss.state)
        persist_session(ss.ds, ss.state)
        ss.log.extend(msgs)
        rebuild(with_brief=True)
        st.rerun()

    st.markdown("### Stretch the conditions")
    with st.form("conditions"):
        horizon = st.number_input("Deadline (days from day 1)", 10, 180, int(cfg.horizon_days))
        jobber = st.slider("Jobber pays (% of cost)", 5, 60, int(round(cfg.jobber_pct * 100)))
        cap_a = st.number_input("Amazon.ae daily ad cap (AED)", 0, 10000, int(cfg.ad_budget.get(AMAZON, 1500)), step=50)
        cap_n = st.number_input("Noon daily ad cap (AED)", 0, 10000, int(cfg.ad_budget.get(NOON, 1000)), step=50)
        fee = st.slider("Marketplace fee (% of price)", 0, 30, int(round(cfg.marketplace_fee_pct * 100)))
        band = st.slider("Noon parity band vs Amazon.ae (±%)", 1, 15, int(round(cfg.parity_band * 100)))
        changes = st.number_input("Max price changes per listing per rolling 7 days", 0, 7, int(cfg.max_price_changes))
        st.caption("Floors and MRPs change by uploading a new constraints file above.")
        if st.form_submit_button("Re-plan with these"):
            cfg.horizon_days = int(horizon)
            cfg.jobber_pct = jobber / 100
            cfg.ad_budget = {AMAZON: float(cap_a), NOON: float(cap_n)}
            cfg.marketplace_fee_pct = fee / 100
            cfg.parity_band = band / 100
            cfg.max_price_changes = int(changes)
            ss.state.config_overrides = cfg.to_dict()
            rebuild(with_brief=True)
            st.rerun()

    st.markdown("### Housekeeping")
    if st.button("Refresh the brief"):
        ss.brief = generate_brief(plan, cfg)
        ss.brief_stale = False
        st.rerun()
    if st.button("Reset everything applied"):
        clear_session(STATE_PATH)
        for k in ["ds", "state", "cfg", "plan", "log", "brief", "brief_stale"]:
            if k in ss:
                del ss[k]
        st.rerun()
    st.caption(
        f"LLM: {SETTINGS.describe_llm()} · state: {'saved' if STATE_PATH.exists() else 'fresh'}"
        + (" · data: uploaded session" if session_data_dir(DATA) != DATA else " · data: originals")
    )

# ------------------------------------------------------------------------------------------
# header + verdict
# ------------------------------------------------------------------------------------------
st.markdown(f"## Stride & Co. liquidation · plan for {plan.plan_date.strftime('%a %d %b')}")
st.caption(
    f"Day {day_n} of {cfg.horizon_days} · {plan.days_left} days to the jobber deadline · data through {ss.ds.today} · "
    f"{v['on_hand']:,} pairs on hand"
)

h, p = v["hold"], v["plan"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Sell-through by deadline",
    f"{v['sell_through_plan']:.0%}",
    f"{(v['sell_through_plan'] - v['sell_through_hold']) * 100:+.0f} pts vs doing nothing",
)
c2.metric("Net recovery (AED)", f"{p['net']:,.0f}", f"{p['net'] - h['net']:+,.0f} vs doing nothing")
c3.metric("Jobber everything today", f"AED {v['jobber_today']:,.0f}", "the floor we must beat", delta_color="off")
c4.metric(
    "Run-rate (pairs/day)",
    f"{v['actual_rate']:.0f} actual",
    f"{v['actual_rate'] - v['required_rate']:+.0f} vs {v['required_rate']:.0f} needed",
)

brief = ss.brief or {}
box = st.container(border=True)
with box:
    st.markdown("#### Will we make it?")
    if brief.get("brief"):
        st.write(brief["brief"])
    else:
        st.write(verdict_text(v))
    tags = []
    if brief.get("provider"):
        tags.append(f"brief by {brief['provider']}")
    if ss.brief_stale:
        tags.append("plan changed since this brief was written; refresh it from the sidebar")
    if brief.get("unverified_numbers"):
        tags.append("numbers not found in the plan: " + ", ".join(brief["unverified_numbers"]))
    if brief.get("error"):
        tags.append(brief["error"])
    if tags:
        st.caption(" · ".join(tags))
    cols = st.columns(2)
    with cols[0]:
        if brief.get("flags"):
            st.markdown("**Worth a human look**")
            for f in brief["flags"][:8]:
                st.markdown(f"- `{f.get('listing_id', '')}` {f.get('note', '')}")
    with cols[1]:
        if brief.get("questions"):
            st.markdown("**Ask the team today**")
            for q in brief["questions"][:3]:
                st.markdown(f"- {q}")
        st.markdown("**Ad spend tomorrow**")
        for ch, val in plan.ad_totals["plan"].items():
            st.markdown(
                f"- {CHANNEL_LABELS.get(ch, ch)}: AED {val:,.0f}/day (today {plan.ad_totals['now'][ch]:,.0f}, cap {cfg.ad_budget.get(ch, 0):,.0f})"
            )

if plan.conflicts:
    st.warning(
        "Rules that are already broken in the data and can't be fixed legally today (no price-change slot left). "
        "These need your decision: " + "; ".join(f"{x.listing_id}: {x.detail}" for x in plan.conflicts)
    )
if plan.violations:
    st.error(
        "The plan could not be made legal, so nothing can be applied: "
        + "; ".join(f"{x.listing_id} {x.rule}: {x.detail}" for x in plan.violations)
    )
can_apply = not plan.violations

# ------------------------------------------------------------------------------------------
# actions
# ------------------------------------------------------------------------------------------
changes = plan.changes
top = changes[: cfg.max_actions]
rest = changes[cfg.max_actions :]

hdr = st.columns([6, 1])
hdr[0].markdown(f"#### Do today — {len(top)} actions, ranked by AED impact")
if changes and can_apply and hdr[1].button(f"Apply all {len(changes)}", type="primary"):
    do_apply(changes)
    st.rerun()


def action_row(a) -> None:
    """Render one action card: rank, headline, reasons, flags, AED impact and its Apply button."""
    with st.container(border=True):
        c = st.columns([0.6, 6.4, 1.4, 1.1])
        c[0].markdown(f"**#{a.rank}**")
        c[1].markdown(f"**{a.headline()}**")
        c[1].caption(a.reason)
        if a.flags:
            c[1].caption("⚑ " + " · ".join(a.flags))
        c[2].metric("impact", f"{a.impact_aed:+,.0f}", label_visibility="collapsed")
        if can_apply and c[3].button("Apply", key=f"apply_{a.listing_id}"):
            do_apply([a])
            st.rerun()


if not changes:
    st.success("Nothing to change today. Every listing is either on track or has no legal move that beats holding.")
for a in top:
    action_row(a)

if rest:
    with st.expander(f"Also recommended, lower impact ({len(rest)})"):
        for a in rest:
            action_row(a)

holds = plan.holds
with st.expander(f"Holding / watching ({len(holds)})"):
    for a in holds:
        st.markdown(f"- **{a.style_id} {a.style_name} · {CHANNEL_LABELS.get(a.channel, a.channel)}** — {a.reason}")

# ------------------------------------------------------------------------------------------
# detail
# ------------------------------------------------------------------------------------------
with st.expander("All listings — the numbers behind the plan"):
    t = plan.table.copy()
    show = pd.DataFrame(
        {
            "listing": t["style_id"] + " " + t["style_name"],
            "channel": t["channel"].map(CHANNEL_LABELS),
            "status": t["status"],
            "stock": t["on_hand"],
            "need/day": t["required_rate"].round(2),
            "actual/day": t["rr7"].round(2),
            "price": t["price_now"],
            "floor": t["floor"],
            "off MRP": (t["discount"] * 100).round(0),
            "sessions/day": t["sessions_7"].round(0),
            "conversion %": (t["cvr_mean"] * 100).round(2),
            "ads/day": t["ad_now"].round(0),
            "ads per sale": t["ad_per_unit"].replace([float("inf")], float("nan")).round(0),
            "slots": t["slots"],
            "elasticity": t["elasticity"].round(2),
        }
    )
    st.dataframe(show, hide_index=True)

    pick = st.selectbox("Listing detail", options=list(plan.table["listing_id"]), format_func=lambda x: x.replace(":", " · "))
    if pick:
        s, ch = pick.split(":")
        hist = ss.ds.history[(ss.ds.history["style_id"] == s) & (ss.ds.history["channel"] == ch)].sort_values("date")
        hist = hist.set_index(pd.to_datetime(hist["date"]))
        cc = st.columns(3)
        cc[0].line_chart(hist[["units_sold"]], height=180)
        cc[1].line_chart(hist[["sessions", "ad_clicks"]], height=180)
        cc[2].line_chart(hist[["price_aed"]], height=180)
        r = plan.table.set_index("listing_id").loc[pick]
        sz = plan.sizes[pick]
        st.caption(
            "Stock by size: "
            + ", ".join(f"EU {a}: {int(b)}" for a, b in zip(sz["sizes"], sz["on_hand"], strict=True))
            + f" · conversion prior {r['cvr_prior']:.2%} on {r['cvr_obs_sessions']:.0f} observed sessions → {r['cvr_mean']:.2%} "
            f"(range {r['cvr_p10']:.2%}–{r['cvr_p90']:.2%}) · elasticity {r['elasticity']:.2f} ± {r['elasticity_sd']:.2f}"
        )

with st.expander("Learning log and data notes"):
    preds = ss.state.predictions
    if preds:
        pdf = pd.DataFrame(preds)
        st.dataframe(pdf, hide_index=True)
        scored = pdf[pdf["actual_rate"].notna()]
        if len(scored):
            err = scored["actual_rate"] - scored["predicted_rate"]
            st.caption(
                f"{len(scored)} predictions scored so far · mean error {err.mean():+.2f} pairs/day · "
                f"{(scored['actual_rate'] >= scored['baseline_rate']).mean():.0%} of applied actions beat their baseline"
            )
    else:
        st.caption(
            "Apply an action, then load the next day's data: predicted vs actual lands here and tightens the confidence band."
        )
    for n in plan.notes:
        st.markdown(f"- {n}")
    if ss.state.exits:
        st.markdown(f"**Exited so far:** {sum(e['units'] for e in ss.state.exits)} pairs for AED {ss.state.exited_value():,.0f}")
    if ss.log:
        st.markdown("**This session**")
        for line in ss.log[-20:]:
            st.markdown(f"- {line}")
