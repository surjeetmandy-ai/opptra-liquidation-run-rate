# How I know it works

`make verify` runs everything below in one go: ruff (lint + format), 117 tests, 15 awkward-input
scenarios and 8 condition-change drills. CI runs the same on every push. The question this file
answers is what passing actually proves, rule by rule, and what it doesn't.

Three layers of tests, each catching what the others can't:

| Layer | File | What it proves |
|---|---|---|
| **The brief, as an independent checker** | `tests/test_invariants.py` | Every plan satisfies the seven hard rules, judged by a checker written from the brief that imports nothing from the engine's guardrail. Run on the default plan, on 12 random condition sets (deadline 15–90 days, jobber 5–60%, caps 0–2,500, fee 0–20%, band 2–10%, 1–3 changes a week) and across a 4-day apply → simulate → merge → re-plan loop. Includes a self-test that deliberately breaks each rule to prove the checker fires. Same inputs → same plan. |
| **Each rule at its boundary** | `tests/test_rules.py` | Hand-built fixtures at the exact edge of every rule and threshold: parity at 5.0% passes, 5.1% fails; budget at the cap passes, one fil over fails; the 7-day window counts a change 6 days ago and not one 7 days ago; cut candidates are exactly the 5-AED steps between 5% and 20% and never below the floor; the config defaults are the brief's numbers. An independent mutation audit found that half of 51 realistic bugs survived the suite before this file; each test is named for the bug it kills. |
| **The engine on the real data** | `tests/test_engine.py` | Loading, the guardrail, the day-1 decisions worth defending (S19 checked before any exit; S14 Noon's ads raised; followers at the shallowest in-band price), the condition drills, the LLM layer's fallback and number check, settings, state, and a regression test for every bug two independent reviews found. |

## Requirement → code → test

**Hard rules (a recommendation that breaks one is a failure)**

| Rule | Enforced in | Proven by |
|---|---|---|
| Price ≥ floor, ≤ MRP | `decide._cut_candidates`, `_follower_price`, `_fix_rule_breaks`; `guardrail.check_plan` | `test_rules.TestGuardrailBoundaries.test_floor_and_mrp_edges`; `test_invariants` rule 1 + self-test (floor, mrp); `test_engine.test_floor_is_caught`, `test_listing_at_floor_is_never_cut`, `test_floor_raised_overnight_is_fixed_first` |
| Noon within ±5% of Amazon.ae, after both channels move | `decide._price_options`, `_follower_price`, `_fix_rule_breaks`; `guardrail` rule 2 | `test_rules…test_parity_at_exactly_the_band…`, `…judged_after_both_channels_move`; `test_invariants` rule 2 + self-test; `test_engine.test_parity_is_caught_and_counts_both_channels`, `test_parity_follower_takes_the_shallowest_in_band_price`, `test_parity_broken_in_data_is_fixed` |
| Daily ad cap per channel (incl. rounding, caps below today's spend) | `allocate.allocate_ads` (cap step, rounding guard); `decide._finalise_ads`, `_apply_guardrail`; `guardrail` rule 3 | `test_rules…test_budget_at_the_cap_passes_and_one_fil_over_fails`, `TestAllocator.test_when_the_cap_drops_below_todays_spend…`; `test_invariants` rule 3 (12 random cap pairs incl. 0); `test_engine.test_half_budget`, `test_zero_budget_channel`, `test_guardrail_withdraws_a_budget_breach` |
| ≤ 2 price changes per listing per rolling 7 days, counting history, applied and parity-forced changes once each | `estimate` (slots: `price_change_dates_from_history`, `merge_change_dates`), `decide._can_move_price`; `guardrail` rule 4 | `test_rules.TestRollingWindow` (6 days in, 7 days out, plan date counts, window setting honoured); `test_engine.test_change_limit_rolling_window`, `test_parity_forced_change_counts`, `test_history_price_change_counts_toward_slots`, `test_applied_change_seen_in_export_a_day_later_counts_once`, `test_any_real_price_move_counts_but_rounding_noise_does_not`; `test_invariants` rule 4 across the 4-day loop |
| Never exit more than is on hand, per size | `decide._size_exit`, `apply._remove_units`; `guardrail` rule 5 | `test_rules…test_exits_up_to_stock_pass_and_one_more_fails`, `TestApplyBySize`; `test_invariants` rule 5 + self-test |
| Holding AED 3/unit/month; jobber 25% of cost; 60 days from tomorrow | `config.Config` defaults; `economics.project`; `estimate.days_left`; `session.load_session` | `test_rules.TestConfigMatchesBrief`; `test_engine.test_projection_matches_day_loop` (closed form = day loop, 400 random cases); `test_invariants` loop asserts `days_left` counts down from 60 |
| One decision per listing per day; a same-day re-plan proposes nothing | `estimate` (`locked`), `allocate`, `decide` | `test_invariants` loop (4 same-day re-plans), `test_engine.test_apply_then_replan_does_not_repeat` |
| The plan never *makes* a pre-existing breach worse; it fixes it or declares a conflict | `decide._fix_rule_breaks`, `_apply_guardrail` | `test_invariants` rule 7; `test_engine.test_unfixable_rule_break_is_a_conflict_not_a_violation`, `test_a_floor_above_the_mrp_is_a_conflict_not_a_rule_fix` |
| The generator proposes only legal moves (the guardrail is a net, not a crutch) | all of `decide` | `test_rules…test_the_generator_never_relies_on_the_guardrail`; `test_invariants` asserts no "guardrail withdrew" note under any of the 12 random conditions |

**Decision rules (the judgement the panel grades)**

| Rule | Code | Proven by |
|---|---|---|
| Hold is the default; a price move must beat it by AED 300 per moved listing, an ad change by AED 150 | `config`; `decide._choose_prices`, `_recheck_price_gate`, `_finalise_ads` | `test_rules…test_the_price_gate_holds_a_style_whose_cut_does_not_pay_enough`; `test_engine.test_no_ad_change_loses_money_at_the_plan_price`, `test_every_action_beats_hold_or_is_a_parity_follower` |
| Cut sizing: smallest cut that closes the gap; else best projected recovery for a critical style; else ~10% first move; 5-AED steps, 5–20%, never below floor, 3-day cooldown | `decide._cut_candidates`, `_pick` | `test_rules.TestPriceCandidatesAndPick` (exact candidate lists, all three pick modes) |
| Parity follower takes the shallowest in-band price | `decide._follower_price` | `test_engine.test_parity_follower_takes_the_shallowest_in_band_price` (every follower on the real data) |
| Ads: stop only when no spend between zero and today's beats zero; otherwise trim to the spend the projection values highest; raises AED 10 at a time to the best marginal listing, never beyond 2.5× / AED 60, never to an on-track or broken listing | `economics.best_spend_up_to`, `should_stop_ads`; `allocate.allocate_ads` | `test_rules.TestAllocator` (4 synthetic cases); `test_engine.test_ads_are_trimmed_to_the_spend_the_projection_values_highest_not_switched_off`, `test_stop_reasons_tell_the_truth`, `test_a_sold_out_listing_with_a_budget_still_set_gets_it_switched_off`, `test_a_listing_nobody_has_seen_is_checked_not_advertised`, `test_starved_converter_gets_ads` |
| Exits: only what won't sell by the deadline even on the optimistic read (p90 conversion, elasticity +1 SD), at the price the listing is heading for (today's if it isn't being cut, the deepest legal price if it is); ≥ 10 pairs or 10%; a broken listing is checked by a human first and its exit waits 3 days | `decide._size_exit`, `_deepest_legal_price`; `state.checks`; `apply.apply_action` | `test_engine.test_broken_listing_gets_ads_stopped_and_a_check_before_any_exit`, `test_exits_are_read_at_the_price_the_listing_is_heading_for`, `test_shorter_deadline_means_more_urgency`, `test_past_the_deadline_everything_left_goes_to_the_jobber` |
| Diagnosis gates the lever (visibility → ads; price → cut; both → price first; broken → stop + check; new → hold) | `decide.diagnose`, `PRICE_LEVER_KINDS` | `test_engine.test_new_listing_without_history`, `test_broken_listing…`; `test_rules.TestEstimator.test_status_thresholds_are_one_and_a_half` |
| Conversion shrunk toward category × channel peers; impossible-session days excluded from conversion but their orders kept; run-rate = trailing 7-day mean | `estimate.estimate_listings`; `load.clean_history` | `test_rules.TestEstimator` (run-rate, invalid days = a missing day for conversion, new listing borrows the other channel's price); `test_engine.test_real_data_loads_and_flags_noon_outage` |
| Learning: each applied action logs predicted vs baseline; once 4+ have 3+ days of actuals, a lever's uplift is re-weighted by Σ(actual−baseline)/Σ(predicted−baseline), shrunk toward 1, bounded 0.5–1.5, applied to the uplift only; a combined price+ads action scores the price lever only | `estimate.lever_calibration`, `_scores_lever`; `economics.expected_rate`; `apply.score_predictions` | `test_engine.test_calibration_weighs_uplift_over_baseline_and_shrinks_toward_one`, `test_scoring_window_closes_at_the_listings_next_action`; `test_rules.TestEstimator.test_calibration_weights_scale_the_uplift_only`, `test_calibration_is_bounded` |
| Impact and verdict are computed from what will actually be applied (after the guardrail) | `decide._project_action`, `_verdict` | `test_engine.test_impact_is_computed_from_what_will_actually_be_applied`; `test_rules.TestVerdict`; `TestGuardrailWithdrawal` |
| The LLM writes, never decides; every number in the brief is checked; a failed call falls back to the engine's verdict | `llm.generate_brief`, `_check_numbers`, `template_brief` | `test_engine.TestLLM` (the SDK client mocked: request shape, schema, model/effort/timeout from settings, every API error mapped to a sentence, truncated and refused replies, no key = no call), `test_the_schema_sent_to_the_model_comes_from_the_model_class`, `test_invented_percentage_in_brief_is_flagged` |

**Plumbing**

| Behaviour | Code | Proven by |
|---|---|---|
| Loose loading: synonyms, export channel names, blanks, duplicates (last wins), negatives, missing days, orphan rows | `load.clean_*`, `normalise_columns`, `normalise_channel`, `build_dataset` | `test_rules.TestLoadEffects`, `TestAwkwardInputs`; `test_engine.TestLoad` |
| New day's data: same-date rows replaced, stock rolled forward, dates sanity-checked, session persisted and restored without double-counting exits | `load.merge_new_history`; `session.*` | `test_rules…test_re_uploading_a_day_replaces_it`, `…messy_two_day_upload…`; `test_engine.test_new_day_rolls_stock_forward`, `test_future_dated_upload_is_refused`, `test_load_session_replays_exits`, `test_restart_after_an_upload_does_not_take_exits_out_twice`, `test_the_cli_data_folder_is_not_overridden_by_a_persisted_session` |
| Boundaries validated: settings, saved overrides, state file, LLM reply | `settings`, `config`, `state`, `llm` | `test_engine.TestBoundaries`, `TestSettings` |
| Applying a parity partner from one card applies its price only | `actions.price_only`; `app.with_parity_partners` | `test_engine.test_a_parity_partner_is_applied_price_only` |

## The mutation audit

A test suite is only as good as the bugs it notices, so an independent reviewer planted 51 realistic
bugs in a copy of the engine, one at a time, and ran the suite after each: a 20% parity band instead
of 5%, a 6-day window, a budget check with AED 500 of slack, the allocator's per-listing cap removed,
raises to on-track listings, candidates 5 AED below the floor, exits inflated by a pair per size, a
zero price gate, swapped budget caps, a 59-day horizon, calibration unbounded or never applied, the
same-day lock removed, and so on.

| | Caught | Missed |
|---|---|---|
| Original suite (55 tests) | 24 | 27 |
| After `test_invariants.py` + `test_rules.py` + the regression tests (117 tests) | 49 | 2 |

The two still "missed" are equivalent mutants, not holes: re-uploaded days are deduplicated
downstream whether or not `merge_new_history` drops the old rows first, and the 1% price-move
tolerance now only gates the elasticity update (the weekly-limit count uses any move of half a
dirham or more, which is tested directly). Two mutations no longer apply because the code they
targeted was replaced (the ratio-based trim rule, the unbounded calibration); the rules that replaced
them have their own tests (`TestAllocator`, `test_calibration_is_bounded`).

The harness and per-mutation logs are not in the repo (they were run from outside it), but every
test added for a missed mutation is named for the bug it kills, so the list above can be read off
`tests/test_rules.py`.

## What passing does *not* prove

- **That the forecasts are right.** The projection is internally consistent and the tool checks
  its own forecasts after the fact, but conversion at a new price is a prior updated from very
  little data. The impact numbers are projections; the learning log is how they are kept honest.
- **That the simulator is reality.** The 4-day loop and the 8-day rehearsal run on
  `tools/simulate_next_day.py`, whose response is drawn from the same kind of model. They prove
  the rules hold over time and that the loop closes, not that the numbers will land.
- **Organic traffic is assumed flat when ads stop.** On Amazon, paid traffic can feed organic rank;
  the first days after a stop are the check, and ads come back after the cooldown if a listing slips.
- **A good share of `test_engine.py` pins specific listings in `data/`** (S19 is broken, S14 Noon is
  starved, S01 Amazon over-buys). On different data those tests would need new fixtures; the
  invariants and rule tests would not.
- **Not built, on purpose** (see README "What I cut"): a marketplace API, an optimiser, price raises,
  size-level pricing, ML forecasting; also demand fade. Each is a line in the defence, not a gap found by accident.

## Running it

```
make verify          # lint, format check, 117 tests, edge cases, drills
make test            # tests only (~75 s)
uv run pytest tests/test_invariants.py -v     # just the brief's rules
```
