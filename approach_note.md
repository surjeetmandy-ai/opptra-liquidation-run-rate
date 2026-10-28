# Approach note — The Liquidation Run-Rate Problem

**Surjeet Singh · Opptra AI Product Engineer (Senior) · Stage 1**

## 1. The problem, and what success looks like

Farah has about 3,000 pairs of last season's sneakers on Amazon.ae and Noon. In 60 days, whatever is left goes to a jobber for a quarter of what it cost. Every morning she asks: more ads, or a lower price?

The question behind that is simpler. For each listing, each day: **is selling one more pair on the marketplace still worth more than handing it to the jobber, and if so, what is the cheapest way to sell it?**

So the goal is **to recover the most money over the 60 days**: sales, minus ad spend, minus storage, plus what the jobber pays for whatever is left or sent early. The no-effort alternative is to jobber everything today (about AED 83,000 on this stock). Success is beating that by as much as the stock allows. Sell-through will be on the page but is not the target: you can empty a warehouse by spending more on ads than a pair is worth, and that is not a win.

"Will we make it?" is a run-rate question. Each listing needs to sell (pairs left ÷ days left) a day. The data says the portfolio needs about 51 pairs a day and sells about 40, with nearly all the ad budget already spent. That gap, and where the ad money is going, is what the tool is for.

Three positions from the start:

- **The jobber is an option every day, not just on day 60.** A pair that won't sell by the deadline costs storage and ads every day it stays. Some pairs should go early: the odd sizes left once the popular ones are gone, or a listing where every sale costs more in ads than it earns.
- **Doing nothing is the default.** A cut you didn't need gives away margin on pairs that would have sold anyway, and there are only two price changes per listing per week. A change has to prove it's worth it.
- **With 14 days of history the tool will be wrong at first**, so it has to be safe when wrong, not just right on average.

## 2. How it decides, every morning, for every listing

Four steps, all plain arithmetic.

1. **How far behind?** Pairs needed per day versus what sold over the last week: on track, behind, or critical.
2. **Why?** Few people see it but a good share of them buy: it needs visibility, so ads. Plenty see it and don't buy: the price is wrong. Both poor: fix the price first; paying for traffic to a listing that doesn't convert is the expensive mistake. No lever closes the gap profitably: send those pairs to the jobber now.
3. **How much?** Ad money goes, per marketplace, to the listings where each dirham brings the most sales, up to the daily budget. Price cuts are the smallest step that gets the listing on track, never straight to the floor, with time to read the response before cutting again. Prices are decided for both marketplaces together, because Noon must stay within 5% of Amazon.ae: a cut on one often forces a cut on the other, and that uses one of Noon's two weekly changes.
4. **Check the rules.** Every recommendation passes one check against all the business rules (floor, 5% parity, daily ad budgets, two changes a week) before Farah sees it. If a rule blocks the right move, the tool says so and offers the best legal one.

Farah sees a short list of actions, each with an exact number and what it's worth ("cut S15 on Amazon.ae 289 → 234, Noon to match; about +AED 2,000"), and one verdict: what sells by the deadline and what we recover if nothing changes, with today's actions, and with the jobber today. Each action has an Apply button. Applying updates the tool's memory, so tomorrow's rules are checked against what was actually done.

## 3. Where the AI is, and why there

Not in the numbers. The rules are hard limits, and a language model can't be trusted to count price changes in a rolling seven-day window correctly every morning. Prices, budgets and exit quantities come from code I can test.

The AI does what I'd otherwise do by hand each morning: it reads the finished plan and writes Farah's brief, verdict first, in plain words. And it points out what arithmetic can't see: two days of zero Noon traffic that look like a reporting gap, not a demand collapse; a cut that will hit the parity rule next week; a style whose remaining sales come from sizes that are nearly gone. It flags; it doesn't decide. One call a day; if it fails, the page still shows the plan.

## 4. How it gets better over the 60 days

Three simple things, because 14 days of data doesn't justify more.

- Each listing's conversion rate is updated daily, but a listing with little data leans on the average for its group (men's on Amazon, kids' on Noon) until it has enough history of its own.
- How much a cut helps is learned from the tool's own cuts, because the history holds exactly one price change (S08 on Noon, a parity move). The two-changes-a-week limit is also the experiment budget: another reason not to waste a change by going straight to the floor.
- Every applied action records what it expected. When the next day's data arrives the tool compares, trusts a lever less if it keeps under-delivering, and widens or narrows the range on the verdict.

Early on it leans on ads, which can be switched off tomorrow. Later, once it has measured its own cuts, it leans on price and exits, which can't be undone.

## 5. What I'm leaving out

No marketplace integration, login, database or backend: one page, local state, three commands to run. No optimiser; a simple greedy allocation across 40 listings is explainable and close enough. No pricing by size; sizes only inform exits. No competitor tracking, no forecasting model on 14 data points. Spare time goes into tests for the rule check. Every assumption is written down in the repo.
