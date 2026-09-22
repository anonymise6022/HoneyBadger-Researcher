# research_cli

A terminal research tool for people who are new to investing.

```bash
python -m research_cli "why did SPY fall today"       # what coincided with a move
python -m research_cli "is AAPL good to invest"       # fundamentals and peer context
python -m research_cli backtest SPY -s ma-crossover   # does a rule beat holding?
```

**It never tells you a cause, and it never tells you what to buy.** Those are
not gaps to be filled in later; they are the design, and they are enforced by
code rather than by good intentions.

## Quick start

Run everything from the repository root.

```bash
pip install -r research_cli/requirements.txt
python -m research_cli --help
python -m research_cli "why did SPY fall today"
```

If the repository has a virtual environment, either activate it first
(`source .venv/bin/activate`) or call its interpreter directly
(`.venv/bin/python -m research_cli ...`). To save typing:

```bash
alias research='python -m research_cli'
```

No API keys needed. Prices and fundamentals come from Yahoo Finance; if that
is unreachable the tool falls back to clearly labelled synthetic data rather
than failing. Add `--mock` to work entirely offline.

```bash
# Why did it move?
python -m research_cli "why did TSLA drop yesterday"
python -m research_cli "why did NVDA jump on 2026-09-15"
python -m research_cli "why did the stock market fall this week"
python -m research_cli "why did EUR/USD rally today"
python -m research_cli "why did aapl fall today"            # lowercase is fine

# Is it worth owning?
python -m research_cli "is AAPL good to invest"
python -m research_cli "should i buy TSLA"
python -m research_cli "tell me about KO"

# Does a trading rule actually work?
python -m research_cli backtest SPY --strategy ma-crossover
python -m research_cli backtest AAPL --strategy rsi --years 8
python -m research_cli backtest QQQ --strategy buy-and-hold --no-walk-forward

# Options
--plain          no colour or tables
--mock           offline, synthetic data
--live           fail loudly instead of falling back to synthetic
--json out.json  also write the evidence bundle
--llm            write the note with Claude in plainer language
--compare        show the template and Claude versions side by side
--help
```

Optional credentials, both free, neither required:

```bash
export FRED_API_KEY=...        # real macro data (fred.stlouisfed.org)
export ANTHROPIC_API_KEY=...   # --llm prose
```

## The app

A real macOS application at `/Applications/Research.app` — double-click it.
No terminal window, its own icon in the Dock.

```bash
python -m research_cli app             # same window, from a shell
python -m research_cli app --terminal  # the older in-terminal interface
```

Built with pywebview over the system WebKit view, so the bundle carries no
browser engine and text is rendered by the same engine as Safari. The window
has four tabs (Ask, Explore, Lab, Settings), six colour themes, four
typefaces — including an 8-bit terminal face carried inside the app — and
three explanation depths. Theme, typeface and depth are remembered between
launches.

Hovering a candle on the Explore chart reads that session's open, high, low
and close back exactly, rather than leaving them to be estimated against the
axis.

Rebuild it after changing anything:

```bash
pyinstaller research_cli/desktop/Research.spec --noconfirm \
  --distpath dist --workpath build
cp -R dist/Research.app /Applications/
```

### Explanation depth

The same evidence, pitched three ways — selectable in the app's title bar or
with `--depth` on the command line:

| Depth | What changes |
|---|---|
| `beginner` | Every term glossed in passing. "The base rate (how often this happens on any random day)…" |
| `intermediate` | Assumes P/E, volatility and base rates need no introduction |
| `analyst` | Terse and quantitative. Adds `z`, standard errors and lift inline |

Depth is a presentation parameter and never a data one. All three render from
the identical `EvidenceBundle` and pass the identical validator, so two
readers at different depths cannot come away with different figures.

### Plain-language reports, written locally

`--llm` rewrites the template report into plain prose. Two models can do it,
and the tool prefers whichever needs least from you:

| Backend | When it is used |
|---|---|
| **Local** — `mlx-community/Qwen3.5-4B-OptiQ-4bit` | Whenever the weights are on disk. No key, no network, nothing leaves the machine |
| **Claude** | When no local model is downloaded and `ANTHROPIC_API_KEY` is set |
| Neither | The template is shown, which is the normal no-configuration path |

The local model is Qwen3.5 4B quantised to four bits for MLX: 4GB on disk,
Apache 2.0, roughly 25 tokens a second on an M4, so a report takes about
forty seconds against an instant template. Install and fetch it with:

```bash
pip install mlx-lm
python -c "from research_cli.synthesis import local_llm; local_llm.ensure_weights()"
```

**Why a 4B model is enough here.** It is not asked to know anything. The
evidence bundle is computed before it is called, handed over pre-sorted, and
`validator.validate_report` afterwards checks that every figure in the prose
traces back to a bundle field — rejecting the draft and regenerating once
with the failures quoted back if it does not. Unvalidated prose is never
shown, whoever wrote it. The guardrail is mechanical, so the model only has
to rewrite faithfully, and a small one does that. Measured on the synthetic
bundles: 52 of 52 figures traced on the first attempt.

**What the small model needed that Claude did not.** One appended block,
`FINAL_CHECKLIST`, restating the banned words at the very end of the prompt.
A 4B model holds the start of a long prompt less firmly than the end, and
the failure this fixes was exactly that — a draft that traced all 86 of its
figures correctly and still wrote "Because the sample size is relatively
small", having forgotten a rule stated two thousand words earlier. Claude's
prompt is unchanged.

### Options (paper trading)

A trading screen rather than a research one: the live chain for a symbol,
with calls, strikes and puts laid out the way a broker shows them, and a
paper account that can buy and sell the contracts in it.

Quotes come from Yahoo through `yfinance` — real strikes, bid, ask, last,
implied volatility, volume and open interest, delayed by roughly a quarter
of an hour. With no network, or for a symbol with no listed options, a chain
is modelled from the underlying's own synthetic history and labelled as
such everywhere it appears.

The account is deliberately unflattering, because a paper account that
shaves the corners teaches the opposite of the lesson:

| Rule | Why |
|---|---|
| Fills cross the spread — buy the ask, sell the bid | Filling at the midpoint hands the trader the half-spread on every trade, which is most of the profit on a small winner |
| $0.65 a contract, each way | Two dollars a round trip is a fifth of the gain on a ten-cent move |
| Shorts are collateralised before they are allowed | A short put ties up the cash to buy the shares at the strike; a short call ties up Regulation T margin. Being unable to open one is the position's risk, stated in advance |
| Positions mark at what it would cost to get out | The bid for a long, the ask for a short. Equity is what the account could be liquidated for, not what it looks like |
| Expiry settles whether or not anyone is watching | Against the underlying's close on the contract's own expiry day, with assignment on anything short and in the money |

Selecting a contract draws all five greeks against the underlying's price,
plus its value now against its value at expiry, and puts the last six months
of the contract's price beside the stock's, both indexed to 100 — which is
where the leverage in an option becomes visible rather than merely stated.
That last chart is **modelled**: there is no free source of historical option
quotes, so each past day is priced from that day's close at today's implied
volatility.

Two things a real account has that this one does not, both said on screen:
early assignment, which a short American option can suffer at any moment and
which is modelled only at expiry; and shares, so an exercised call settles in
cash rather than delivering stock.

The account lives in `~/.config/research_cli/paper_options.json` and can be
reset from the Activity panel. Greeks are quoted the way a platform quotes
them — delta per share, theta per day, vega per point of implied volatility —
which is arithmetic, not convention, and is covered by tests against
published values.

Lab → Option overlays is a different thing: a historical backtest of an
option strategy rolled over many years, which is what the years selector
there is for.

### Quant (alpha)

GARCH volatility forecasting, Hurst exponent and variance-ratio regime
detection, and Ornstein-Uhlenbeck mean-reversion fits. Marked experimental
throughout, and deliberately kept out of the attribution and snapshot
reports.

```bash
research quant SPY
```

The models are tested against processes whose answers are known — a simulated
random walk must score a Hurst of 0.5 and a variance ratio of 1, and does.
That is a much weaker claim than being useful: run on real daily equity data
these diagnostics mostly report "random walk", which is the honest answer.

Where they genuinely earn their place is the backtest, where a volatility
forecast or a regime filter can be tested walk-forward against buy-and-hold:

| Strategy | SPY, 12 years, out-of-sample |
|---|---|
| Buy and hold | +223%, Sharpe 0.78, −34% drawdown |
| **Volatility targeted** | **+140%, Sharpe 0.81, −20% drawdown** |
| Momentum (12-1) | +141%, Sharpe 0.77 |
| MA crossover | +133%, Sharpe 0.75 |
| Mean reversion | +13%, Sharpe 0.16 |

Volatility targeting is the only rule to beat buy-and-hold on risk-adjusted
return, and it does so by giving up raw return — which is exactly what the
literature says it does. Mean reversion fails badly, which is consistent with
the regime module calling SPY a random walk.

**Credentials and GUI launches.** An app started from Finder does not run your
login shell, so `export FRED_API_KEY=...` in `.zshrc` is invisible to it. Keys
are therefore stored in `~/.config/research_cli/keys.env` (mode 0600), which
every launch path reads:

```bash
research keys                     # what is set, and what each key unlocks
research keys FRED_API_KEY        # prompts without echoing
research keys FRED_API_KEY --remove
```

Environment variables still win over the file, so existing setups keep working.

## What the reports actually say

### "Why did X move?"

Three ideas most market commentary skips.

**Was the move even unusual?** A 0.4% drop in a market that typically moves
0.9% a day is noise. The report says so before listing anything else, because
presenting a list of explanations for a non-event is the most misleading thing
a tool like this can do.

**How often has this factor coincided with moves like this?** Each candidate
— the VIX, the 10-year yield, credit, the dollar, sector breadth — gets a
*hit rate*: of the past days when that factor was in the state it is in now,
what share saw the symbol move this way? Four years of history, conditioned on
quintiles, so each rate rests on roughly 200 days.

**Is that better than guessing?** Always shown beside the *base rate*. A 62%
hit rate against a 60% base rate is worth nothing, and the report says
"within the margin of error" rather than ranking it as evidence.

Factors that are *part of* the symbol — sector breadth for an index, a
mega-cap constituent — are shown separately under "what the move was made of".
Their 88% hit rates are arithmetic, not findings.

### "Is X good to invest?"

Fundamentals against a sector peer group, price and volatility over the past
year, recent headlines, and a closing section that names the specific
unresolved tension in *these* figures rather than a generic disclaimer.

### "backtest"

Three textbook rules (moving-average crossover, RSI threshold, buy-and-hold)
run walk-forward: parameters are chosen on each training window and scored on
the period that follows. Output is a console table beside the buy-and-hold
benchmark, plus an equity-and-drawdown chart in `research_cli/results/`.

## Architecture

```
query_parser.py          natural language -> structured request (regex, never an LLM)
data/
  price_ingest.py        OHLCV + the volatility baseline that judges a move
  macro_ingest.py        FRED series, publication lags, release calendar
  fundamentals.py        company financials, unit-normalized at the boundary
  mock_data.py           deterministic synthetic fallback for every source
evidence/
  schema.py              EvidenceBundle -- the only thing a report may reference
  why_moved.py           observation + factors + hit rates + counterevidence
  snapshot.py            fundamentals + peers + trend + news
synthesis/
  template_report.py     bundle -> plain text, no model
  snapshot_report.py     the same, for company snapshots
  llm_report.py          the same, written by Claude, then verified
  validator.py           every number in the prose must trace to a bundle field
backtest/
  engine.py              event-driven loop; lookahead structurally impossible
  strategies.py          the three built-in rules
  walk_forward.py        fit on the past, measure on the future
  risk_metrics.py        Sharpe, drawdown, win rate
  plots.py               equity + drawdown chart
display/formatting.py    rich terminal rendering
```

Two load-bearing constraints:

**Synthesis receives a bundle and nothing else**, and the validator afterwards
checks that every figure in the prose traces back to a bundle field. A number
not in a bundle cannot appear in a report. When Claude writes the note, a
draft that fails is regenerated with the failures quoted back; if the second
attempt also fails, the plain template is shown instead. Unvalidated prose is
never displayed.

**A strategy can never see the future.** It is not handed the price history —
it is handed a `SealedWindow` whose frame has already been truncated at the
current bar. Tomorrow's price is not hidden behind a flag, it is absent from
the object. Orders fill at the *next* bar's open, and walk-forward gives
parameter selection and scoring physically separate slices, with a runtime
check on every fold.

## Running the tests

```bash
pytest research_cli/tests/ -q      # 370 tests, no network, no API keys
```

Synthetic data is seeded from a hash of the symbol, so `mock_price_history("AAPL")`
returns identical data on every machine and tests assert real values.

## Design decisions

Choices that were genuinely ambiguous, and what was decided.

**No economic-calendar API.** There is no free, terms-of-service-clean source
for a forward-looking economic calendar with consensus forecasts. `investpy`,
the package usually reached for, has been blocked by its upstream since 2022
and would pin an old pandas against this repository's pandas 3.0. Release
dates are therefore derived from each FRED series' typical publication lag and
can be off by a few days. Every report carrying a macro release says so.

**"Surprise" means change, not surprise.** Consensus forecasts are licensed.
Surprises are measured against the trailing *average change* — a drift-aware
random walk — not against what forecasters expected. An early version divided
the raw change by its standard deviation, which scored every ordinary CPI
print as a three-sigma shock, because a price *index* rises nearly every month
and the expected change is not zero.

**Hit rates use quintiles, not thresholds.** A rule like "VIX up more than 5%"
is a free parameter that can be tuned until the result looks impressive.
Quintiles have none, and always retain about a fifth of the history.

**Mechanical factors are separated from evidence.** Sector breadth "predicts"
an S&P move with an 88% hit rate because the S&P *is* those sectors. Such
factors are excluded from the evidence ranking however high they score.

**Factors are ranked by hit rate × confidence**, as specified — never by
narrative appeal. Because that formula can rank a high-hit-rate factor above a
more informative one, `lift` and a two-standard-error test are shown beside
every factor.

**Unknown symbols are refused, not simulated.** The synthetic fallback exists
so the tool still works when Yahoo is down. It must not cover a symbol that
does not exist: a user typing "AAPI" once received a complete, confident
report about an invented company, with only a warning line to say so. An empty
vendor response is now a hard error.

**Units are normalized at the data boundary.** yfinance's `.info` mixes
conventions within one payload — `dividendYield` is a percent (KO returns 2.4),
`profitMargins` is a fraction (0.276). Formatting the first as a fraction
prints "240%". Everything is converted to fractions once, at the edge, and
implausible values are flagged rather than clamped.

**Quoted text is excluded from the language checks.** A report that echoes
"is AAPL a buy" is repeating the user's words, and a headline reading "Wells
Fargo raises price target" is a fact about a headline. Neither is the tool
making a recommendation, so both are removed before the verdict scan.

**Two CLI commands, but bare questions still work.** Typer requires an explicit
subcommand once an app has more than one. `main()` inserts `ask` when the first
argument is neither a command nor a flag, so `research "why did SPY fall"`
keeps working.

**Credentials live in a file, not only the environment.** A `.app` launched
from Finder inherits no shell environment on macOS, and Windows has the same
split. A user who sets a key, confirms it in Terminal, then double-clicks the
app and sees synthetic data again has no way to guess why. The file is the
fix; the environment still takes priority.

**The app is Textual, not native widgets.** The terminal aesthetic is the
point — dense, monospaced, information-first — and Textual keeps it while
adding an input box, tabs, scrolling and a progress indicator. It is also pure
Python, so there is no Node or Rust toolchain in the build.

**Rescaled range takes returns, not prices.** R/S analysis cumulates the
series it is given, so passing a price level integrates twice and returns
roughly H + 1 — a random walk's prices score 1.01 instead of its returns'
0.50, and every series reads as strongly trending. The parameter is named
`increments` for that reason.

**Hurst needs the Anis-Lloyd correction.** Uncorrected, rescaled range is
biased upward on finite samples: independent noise scores about 0.56 with a
regression error near 0.01, which reads as a five-sigma trend. The null
expectation is subtracted, and the classification band is widened to twice
the regression error (floored at 0.05) because that error is itself
optimistic — the log-log points are correlated across lags.

**Mean reversion is decided by a Dickey-Fuller test, not a t-statistic.**
Near a unit root the OLS estimate of the AR(1) coefficient is biased
downward *and* its t-statistic does not follow a normal distribution — the 5%
critical value is near −2.86, not −1.96. An earlier version used the naive
bar and reported every simulated random walk as mean-reverting with a
140-day half-life. With a proper ADF test the false-positive rate drops to
the test's own 5% size.

**The app is Textual no longer.** The terminal-in-a-window version is kept
behind `--terminal`, but the default is a native WebKit window: real
typography, CSS themes, SVG charts and subtle motion, none of which a
character grid can do. The brief was "terminal-like, not 8-bit", and the
distinction is exactly that.

**Ambiguous queries resolve, then announce.** "Why did the stock market fall"
has no ticker, so SPY is substituted — and the report says so. Lowercase words
that are not ordinary English are read as tickers, also announced.

**Modules beyond the specified layout.** `data/fundamentals.py`,
`backtest/strategies.py` and `backtest/plots.py` were added; folding them into
their neighbours would have made those files do two jobs.

## Limitations

- Hit rates measure **coincidence**, not causation, and past coincidence need
  not continue.
- Index weights used for single-stock contribution are static and approximate.
- FRED data is the current vintage, not what was known at the time.
- FRED serves the ICE high-yield spread (`BAMLH0A0HYM2`) as a rolling ~3-year
  window under licence, so that one series has much less history than the rest.
  Rows before the window start are left empty rather than filled.
- Peer groups are a fixed list of large names per sector, not a screen matched
  on size or business mix.
- Headlines are titles only, from a feed that is loosely tied to the ticker.
- The backtest models commission and slippage but not borrow costs, financing,
  taxes, dividends, or market impact.
- One ticker per query.

## Not investment advice

This tool summarizes evidence. It does not identify causes, does not value
securities, and does not recommend transactions. Whether anything here is
relevant to you depends on your circumstances, which it knows nothing about.
