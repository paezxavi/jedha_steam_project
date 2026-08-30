# What should Ubisoft's next game be?

A market study of the **55 691 games** listed on Steam, run for Ubisoft's studio leadership.

Jedha *Full Stack Data Scientist* — **Block 2, Big Data Project**. PySpark on Databricks.

The analysis, with every cleaning rule and its measured effect, is in
[`steam_project.ipynb`](steam_project.ipynb).

## The problem

Ubisoft wants to launch a new game and asked for a global reading of the Steam marketplace before
the concept is locked. The brief lists a dozen questions on three levels — the market as a whole,
genres, and platforms. The notebook answers them in that order, and each level closes on the one
thing it changes in the product brief: **genre, price, platforms, languages, release window, age
rating**.

## The dataset

`steam_game_output.json` — a 61 MB JSON array, one object per game, `{"id": ..., "data": {...}}`,
from `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/`. It is a **SteamSpy snapshot** and
the newest release in it is **11 November 2022**.

Three properties drive the whole pipeline:

- **The array sits on one line**, so `multiLine` is mandatory — without it Spark reads a single
  unparseable record.
- **`tags` is an object keyed by tag name.** Left to schema inference it becomes a struct with
  441 columns, one per tag. Declared as `MAP<STRING, BIGINT>` it becomes one column that
  `explode()` opens into (tag, votes) rows.
- **Nothing in the file is a sales figure.** `owners` is a bracket (`"10,000,000 .. 20,000,000"`)
  and 68% of the catalogue sits in the bottom one, so the notebook measures success with the
  **median review count** and the **breakout rate** — the share of games above 100 000 owners.
  On the log scale, reviews and owners correlate at **0.76**.

## What we found

### Steam is a long tail with thirty companies on the end of it

**41% of the catalogue comes from publishers that have released exactly one game**, and the twenty
largest account for 5% of releases. Big Fish Games has published the most (423). Rank the same
publishers by revenue instead of volume and the list changes completely — **Ubisoft comes first**,
ahead of EA and Valve. The competitor set is that table, not the 29 824 publishers on the store.

### Covid did not slow releases — the dip is the year before

8 305 games in 2020 and 8 823 in 2021, the two largest years in the dataset, against a dip to
6 968 in 2019. But **November and December are the worst months to launch in**: 19 median reviews
and a 7.5% breakout rate, against 26-27 and 9-10% in February to May. The holiday window belongs
to the titles that can buy visibility in it.

### The crowded genres are the poor ones

Setting each genre's share of proxy revenue against its share of releases, three return more than
they take — **Massively Multiplayer 2.96, RPG 1.81, Action 1.52** — and two are badly crowded:
**Casual takes 40% of the shelf for 8.7% of the value, Indie 71% for 35%**.

MMO buys the best economics and the worst product: its median positive ratio is **0.665**, fifteen
points below every other genre. RPG is second on economics with no such penalty.

### Porting is the cheapest lever, and Windows is not a decision

**55 675 of 55 690 games support Windows.** Mac reaches 22.9% and Linux 15.2%, and games that
ship on more than one platform lead on both success measures inside every price band — 494 median
reviews against 238 at $20-40. Porting rates barely move with publisher size (24.0% / 27.8% /
26.6%), which removes the obvious confounder without making the relationship causal.

### The brief that comes out of it

| | decision |
|---|---|
| **Genre** | Action-RPG, single-player, premium — not Casual, not live service |
| **Price** | $29.99 - $39.99 |
| **Platforms** | Windows at launch, Mac and Linux planned in |
| **Languages** | twelve: EN, DE, FR, RU, zh-Hans, ES, JA, IT, KO, pt-BR, PL, zh-Hant |
| **Window** | February to May |
| **Rating** | mature is not a commercial handicap |

Quality bar: **90% positive is a good Ubisoft-scale result on Steam, 95% at scale is exceptional**
— Portal 2 sits at 98.8% over 309 441 reviews.

## What this data cannot decide

It stops on 11 November 2022. There are no sales, only SteamSpy's brackets, and the revenue proxy
(`owners x list price`) ignores Steam's cut, regional pricing, discounts, refunds and bundles —
and values every free-to-play game at zero. Delisted games are absent, so the breakout rates are
optimistic. And nothing here is causal: price, ports and localisation are all things a studio
chooses *because* it already expects the game to sell.

## Running it on Databricks

The notebook is written for Databricks and imports either as `.ipynb` or as
[`steam_project_databricks.py`](steam_project_databricks.py) (Databricks source format).

1. **Workspace → Import** the notebook, attach it to serverless compute.
2. Run it. The first cell creates the Unity Catalog volume `workspace.default.steam` and
   downloads the 61 MB JSON into it — no manual upload. If outbound network is blocked, upload
   the file to that volume by hand and re-run.
3. Every table comes out of a `display()` call, so Databricks' visualisation tool can chart it
   in place. Each one is preceded by a `*Chart: …*` line saying which chart to build.

### Screenshots for the jury

Databricks removed the **Publish** button, so the deliverable is the notebook plus screenshots of
its cell outputs, taken in the workspace and dropped in `images/`. Ten are enough to carry the
whole argument:

| # | section | output |
|---|---|---|
| 1 | 1.1 / 1.2 | the 441 columns inference invents for `tags`, then the explicit schema |
| 2 | 2.10 | the cleaning report |
| 3 | 3.1 | top publishers by volume, then by revenue proxy |
| 4 | 3.2 | releases per year, and the launch month table |
| 5 | 3.3 | the price band table |
| 6 | 3.4 | the localisation band table |
| 7 | 3.6 | the Wilson ranking |
| 8 | 4.4 | the opportunity scatter |
| 9 | 4.6 | genre x breakout rate in the $20-40 band |
| 10 | 5.3 | porting, controlled by price band and by publisher size |

## Running it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -o data/steam_game_output.json \
  https://full-stack-bigdata-datasets.s3.amazonaws.com/Big_Data/Project_Steam/steam_game_output.json
```

Then open `steam_project.ipynb` with the `.venv` kernel. The first cell detects that it is not on
Databricks, builds a local Spark session and defines a `display()` that renders Spark DataFrames
as tables — every other cell is identical in both environments. A **JDK 17** must be reachable
through `JAVA_HOME`.

## Layout

```
steam_project.ipynb              the deliverable: ingestion, cleaning, the three levels, the brief
steam_project_databricks.py      the same notebook in Databricks source format, for import
images/                          screenshots of the Databricks cell outputs
build/                           source of the notebook and the scripts that build and run it
data/                            the dataset (not committed — 61 MB, fetched by the commands above)
requirements.txt                 pinned dependencies for a local run
```
