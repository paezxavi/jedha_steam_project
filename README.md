# What should Ubisoft's next game be?

A market study of the **55 690 games** listed on Steam, run for Ubisoft's studio leadership.

Jedha *Full Stack Data Scientist* — **Block 2, Big Data Project**. PySpark on Databricks.

The analysis, with every cleaning rule and its measured effect, is in
[`steam_project.ipynb`](steam_project.ipynb).

## The problem

Ubisoft wants to launch a new game and asked for a global reading of the Steam marketplace before
the concept is locked. The brief lists a dozen questions on three levels — the market as a whole,
genres, and platforms. The notebook answers those twelve questions in that order and stops there.
Section 6 gathers what the answers imply for a product brief, separating what is measured from
what is inferred; section 7 lists what the dataset cannot decide at all.

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
  and 68% of the catalogue sits in the bottom one, so its midpoint is `10,000` for two games out
  of three. The notebook never ranks on it directly: it enters only as one factor of
  `owners_x_price`, the list-price value of that midpoint, which section 7 takes apart.

## What we found

### Steam is a long tail with two hundred companies on the end of it

**41% of the catalogue comes from publishers that have released exactly one game.** Big Fish Games
has published the most (423), ahead of 8floor (202) and SEGA (165); Ubisoft is tenth with 128. At
the other end, 195 publishers have released 21 games or more and hold 17% of the catalogue — that
is the field, not the 29 824 names on the store.

Those large publishers do not spread across genres, they repeat a formula — and usually a *family*
of labels rather than one genre, since a game carries several at once. Choice of Games is 99.3%
RPG, 97.1% Indie and 80.0% Adventure **on the same 140 titles**. Six of the eight biggest publish
to a formula; only SEGA and Strategy First spread.

### Covid did not slow releases — the dip is the year before

8 305 games in 2020 and 8 823 in 2021, the two largest years in the dataset, against a dip to
6 968 in 2019 — a plateau that starts in 2018, neither broken nor accelerated by the pandemic.
Before 2014 the same column counts something else entirely: a curated storefront, 61 games in
2006 and still only 471 in 2013, which is why the two jumps in it sit on Steam's own calendar
(Greenlight, then Steam Direct) rather than on the industry's.

Inside the year the calendar is a season: **the six lightest release months are exactly January to
June**, and October ships 4 451 games against January's 3 096.

### The store is cheap, and it prices in `.99`

**78.6% of the catalogue is free or under $10**, 42.2% under $5 alone, and **95.6% of paid games
end on those two digits**. The paid median is $5.99 against a mean of $8.99 — a thin tail
stretching to a $999 outlier. Discounts are rare on any given day, 2 518 games or 4.5%, and they
are a tactic of the cheap end: **1 884 of those 2 518 are games under $5**, and both the rate and
the depth fall with every band.

### Localisation is the exception, not the rule

English is on **99%** of the catalogue and is not a decision. Below it comes a block of seven —
German 25.2%, French 24.1%, Russian 23.2%, Simplified Chinese 23.0%, Spanish 22.0%, Japanese
18.6%, Italian 16.7% — and then the list steps down to 12.1%. But **53.3% of games ship in a
single language**, and only one game in ten carries more than nine.

### The age field is abandoned, so the question has three answers

`required_age` is 0 for 98.8% of the catalogue: Steam gates mature content on developer-declared
content descriptors, not on this legacy field. Read literally, **301 games are 16+ or over**. Read
through the community tags mapped onto Steam's own five descriptors, **807 games carry the two
that force an 18+ affirmation** and **16 295 — 29.3% — carry any mature-content tag**. The field
is not noise, it is unused: 88% of the games that *do* declare 16+ carry a mature tag.

### The best-paying genre is the worst-liked one

**Massively Multiplayer leads on value by a wide margin — 10.93 M$ of stock value a paid game,
three times RPG's 3.67 and Action's 3.04** — while Casual, 40% of the shelf, closes the nine real
genres at 0.43. And MMO is last of the nine on satisfaction, **0.648 against 0.748 to 0.803 for
the other eight**, ten points below the lowest of them.

Its cheap published price was an artefact: half of MMO is free, and a list price of zero times any
number of owners is zero. On its paid half it charges **$11.08, the most of any real genre**.

### Ubisoft is already in the right genres, and behind on quality

Its two largest genres — **Action with 75 titles, Adventure with 49** — are third and second of
the nine on satisfaction, and it has only 4 games in MMO. But held at equal review volume its
catalogue sits **four to seven points below the market in the three bands where it has a real
sample**: above 10 000 reviews the market's median positive ratio is **89.4%** and Ubisoft's own
39 titles in that band are at **83.0%**. It is ahead only in the 10-99 band, on seven titles.

### Windows is not a decision, and Ubisoft ports less than anyone

**55 675 of the 55 690 games support Windows**, Mac reaches 22.9% and Linux 15.2%, and the largest
group on the store is Windows-only: **41 271 games, three quarters of it**. Genre barely moves
that — 18.5% to 27.6% on Mac across the nine real genres, a 1.5 ratio. Ubisoft is the outlier in
the other direction: **4.4% Mac, 1.5% Linux, and 94.8% of its games Windows-only.**

## What this data cannot decide

It stops on 11 November 2022, and the Steam Deck is only eight months old in it. There are no
sales, only SteamSpy's brackets: `owners_x_price` counts what a customer would pay, not what a
publisher receives, and it values every free game at zero. Reviews stand in for reach throughout,
and how tightly the two track is not measured. `publisher` is free text, so the rankings rank
names rather than firms. Delisted games are absent, so every share measured is optimistic. And
nothing here is causal — a genre is chosen by studios that already expect a game to sell.

## Running it on Databricks

The notebook is written for Databricks and imports either as `.ipynb` or as
[`steam_project_databricks.py`](steam_project_databricks.py) (Databricks source format).

1. **Workspace → Import** the notebook, attach it to serverless compute.
2. Run it. The first cell creates the Unity Catalog volume `workspace.default.steam` and
   downloads the 61 MB JSON into it — no manual upload. If outbound network is blocked, upload
   the file to that volume by hand and re-run.
3. Every table comes out of a `display()` call, so Databricks' visualisation tool can chart it
   in place. Each one is preceded by a `*Chart: …*` line saying which chart to build.

Serverless compute runs Spark 4.2 and has no `cache()`. The two frames the notebook re-reads
most — the cleaned `games` and the exploded `genre_rows` — go through `materialise()` instead,
which writes them as the Delta tables `workspace.default.steam_games` and
`workspace.default.steam_genre_rows`. That matters here: the source JSON is a single line, so it
is non-splittable and every uncached action re-parses all 61 MB in one task. Locally the same
function just calls `cache()`.

### Screenshots for the jury

Databricks removed the **Publish** button, so the deliverable is the notebook plus an export that
carries its rendered outputs: [`steam_project_databricks.html`](steam_project_databricks.html).
The HTML export is the only one that embeds the visualisations — Databricks attaches them to the
command result, not to the code, so a `.py` or `.ipynb` export carries the tables and loses the
charts. Seventeen charts are built in it, one per `*Chart: …*` line in the notebook.

Screenshots, if they are wanted as well, go in `images/`. One per numbered section covers the
whole argument:

| # | section | output |
|---|---|---|
| 1 | 1.1 / 1.2 | the 441 columns inference invents for `tags`, then the explicit schema |
| 2 | 2.10 | the cleaning report |
| 3 | 3.1 | top publishers by volume, then the concentration table |
| 4 | 3.2 | releases per year, then releases per month |
| 5 | 3.3 | the price spread and the bands, then discounts by band |
| 6 | 3.4 | the language ranking, then the language bands |
| 7 | 3.5 | the `age_rating` distribution, then the two tag readings by age group |
| 8 | 3.6 | the Wilson ranking, then Ubisoft against the market by review band |
| 9 | 4.1 | the genre shelf |
| 10 | 4.2 | genres by satisfaction, then Ubisoft's own genres |
| 11 | 4.3 | the eight largest publishers and their genre families |
| 12 | 4.4 | value by genre, published and paid |
| 13 | 4.5 | the genre mix, 2017 against 2022 |
| 14 | 5.1 | what the store runs on, then what Ubisoft runs on |
| 15 | 5.2 | porting rates by genre |

## Running it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -o data/steam_game_output.json \
  https://full-stack-bigdata-datasets.s3.amazonaws.com/Big_Data/Project_Steam/steam_game_output.json
```

Then open `steam_project.ipynb` with the `.venv` kernel. The first cell detects that it is not on
Databricks, builds a local Spark session and defines the `display()` and `materialise()` that
Databricks provides itself — every other cell is identical in both environments. A **JDK 17**
must be reachable through `JAVA_HOME`. Local Spark is 3.5.3 against 4.2 on serverless; the
DataFrame API this notebook uses is unchanged between them.

## Layout

```
steam_project.ipynb              the deliverable: ingestion, cleaning, the three levels, the brief
steam_project_databricks.py      the same notebook in Databricks source format, for import
steam_project_databricks.html    the Databricks export, the one that carries the 17 charts
images/                          screenshots of the Databricks cell outputs
build/                           source of the notebook and the scripts that build and run it
data/                            the dataset (not committed — 61 MB, fetched by the commands above)
requirements.txt                 pinned dependencies for a local run
```
