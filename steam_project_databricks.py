# Databricks notebook source
# MAGIC %md
# MAGIC # What should Ubisoft's next game be?
# MAGIC
# MAGIC A market study of the **55 691 games** listed on Steam, run for Ubisoft's studio leadership.
# MAGIC
# MAGIC Jedha *Full Stack Data Scientist* — **Block 2, Big Data Project**. PySpark on Databricks.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Ubisoft wants to launch a new game and asked for a global reading of the Steam marketplace
# MAGIC before the concept is locked. The brief lists a dozen questions on three levels — the market as
# MAGIC a whole, genres, and platforms. This notebook answers them in that order, and each level closes
# MAGIC on the one thing it changes in the product brief: **genre, price, platforms, languages, release
# MAGIC window, age rating.**
# MAGIC
# MAGIC Section 6 collects those six decisions. Section 7 says what this dataset cannot decide.
# MAGIC
# MAGIC ## The data
# MAGIC
# MAGIC `steam_game_output.json` — a 61 MB JSON array, one object per game, `{"id": ..., "data": {...}}`,
# MAGIC served from `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/`. It is a **SteamSpy
# MAGIC snapshot** and the newest release in it is **11 November 2022**, so every count for 2022 is
# MAGIC partial and no game released after that date exists here.
# MAGIC
# MAGIC Three of the 22 fields are nested — `tags` (a tag → number-of-votes object), `platforms`
# MAGIC (three booleans) and `categories` (a list) — which is what makes the file semi-structured and
# MAGIC what `explode()` and `getField()` are for.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Environment
# MAGIC
# MAGIC The notebook is written for **Databricks**, and falls back to a local Spark session so the code
# MAGIC can be re-run outside a workspace. Everything below this cell is identical in both cases,
# MAGIC including the `display()` calls that drive Databricks' visualisation tool.

# COMMAND ----------

import os

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
SOURCE_URL = ("https://full-stack-bigdata-datasets.s3.amazonaws.com"
              "/Big_Data/Project_Steam/steam_game_output.json")

if IS_DATABRICKS:
    # Unity Catalog volume: the file is fetched once, then read from storage.
    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.steam")
    DATA_PATH = "/Volumes/workspace/default/steam/steam_game_output.json"
    if not os.path.exists(DATA_PATH):
        import urllib.request
        urllib.request.urlretrieve(SOURCE_URL, DATA_PATH)
else:
    # Local run: build the session by hand and emulate Databricks' display().
    from pyspark.sql import SparkSession, DataFrame
    from IPython.display import display as _ipython_display

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (SparkSession.builder.appName("steam")
             .master("local[*]")
             .config("spark.driver.memory", "10g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    def display(x, n=1000):
        _ipython_display(x.limit(n).toPandas() if isinstance(x, DataFrame) else x)

    DATA_PATH = "data/steam_game_output.json"

from pyspark.sql import functions as F, Window
from pyspark.sql.types import (StructType, StructField, StringType, LongType,
                               BooleanType, ArrayType, MapType)

print("Spark", spark.version, "| Databricks" if IS_DATABRICKS else "| local", "|", DATA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reading a semi-structured file
# MAGIC
# MAGIC The whole array sits on **one line**, so `multiLine` has to be on: without it Spark splits the
# MAGIC file on newlines and finds a single unparseable record.
# MAGIC
# MAGIC ### 1.1 Why schema inference is not enough
# MAGIC
# MAGIC Let Spark infer the schema and `tags` — an object whose *keys are the tag names* — becomes a
# MAGIC struct with one column per tag seen anywhere in the file. Inference also has to scan the 61 MB
# MAGIC twice, and it silently picks a type for `required_age`, a field that holds both numbers and
# MAGIC strings.

# COMMAND ----------

inferred = spark.read.option("multiLine", True).json(DATA_PATH)

tags_field = inferred.schema["data"].dataType["tags"].dataType
print("columns Spark invented for `tags`:", len(tags_field.fields))
print("first five:", [f.name for f in tags_field.fields[:5]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 An explicit schema instead
# MAGIC
# MAGIC Declaring `tags` as `MAP<STRING, BIGINT>` turns those 441 phantom columns into one map column
# MAGIC that `explode()` can open into (tag, votes) rows. `required_age` is declared `STRING` and parsed
# MAGIC later, where the parsing rule is visible.

# COMMAND ----------

DATA_SCHEMA = StructType([
    StructField("appid", LongType()),
    StructField("name", StringType()),
    StructField("short_description", StringType()),
    StructField("developer", StringType()),
    StructField("publisher", StringType()),
    StructField("genre", StringType()),                       # comma-separated
    StructField("tags", MapType(StringType(), LongType())),   # tag -> community votes
    StructField("type", StringType()),
    StructField("categories", ArrayType(StringType())),
    StructField("owners", StringType()),                      # bucketed range, e.g. "0 .. 20,000"
    StructField("positive", LongType()),
    StructField("negative", LongType()),
    StructField("price", StringType()),                       # US cents, as a string
    StructField("initialprice", StringType()),
    StructField("discount", StringType()),                    # percent, as a string
    StructField("ccu", LongType()),                           # peak concurrent users
    StructField("languages", StringType()),                   # comma-separated
    StructField("platforms", StructType([
        StructField("windows", BooleanType()),
        StructField("mac", BooleanType()),
        StructField("linux", BooleanType()),
    ])),
    StructField("release_date", StringType()),
    StructField("required_age", StringType()),
    StructField("website", StringType()),
    StructField("header_image", StringType()),
])

SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("data", DATA_SCHEMA),
])

raw = (spark.read.schema(SCHEMA).option("multiLine", True).json(DATA_PATH)
       .select("id", "data.*"))          # flatten the nested `data` struct

print(f"{raw.count():,} games x {len(raw.columns)} columns")
raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 The map opens
# MAGIC
# MAGIC `explode()` on the `tags` map is what gives every later tag question a row-per-tag table.

# COMMAND ----------

display(
    raw.filter(F.col("name") == "Counter-Strike")
       .select("name", F.explode("tags").alias("tag", "votes"))
       .orderBy(F.desc("votes")).limit(8)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cleaning
# MAGIC
# MAGIC Nine rules. Each one is stated with the number of rows it touches, so nothing is silently
# MAGIC dropped or rewritten.
# MAGIC
# MAGIC ### 2.1 Keys and scope
# MAGIC
# MAGIC `id` duplicates `appid`, and every `appid` is unique — there is nothing to deduplicate. One row
# MAGIC is not a game.

# COMMAND ----------

print("rows where id != appid:", raw.filter(F.col("id") != F.col("appid").cast("string")).count())
print("distinct appid:", raw.select("appid").distinct().count(), "of", raw.count(), "rows")
display(raw.groupBy("type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Money
# MAGIC
# MAGIC `price`, `initialprice` and `discount` are strings. Prices are US cents, so a `999` is $9.99.
# MAGIC `price` is the current price and `initialprice` the list price: the two differ on exactly the
# MAGIC 2 518 rows that carry a discount, so the field is internally consistent and `initialprice` is
# MAGIC the one to use for anything about a game's positioning.

# COMMAND ----------

games = (raw.filter(F.col("type") == "game")
         .withColumn("price_usd", F.col("price").cast("int") / 100)
         .withColumn("initial_price_usd", F.col("initialprice").cast("int") / 100)
         .withColumn("discount_pct", F.col("discount").cast("int"))
         .withColumn("is_free", F.col("price").cast("int") == 0))

display(games.select(
    F.round(F.min("price_usd"), 2).alias("min_price"),
    F.round(F.max("price_usd"), 2).alias("max_price"),
    F.sum(F.col("is_free").cast("int")).alias("free_games"),
    F.sum((F.col("discount_pct") > 0).cast("int")).alias("discounted"),
    F.sum((F.col("price") != F.col("initialprice")).cast("int")).alias("price_below_list"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Release dates
# MAGIC
# MAGIC Four shapes in one column: `2020/06/18`, `2020/06/8` (one-digit day), `2015/09` (no day at all)
# MAGIC and empty. Matching each shape before parsing keeps the 99 empty strings as an explicit `NULL`
# MAGIC instead of a silent failure.

# COMMAND ----------

games = (games
    .withColumn("release_date_parsed",
        F.when(F.col("release_date").rlike(r"^\d{4}/\d{1,2}/\d{1,2}$"),
               F.to_date("release_date", "yyyy/M/d"))
         .when(F.col("release_date").rlike(r"^\d{4}/\d{1,2}$"),
               F.to_date("release_date", "yyyy/M")))
    .withColumn("release_year", F.year("release_date_parsed"))
    .withColumn("release_month", F.month("release_date_parsed")))

display(games.select(
    F.min("release_date_parsed").alias("first_release"),
    F.max("release_date_parsed").alias("last_release"),
    F.sum(F.col("release_date_parsed").isNull().cast("int")).alias("unparseable"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Comma-separated lists
# MAGIC
# MAGIC `genre` and `languages` pack several values into one string. Splitting on the comma is not quite
# MAGIC enough: four rows carry a parenthetical note (`English (full audio)`), two a stray semicolon,
# MAGIC and one lists English twice. Stripping the noise, trimming and deduplicating gives clean arrays —
# MAGIC and note that `Spanish - Spain` and `Design & Illustration` mean the separator can only ever be
# MAGIC the comma.

# COMMAND ----------

def split_list(column):
    """Comma-separated string -> trimmed, deduplicated array, parenthetical notes removed."""
    parts = F.split(F.regexp_replace(F.col(column), r"\([^)]*\)|[;*]", ""), ",")
    return F.array_distinct(F.array_remove(F.transform(parts, lambda x: F.trim(x)), ""))

games = (games
    .withColumn("genres", split_list("genre"))
    .withColumn("languages_list", split_list("languages"))
    .withColumn("n_genres", F.size("genres"))
    .withColumn("n_languages", F.size("languages_list")))

display(games.filter(F.col("languages").contains("(")).select("name", "languages", "languages_list"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Publisher names
# MAGIC
# MAGIC Ubisoft appears under five spellings, two of which differ only by a trademark sign or four
# MAGIC trailing tabs. Trimming whitespace and stripping `®`/`™` rewrites 271 rows; it does **not**
# MAGIC merge `Ubisoft` with `Ubisoft Entertainment`, and it should not — deciding that two different
# MAGIC company names are the same firm is a judgement call, not a cleaning rule. The publisher counts in
# MAGIC section 3.1 are therefore a floor, not an exact figure.
# MAGIC
# MAGIC The field also holds co-publisher lists (`Team17, NEXT Studios`), but 3 215 rows contain a comma
# MAGIC and most of them are `Ltd.`-style suffixes, so splitting on it would create more noise than it
# MAGIC removes. The string is kept whole.

# COMMAND ----------

def normalise_name(column):
    cleaned = F.trim(F.regexp_replace(F.regexp_replace(F.col(column), r"[®™]", ""), r"\s+", " "))
    return F.when(cleaned != "", cleaned)

games = (games
    .withColumn("publisher_clean", normalise_name("publisher"))
    .withColumn("developer_clean", normalise_name("developer")))

print("distinct publisher strings:", games.select("publisher").distinct().count(),
      "-> after normalisation:", games.select("publisher_clean").distinct().count())

display(games.filter(F.col("publisher").rlike("^Ubisoft"))
             .groupBy("publisher", "publisher_clean").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Owners
# MAGIC
# MAGIC SteamSpy does not publish a sales figure, it publishes a bracket: `"10,000,000 .. 20,000,000"`.
# MAGIC Two regexes give the bounds, and the midpoint stands in for the value. **68% of the catalogue
# MAGIC sits in the bottom bracket**, so the median of that midpoint is `10,000` almost everywhere and is
# MAGIC useless as a comparison — the rest of the notebook uses two other measures instead: the **median
# MAGIC review count**, and the **share of games above 100 000 owners** (the "breakout rate").

# COMMAND ----------

owners_digits = F.regexp_replace(F.col("owners"), ",", "")
games = (games
    .withColumn("owners_min", F.regexp_extract(owners_digits, r"^(\d+)", 1).cast("long"))
    .withColumn("owners_max", F.regexp_extract(owners_digits, r"\.\.\s*(\d+)$", 1).cast("long")))
games = games.withColumn("owners_mid", (F.col("owners_min") + F.col("owners_max")) / 2)

display(games.groupBy("owners", "owners_min", "owners_max").count().orderBy("owners_min"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Required age
# MAGIC
# MAGIC The field mixes integers and strings, and its odd values are `"MA 15+"`, `"21+"`, `"7+"`, `"35"`
# MAGIC and `"180"` (four times). Pulling the first number out handles the `+` suffixes; anything outside
# MAGIC 0-21 is not an age rating and becomes `NULL` — five rows.
# MAGIC
# MAGIC That fixes the parsing but not the field: **it is 0 for 98.8% of the catalogue**, because Steam
# MAGIC gates mature content through its own content descriptors rather than this legacy attribute.
# MAGIC Section 3.5 answers the brief's question about age-restricted games from the community tags
# MAGIC instead.

# COMMAND ----------

age = F.regexp_extract(F.col("required_age"), r"(\d+)", 1).cast("int")
games = games.withColumn("age_rating", F.when(age.between(0, 21), age))

display(games.groupBy("required_age", "age_rating").count().orderBy(F.desc("count")).limit(25))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 Reviews
# MAGIC
# MAGIC `positive` and `negative` are review counts, not scores. The raw share of positive reviews is
# MAGIC unusable as a ranking on its own: **8 634 games sit at exactly 100%**, most of them on a handful
# MAGIC of reviews. The Wilson 95% lower bound answers the question actually being asked — *how good is
# MAGIC this game, given how little we know about it* — by pulling small samples towards the middle.

# COMMAND ----------

games = (games
    .withColumn("reviews", F.col("positive") + F.col("negative"))
    .withColumn("positive_ratio",
                F.when(F.col("positive") + F.col("negative") > 0,
                       F.col("positive") / (F.col("positive") + F.col("negative")))))

z, n, p = F.lit(1.96), F.col("reviews"), F.col("positive_ratio")
wilson_lower_bound = (p + z * z / (2 * n) - z * F.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)
games = games.withColumn("wilson_score", F.when(n > 0, wilson_lower_bound))

display(games.filter(F.col("positive_ratio") == 1)
             .select("name", "positive", "negative", "positive_ratio", "wilson_score")
             .orderBy("reviews").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9 Platforms, and a revenue proxy
# MAGIC
# MAGIC The `platforms` struct becomes three flags and a count. Revenue is not in the dataset; the
# MAGIC closest available stand-in is **owners x list price**, which section 4.4 uses and section 7
# MAGIC takes apart.

# COMMAND ----------

games = (games
    .withColumn("windows", F.col("platforms.windows"))
    .withColumn("mac", F.col("platforms.mac"))
    .withColumn("linux", F.col("platforms.linux"))
    .withColumn("n_platforms", F.col("platforms.windows").cast("int")
                             + F.col("platforms.mac").cast("int")
                             + F.col("platforms.linux").cast("int"))
    .withColumn("revenue_proxy", F.col("owners_mid") * F.col("initial_price_usd")))

games = games.cache()
print(f"{games.count():,} games ready, {len(games.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.10 Cleaning report

# COMMAND ----------

report = spark.createDataFrame([
    ("2.1  drop type != 'game'",            games.count(),  raw.count() - games.count()),
    ("2.2  price strings -> USD",           games.count(),  games.filter(F.col("discount_pct") > 0).count()),
    ("2.3  release_date -> date",           games.count(),  games.filter(F.col("release_date_parsed").isNull()).count()),
    ("2.4  genre -> array",                 games.count(),  games.filter(F.col("n_genres") == 0).count()),
    ("2.4  languages -> array",             games.count(),  games.filter(F.col("n_languages") == 0).count()),
    ("2.5  publisher name normalised",      games.count(),  games.filter(F.col("publisher") != F.col("publisher_clean")).count()),
    ("2.6  owners bracket -> bounds",       games.count(),  games.filter(F.col("owners_min").isNull()).count()),
    ("2.7  required_age -> 0-21 or NULL",   games.count(),  games.filter(F.col("age_rating").isNull()).count()),
    ("2.8  reviews -> Wilson score",        games.count(),  games.filter(F.col("wilson_score").isNull()).count()),
], ["rule", "rows_in", "rows_affected_or_null"])

display(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.11 How success is measured, from here on
# MAGIC
# MAGIC Nothing in this dataset is a sales figure. Two measures stand in, and every comparison in the
# MAGIC notebook reports both:
# MAGIC
# MAGIC - **median review count** — continuous, and it separates the bottom of the catalogue where the
# MAGIC   owners bracket cannot;
# MAGIC - **breakout rate** — the share of games that reached at least 100 000 owners.
# MAGIC
# MAGIC They are not independent: on the log scale, review count and owner midpoint correlate at
# MAGIC **0.76**, which is what justifies using reviews as a stand-in for reach at all.

# COMMAND ----------

display(games.filter(F.col("reviews") > 0).select(
    F.round(F.corr(F.log("reviews"), F.log(F.greatest("owners_mid", F.lit(1)))), 3).alias("corr_log_reviews_owners")))

def outcome(df, *group_by):
    """Volume and the two success measures, for any grouping."""
    return (df.groupBy(*group_by).agg(
        F.count("*").alias("games"),
        F.percentile_approx("reviews", 0.5).alias("median_reviews"),
        F.round(100 * F.avg((F.col("owners_min") >= 100000).cast("int")), 1).alias("breakout_pct"),
        F.round(F.avg("positive_ratio"), 3).alias("mean_positive_ratio")))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 3. The market
# MAGIC
# MAGIC ## 3.1 Who publishes on Steam
# MAGIC
# MAGIC *Chart: bar, `publisher_clean` x `games`.*

# COMMAND ----------

by_publisher = (games.groupBy("publisher_clean")
    .agg(F.count("*").alias("games"),
         F.round(F.sum("revenue_proxy") / 1e6).alias("revenue_proxy_musd"))
    .filter(F.col("publisher_clean").isNotNull()))

display(by_publisher.orderBy(F.desc("games")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **Big Fish Games** has released the most games — 423, casual hidden-object titles — ahead of
# MAGIC 8floor (202) and SEGA (165). Ubisoft is tenth with 128.
# MAGIC
# MAGIC That ranking says less than the shape behind it. Steam has **29 824 named publishers for
# MAGIC 55 690 games** (134 of which carry no publisher and sit outside every table here), and the
# MAGIC concentration is the finding:

# COMMAND ----------

concentration = (by_publisher
    .withColumn("size", F.when(F.col("games") == 1, "1 game")
                         .when(F.col("games") <= 5, "2-5 games")
                         .when(F.col("games") <= 20, "6-20 games")
                         .otherwise("21+ games"))
    .groupBy("size").agg(F.count("*").alias("publishers"), F.sum("games").alias("games"))
    .withColumn("pct_of_catalogue", F.round(100 * F.col("games") / games.count(), 1)))

display(concentration.orderBy("publishers"))

# COMMAND ----------

# MAGIC %md
# MAGIC **41% of the catalogue comes from publishers that have released exactly one game**, and the
# MAGIC twenty largest publishers together account for 5% of releases. Steam is not a market of a few
# MAGIC big houses — it is a very long tail with a handful of large firms on the end of it.
# MAGIC
# MAGIC Rank the same publishers by the revenue proxy instead of by volume and a different set of names
# MAGIC appears, Ubisoft first among them.
# MAGIC
# MAGIC *Chart: bar, `publisher_clean` x `revenue_proxy_musd`.*

# COMMAND ----------

display(by_publisher.filter(F.col("games") >= 5)
                    .withColumn("revenue_per_game_musd", F.round(F.col("revenue_proxy_musd") / F.col("games"), 1))
                    .orderBy(F.desc("revenue_proxy_musd")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Volume and revenue are close to unrelated: Big Fish Games' 423 releases are worth less than
# MAGIC CD PROJEKT RED's 7. **For Ubisoft the competitor set is not "everyone on Steam"** — it is the
# MAGIC thirty-odd publishers in this table.
# MAGIC
# MAGIC ## 3.2 The release calendar
# MAGIC
# MAGIC *Chart: bar, `release_year` x `games`.*

# COMMAND ----------

per_year = (games.filter(F.col("release_year").isNotNull())
                 .groupBy("release_year").agg(F.count("*").alias("games"))
                 .orderBy("release_year"))
display(per_year.filter(F.col("release_year") >= 2006))

# COMMAND ----------

# MAGIC %md
# MAGIC Releases grew every year to 2018, dipped in 2019 (6 968 against 7 678), then **rose through
# MAGIC Covid — 8 305 in 2020 and 8 823 in 2021**, the two largest years in the dataset. Whatever the
# MAGIC pandemic did to the industry, it did not slow the flow of new games on Steam; the only visible
# MAGIC dip is *before* it.
# MAGIC
# MAGIC 2022 shows 7 455, but the snapshot stops on 11 November, so that year covers ten and a half
# MAGIC months. At the 2022 daily pace it would have landed around 8 600 — flat against 2021, not down.

# COMMAND ----------

display(games.filter(F.col("release_year") == 2022)
             .groupBy("release_month").agg(F.count("*").alias("games")).orderBy("release_month"))

# COMMAND ----------

# MAGIC %md
# MAGIC Within the year, the question that matters for a launch is not how many games ship in a month
# MAGIC but how the games that ship in it do. Restricted to paid games from the seven complete years
# MAGIC 2015-2021:
# MAGIC
# MAGIC *Chart: combo — bars `games`, line `median_reviews`, on `release_month`.*

# COMMAND ----------

display(outcome(games.filter(F.col("release_year").between(2015, 2021) & ~F.col("is_free")), "release_month")
        .orderBy("release_month"))

# COMMAND ----------

# MAGIC %md
# MAGIC **November and December are the worst months to launch in** — median 19 reviews and a 7.5%
# MAGIC breakout rate, against 26-27 reviews and 9-10% in February to May. October and November take the
# MAGIC most releases and return the least attention per release. The holiday window belongs to the
# MAGIC titles that can buy visibility in it.
# MAGIC
# MAGIC **Decision — release window: February to May, and not November or December.**
# MAGIC
# MAGIC ## 3.3 Price
# MAGIC
# MAGIC *Chart: bar, `price_usd` x `games`, on the 12 most common price points.*

# COMMAND ----------

display(games.filter(~F.col("is_free")).select(
    F.round(F.percentile_approx("price_usd", 0.5), 2).alias("median_paid"),
    F.round(F.percentile_approx("price_usd", 0.9), 2).alias("p90"),
    F.round(F.percentile_approx("price_usd", 0.99), 2).alias("p99")))

display(games.groupBy("price_usd").agg(F.count("*").alias("games")).orderBy(F.desc("games")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC Steam prices are anchored on `.99`: after the 7 779 free games, the catalogue piles up on $4.99,
# MAGIC $9.99 and $0.99. The median paid game is **$5.99** and the 99th percentile is $49.99 — a $70
# MAGIC release is off this chart entirely.
# MAGIC
# MAGIC *Chart: combo — bars `games`, line `breakout_pct`, on `band`.*

# COMMAND ----------

price_band = (F.when(F.col("is_free"), "0 free")
               .when(F.col("price_usd") < 5, "1 under $5")
               .when(F.col("price_usd") < 10, "2 $5-10")
               .when(F.col("price_usd") < 20, "3 $10-20")
               .when(F.col("price_usd") < 40, "4 $20-40")
               .otherwise("5 $40+"))

display(outcome(games.withColumn("band", price_band), "band").orderBy("band"))

# COMMAND ----------

# MAGIC %md
# MAGIC Every step up the price ladder buys more of both measures: from 15 median reviews and a 5.9%
# MAGIC breakout rate under $5, to **378 reviews and 31.9% in the $20-40 band**. Above $40 the sample
# MAGIC thins to 567 games and stops improving.
# MAGIC
# MAGIC The arrow does not point the way it looks. A studio does not become successful by charging $30 —
# MAGIC it charges $30 because it built something that can carry the price. What the table does say is
# MAGIC that the $20-40 band is **where games of Ubisoft's scale actually live**, and that pricing a
# MAGIC premium title below $20 puts it among games that are not competing for the same attention.
# MAGIC
# MAGIC **Decision — price: $29.99 to $39.99.**

# COMMAND ----------

display(games.filter(F.col("discount_pct") > 0).select(
    F.count("*").alias("discounted_games"),
    F.round(100 * F.count("*") / games.count(), 1).alias("pct_of_catalogue"),
    F.round(F.avg("discount_pct"), 1).alias("mean_discount"),
    F.percentile_approx("discount_pct", 0.5).alias("median_discount")))

# COMMAND ----------

# MAGIC %md
# MAGIC Discounts are worth one line: **2 518 games, 4.5% of the catalogue, at a median 60% off**. But
# MAGIC this is a one-day snapshot, and Steam's sales are periodic — the figure measures the day the data
# MAGIC was pulled, not how often games go on sale. Nothing in this dataset can answer the second
# MAGIC question.
# MAGIC
# MAGIC ## 3.4 Languages
# MAGIC
# MAGIC *Chart: bar, `language` x `games`.*

# COMMAND ----------

by_language = (games.select(F.explode("languages_list").alias("language"))
                    .groupBy("language").agg(F.count("*").alias("games")))
display(by_language.orderBy(F.desc("games")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **English is on 99% of the catalogue** and is not a decision. The next tier is: German (14 019),
# MAGIC French (13 426), Russian (12 922), Simplified Chinese (12 782), Spanish (12 233), Japanese
# MAGIC (10 368), Italian (9 304). The real question is how far down that list to go.
# MAGIC
# MAGIC *Chart: combo — bars `games`, line `breakout_pct`, on `languages`.*

# COMMAND ----------

language_band = (F.when(F.col("n_languages") <= 1, "1")
                  .when(F.col("n_languages") <= 4, "2-4")
                  .when(F.col("n_languages") <= 9, "5-9")
                  .when(F.col("n_languages") <= 14, "10-14")
                  .when(F.col("n_languages") <= 20, "15-20")
                  .otherwise("21+"))

display(outcome(games.withColumn("languages", language_band), "languages")
        .orderBy(F.desc("median_reviews")))

# COMMAND ----------

# MAGIC %md
# MAGIC Localisation tracks success up to about twenty languages — 16 median reviews and a 7% breakout
# MAGIC rate for English-only, **360 reviews and 39% at 15-20 languages** — and then collapses at 21+.
# MAGIC
# MAGIC That collapse is not a finding, it is a cluster of shovelware. The 1 265 games listing 21 or more
# MAGIC languages are dominated by a handful of publishers pushing dozens of near-identical $0.99 titles,
# MAGIC each shipped with every language Steam offers:

# COMMAND ----------

display(games.filter(F.col("n_languages") >= 21)
             .groupBy("publisher_clean")
             .agg(F.count("*").alias("games"),
                  F.round(F.avg("price_usd"), 2).alias("mean_price"),
                  F.percentile_approx("reviews", 0.5).alias("median_reviews"))
             .orderBy(F.desc("games")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Adding a language string costs nothing and proves nothing. The band that means something is
# MAGIC 10-20, where the localisation is real work.
# MAGIC
# MAGIC **Decision — languages: twelve — EN, DE, FR, RU, zh-Hans, ES, JA, IT, then KO, pt-BR, PL and
# MAGIC zh-Hant.**
# MAGIC
# MAGIC ## 3.5 Age restriction

# COMMAND ----------

display(games.groupBy("age_rating").agg(F.count("*").alias("games")).orderBy(F.desc("games")).limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC Read literally, **only 656 games out of 55 690 (1.2%) carry any age restriction at all**, and 301
# MAGIC are 16+ or over. That is not a description of Steam's catalogue, it is a description of a field
# MAGIC nobody fills in: the store gates mature content through its own content descriptors, and
# MAGIC `required_age` is a legacy attribute left at 0.
# MAGIC
# MAGIC The community tags do carry the signal, so the brief's question is better answered with them.
# MAGIC
# MAGIC *Chart: bar, `mature` x `breakout_pct`.*

# COMMAND ----------

MATURE_TAGS = ["Violent", "Gore", "Nudity", "Sexual Content", "NSFW", "Hentai", "Mature"]

games = games.withColumn("mature",
    F.size(F.array_intersect(F.map_keys("tags"), F.array(*[F.lit(t) for t in MATURE_TAGS]))) > 0)

display(outcome(games, "mature"))
print("mature-tagged games that also declare an age rating:",
      games.filter(F.col("mature") & (F.col("age_rating") > 0)).count(),
      "of", games.filter("mature").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **7 103 games — 12.8% of the catalogue — carry mature content tags**, and only 388 of them
# MAGIC declare an age rating for it. They also do better than the rest: median 71 reviews against 23,
# MAGIC and a 19.6% breakout rate against 10.8%.
# MAGIC
# MAGIC Caution on the direction: mature themes are concentrated in the kind of large action and RPG
# MAGIC titles that would outperform anyway. The table is not evidence that adding blood sells copies.
# MAGIC It is evidence that **a mature rating is not a commercial handicap on Steam**, which is the only
# MAGIC thing the decision needs.
# MAGIC
# MAGIC **Decision — age rating: mature (17+/PEGI 18) is not a constraint on the concept.**
# MAGIC
# MAGIC ## 3.6 The best rated games

# COMMAND ----------

print("games with a 100% positive ratio:", games.filter(F.col("positive_ratio") == 1).count())
display(games.filter(F.col("positive_ratio") == 1)
             .select("name", "positive", "negative", "positive_ratio")
             .orderBy("reviews").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Ranking on the raw ratio returns 8 634 games tied at 100%, most of them on one or two reviews.
# MAGIC The Wilson lower bound breaks the tie by asking how much evidence sits behind the score; with a
# MAGIC floor of 500 reviews it produces a list that means something.

# COMMAND ----------

display(games.filter(F.col("reviews") >= 500)
             .select("name", "publisher_clean", "reviews",
                     F.round("positive_ratio", 4).alias("positive_ratio"),
                     F.round("wilson_score", 4).alias("wilson_score"),
                     "release_year")
             .orderBy(F.desc("wilson_score")).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC The top of the list is not made of blockbusters. *Aseprite* is a pixel-art editor, *A Short Hike*
# MAGIC and *Patrick's Parabox* are one-person indie games, and the highest-rated large title is
# MAGIC **Portal 2 at 98.8% over 309 441 reviews** — which is the more useful benchmark, because
# MAGIC sustaining that ratio at that volume is the hard part.
# MAGIC
# MAGIC **Reference point — a Ubisoft-scale title doing well on Steam sits near 90% positive; 95%+ at
# MAGIC 100 000+ reviews is exceptional.**

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 4. Genres
# MAGIC
# MAGIC `genre` holds one to seven labels per game. Exploding it gives one row per (game, genre), so a
# MAGIC game counted under Action is also counted under RPG — the shares below add up to more than 100%
# MAGIC by construction.

# COMMAND ----------

# the raw comma-separated string is dropped: the exploded label replaces it
genre_rows = games.drop("genre").select("*", F.explode("genres").alias("genre")).cache()
print(f"{genre_rows.count():,} (game, genre) rows for {games.count():,} games")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 What is on the shelf
# MAGIC
# MAGIC *Chart: bar, `genre` x `games`.*

# COMMAND ----------

display(genre_rows.groupBy("genre").agg(F.count("*").alias("games"))
        .withColumn("pct_of_catalogue", F.round(100 * F.col("games") / games.count(), 1))
        .orderBy(F.desc("games")).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC **Indie is on 71% of the catalogue** — and it is not a genre. Neither are *Early Access* (11%)
# MAGIC nor *Free to Play* (6%): they describe how a game is funded and sold, not what it is. Excluding
# MAGIC those three, the shelf is **Action (43%), Casual (40%), Adventure (38%), Strategy (20%),
# MAGIC Simulation (19%), RPG (17%)**, and everything else is under 5%.
# MAGIC
# MAGIC For Ubisoft the labels that matter are the six real ones. The rest of section 4 keeps all of
# MAGIC them in the tables, because the contrast between *Indie* and the others is itself informative.
# MAGIC
# MAGIC ## 4.2 Which genres are liked
# MAGIC
# MAGIC Restricted to the 21 669 games with at least 50 reviews, so the ratio means something.
# MAGIC
# MAGIC *Chart: bar, `genre` x `median_positive_ratio`.*

# COMMAND ----------

display(genre_rows.filter(F.col("reviews") >= 50).groupBy("genre")
        .agg(F.count("*").alias("games"),
             F.round(F.percentile_approx("positive_ratio", 0.5), 3).alias("median_positive_ratio"),
             F.round(F.sum("positive") / (F.sum("positive") + F.sum("negative")), 3).alias("pooled_ratio"))
        .filter(F.col("games") >= 100).orderBy(F.desc("median_positive_ratio")))

# COMMAND ----------

# MAGIC %md
# MAGIC The spread is narrow — **0.82 for Casual and Adventure down to 0.78 for Strategy** — with one
# MAGIC exception. **Massively Multiplayer sits at 0.665**, fifteen points below everything else, and it
# MAGIC is the genre where the median game is actively disliked. Live-service games are judged on
# MAGIC servers, monetisation and updates long after launch, and this is what that judgement looks like
# MAGIC in aggregate.
# MAGIC
# MAGIC The two columns disagree on purpose. The pooled ratio weights every review equally, so it
# MAGIC measures *the average experience across the genre*; the median weights every game equally, so it
# MAGIC measures *the typical game*. Indie is 0.813 typical and 0.886 pooled — its big titles are much
# MAGIC better liked than its median one.
# MAGIC
# MAGIC **Decision — avoid a live-service / MMO structure. The satisfaction penalty is the largest
# MAGIC single effect in the genre data.**
# MAGIC
# MAGIC ## 4.3 Do publishers have favourite genres
# MAGIC
# MAGIC *Chart: stacked bar, `publisher_clean` x `games`, grouped by `genre`.*

# COMMAND ----------

top_publishers = [r[0] for r in by_publisher.orderBy(F.desc("games")).limit(8).collect()]

display(genre_rows.filter(F.col("publisher_clean").isin(top_publishers))
        .groupBy("publisher_clean", "genre").agg(F.count("*").alias("games"))
        .withColumn("rank", F.row_number().over(
            Window.partitionBy("publisher_clean").orderBy(F.desc("games"))))
        .filter(F.col("rank") <= 3).orderBy("publisher_clean", "rank"))

# COMMAND ----------

# MAGIC %md
# MAGIC Emphatically yes, and the specialisation is near-total: **Big Fish Games is 419 Casual and 393
# MAGIC Adventure out of 423 games**, 8floor is 202 Casual out of 202, Strategy First is Strategy,
# MAGIC Choice of Games is RPG. These are not diversified catalogues — each of these publishers has one
# MAGIC formula and repeats it.
# MAGIC
# MAGIC Ubisoft's own Steam catalogue is the same shape, around a different centre:

# COMMAND ----------

display(genre_rows.filter(F.col("publisher_clean").rlike("^Ubisoft"))
        .groupBy("genre").agg(F.count("*").alias("games"),
                              F.round(F.sum("revenue_proxy") / 1e6).alias("revenue_proxy_musd"))
        .orderBy(F.desc("games")).limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC **Action (75 titles) and Adventure (49)**, then Strategy and RPG. Whatever the next game
# MAGIC is, it will be read by players against that catalogue.
# MAGIC
# MAGIC ## 4.4 Which genres are worth entering
# MAGIC
# MAGIC The revenue proxy is `owners x list price`. Summed by genre it gives each genre's share of the
# MAGIC market's value, which can be set against its share of releases:
# MAGIC
# MAGIC **opportunity = share of proxy revenue / share of releases.** Above 1, the genre returns more
# MAGIC value than the shelf space it takes.
# MAGIC
# MAGIC *Chart: scatter, `share_of_releases` x `share_of_revenue`, size `games`, label `genre`.*

# COMMAND ----------

total_games = games.count()
total_revenue = games.agg(F.sum("revenue_proxy")).collect()[0][0]

genre_economics = (genre_rows.groupBy("genre")
    .agg(F.count("*").alias("games"),
         F.sum("revenue_proxy").alias("revenue"),
         F.round(F.avg("revenue_proxy")).alias("mean_revenue_per_game"),
         F.round(100 * F.avg((F.col("owners_min") >= 100000).cast("int")), 1).alias("breakout_pct"),
         F.round(100 * F.avg(F.col("is_free").cast("int")), 1).alias("pct_free"))
    .filter(F.col("games") >= 150)
    .withColumn("share_of_releases", F.round(100 * F.col("games") / total_games, 1))
    .withColumn("share_of_revenue", F.round(100 * F.col("revenue") / total_revenue, 1))
    .withColumn("opportunity", F.round((F.col("revenue") / total_revenue) / (F.col("games") / total_games), 2)))

display(genre_economics.drop("revenue").orderBy(F.desc("opportunity")))

# COMMAND ----------

# MAGIC %md
# MAGIC Three genres return more than they take — **Massively Multiplayer at 2.96, RPG at 1.81, Action
# MAGIC at 1.52** — and two are badly crowded: **Casual takes 40% of the shelf for 8.7% of the value
# MAGIC (0.22), Indie 71% for 35% (0.49)**. Strategy, Adventure and Simulation sit at parity.
# MAGIC
# MAGIC Two readings to be careful with. **Free to Play scores 0.02 not because free games make no money
# MAGIC but because the proxy cannot see any revenue that is not a store price** — an in-game economy is
# MAGIC invisible here. The same blindness deflates Massively Multiplayer, 53% of which is free, and it
# MAGIC still comes out on top; its true opportunity is higher than 2.96, not lower. And the whole column
# MAGIC inherits the proxy's biases, so it ranks genres, it does not value them.
# MAGIC
# MAGIC Set against 4.2, that gives one clean answer: **MMO buys the best economics and the worst
# MAGIC satisfaction. RPG is second on economics with no such penalty.**
# MAGIC
# MAGIC ## 4.5 Is any genre emerging
# MAGIC
# MAGIC *Chart: grouped bar, `genre` x `pct`, grouped by `release_year`.*

# COMMAND ----------

mix = (genre_rows.filter(F.col("release_year").isin(2017, 2022))
       .groupBy("release_year", "genre").agg(F.count("*").alias("games")))
year_totals = mix.groupBy("release_year").agg(F.sum("games").alias("total"))

genre_mix = (mix.join(year_totals, "release_year")
    .withColumn("pct", F.round(100 * F.col("games") / F.col("total"), 1))
    .groupBy("genre").pivot("release_year", [2017, 2022]).agg(F.first("pct"))
    .withColumnRenamed("2017", "pct_2017").withColumnRenamed("2022", "pct_2022")
    .withColumn("shift_pts", F.round(F.col("pct_2022") - F.col("pct_2017"), 1)))

display(genre_mix.orderBy(F.desc("pct_2022")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC Five years apart, the mix barely moves: Indie 25.1% → 24.7%, Action 15.6% → 14.7%, Casual
# MAGIC 14.0% → 14.5%. The only shifts worth naming are **Early Access, 3.5% → 5.7%**, and **Free to Play
# MAGIC collapsing from 2.3% to 0.4%** — the latter partly a labelling change, since free games kept
# MAGIC their share of the catalogue while the genre tag stopped being applied.
# MAGIC
# MAGIC **No genre is emerging.** A concept does not need to catch a wave here, because there isn't one;
# MAGIC it needs to be good in a category that already pays.
# MAGIC
# MAGIC ## 4.6 The slot
# MAGIC
# MAGIC Putting 4.2 and 4.4 together with the price band from 3.3 — games released from 2018, priced
# MAGIC $20-40:
# MAGIC
# MAGIC *Chart: combo — bars `games`, line `breakout_pct`, on `genre`.*

# COMMAND ----------

display(outcome(genre_rows.filter((F.col("release_year") >= 2018)
                                  & F.col("price_usd").between(20, 40)), "genre")
        .filter(F.col("games") >= 150).orderBy(F.desc("breakout_pct")))

# COMMAND ----------

# MAGIC %md
# MAGIC In the band Ubisoft would actually price into, **Action reaches a 34.9% breakout rate over 776
# MAGIC games and RPG 33.8% over 471**, against 27.8% for Adventure. Adventure trades six points of
# MAGIC breakout for the best satisfaction of the three (0.790). Action and RPG overlap heavily — most of these games carry both labels —
# MAGIC which is the combination the recommendation lands on.
# MAGIC
# MAGIC **Decision — genre: Action-RPG, single-player, premium. Not Casual, not Indie-positioned, not
# MAGIC live service.**

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 5. Platforms
# MAGIC
# MAGIC ## 5.1 What Steam runs on
# MAGIC
# MAGIC *Chart: bar, `platform` x `games`.*

# COMMAND ----------

display(games.select(
    F.sum(F.col("windows").cast("int")).alias("windows"),
    F.sum(F.col("mac").cast("int")).alias("mac"),
    F.sum(F.col("linux").cast("int")).alias("linux")))

display(games.groupBy("windows", "mac", "linux").agg(F.count("*").alias("games"))
        .orderBy(F.desc("games")))

# COMMAND ----------

# MAGIC %md
# MAGIC **Windows is not a choice: 55 675 of 55 690 games support it, and the 15 that do not are
# MAGIC curiosities.** Mac reaches 22.9% of the catalogue and Linux 15.2%, and they travel together —
# MAGIC 6 806 games ship all three, more than the 5 951 that add Mac alone.
# MAGIC
# MAGIC So the platform question is only ever "do we port", never "which one".
# MAGIC
# MAGIC ## 5.2 Which genres get ported
# MAGIC
# MAGIC *Chart: bar, `genre` x `pct_mac` and `pct_linux`.*

# COMMAND ----------

display(genre_rows.groupBy("genre").agg(
            F.count("*").alias("games"),
            F.round(100 * F.avg(F.col("mac").cast("int")), 1).alias("pct_mac"),
            F.round(100 * F.avg(F.col("linux").cast("int")), 1).alias("pct_linux"))
        .filter(F.col("games") >= 150).orderBy(F.desc("pct_mac")))

# COMMAND ----------

# MAGIC %md
# MAGIC The spread runs from **Strategy at 27.6% Mac and 16.8% Linux down to Early Access at 14.6% and
# MAGIC 10.3%**, with Action near the bottom of the range at 19.2% and 14.2%. Turn-based and 2D-heavy
# MAGIC categories port more; action games, which lean hardest on the graphics stack, port least. The
# MAGIC technical cost of the port is visible in the genre mix.
# MAGIC
# MAGIC Note the two ends: Strategy is 8 points above Action on Mac, but no genre is anywhere near
# MAGIC Windows' 99.97%.
# MAGIC
# MAGIC ## 5.3 Does porting pay
# MAGIC
# MAGIC *Chart: combo — bars `games`, line `breakout_pct`, on `n_platforms`.*

# COMMAND ----------

display(outcome(games, "n_platforms").orderBy("n_platforms"))

# COMMAND ----------

# MAGIC %md
# MAGIC Three platforms beats one on both measures — 79 median reviews against 21, a 21.2% breakout rate
# MAGIC against 10.0%. That comparison is worthless on its own: a studio that expects a hit is exactly
# MAGIC the studio that pays for ports, so the gap could be entirely selection.
# MAGIC
# MAGIC Two checks. First, hold the price band and the release era fixed:

# COMMAND ----------

display(outcome(games.filter(F.col("release_year") >= 2018)
                     .withColumn("band", price_band)
                     .withColumn("ported", F.col("n_platforms") > 1),
                "band", "ported").orderBy("band", "ported"))

# COMMAND ----------

# MAGIC %md
# MAGIC The gap survives in five of the six bands: at $10-20, 87 median reviews ported against 38; at
# MAGIC $20-40, 494 against 238 and a 31.4% breakout rate against 25.3%. It reverses only above $40, on
# MAGIC 48 ported games — too few to read. Second, if porting were a big-publisher behaviour, porting
# MAGIC rates would climb with catalogue size — they do not:

# COMMAND ----------

publisher_size = by_publisher.select("publisher_clean", F.col("games").alias("publisher_games"))

display(games.join(publisher_size, "publisher_clean")
        .withColumn("publisher_size", F.when(F.col("publisher_games") == 1, "1 game")
                                       .when(F.col("publisher_games") <= 5, "2-5 games")
                                       .otherwise("6+ games"))
        .groupBy("publisher_size").agg(
            F.count("*").alias("games"),
            F.round(100 * F.avg((F.col("n_platforms") > 1).cast("int")), 1).alias("pct_ported")))

# COMMAND ----------

# MAGIC %md
# MAGIC **24.0% for one-game publishers, 27.8% for small ones, 26.6% for large ones** — flat. Porting is
# MAGIC not something only big studios do, which removes the most obvious confounder without removing
# MAGIC the causality problem: within any size class, the games that get ported are still the ones
# MAGIC someone believed in.
# MAGIC
# MAGIC What survives is a floor, not a lift: the association is consistent, and the Linux port also
# MAGIC covers the Steam Deck, which shipped in February 2022 and is not yet visible in this snapshot.
# MAGIC
# MAGIC **Decision — platforms: Windows at launch, Mac and Linux planned in. Linux is the Steam Deck
# MAGIC route and the cheaper of the two to add once the engine is portable.**

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 6. The brief for the next game
# MAGIC
# MAGIC Six decisions, each from the section that produced it.
# MAGIC
# MAGIC | | decision | evidence |
# MAGIC |---|---|---|
# MAGIC | **Genre** | Action-RPG, single-player, premium | Action 1.52 and RPG 1.81 on revenue-to-releases; 34.9% and 33.8% breakout in the $20-40 band (4.4, 4.6) |
# MAGIC | **Not** | live service or MMO | MMO carries the best economics *and* a 0.665 median positive ratio, fifteen points below every other genre (4.2) |
# MAGIC | **Price** | $29.99 - $39.99 | the $20-40 band returns 378 median reviews and a 31.9% breakout rate; above $40 the market thins to 567 games (3.3) |
# MAGIC | **Platforms** | Windows at launch, Mac and Linux planned in | Windows is 99.97% of the catalogue; ported games lead inside every price band, and porting is not a big-studio behaviour (5.1, 5.3) |
# MAGIC | **Languages** | twelve: EN, DE, FR, RU, zh-Hans, ES, JA, IT, KO, pt-BR, PL, zh-Hant | success climbs to the 15-20 band, 39% breakout, then collapses into shovelware above 21 (3.4) |
# MAGIC | **Window** | February to May | November and December take the most releases and return the least: 19 median reviews and 7.5% breakout, against 26-27 and 9-10% in spring (3.2) |
# MAGIC | **Rating** | mature is not a constraint | 12.8% of the catalogue is mature-tagged and outperforms the rest; Steam's own age field is empty for 98.8% of games (3.5) |
# MAGIC
# MAGIC Two things this study says about the field the game will land in, beyond the product itself:
# MAGIC
# MAGIC - **The competitor set is about thirty publishers, not fifty thousand.** 41% of Steam's catalogue
# MAGIC   comes from publishers with a single release, and the twenty largest account for 5% of it.
# MAGIC   Ubisoft's rivals are the names in 3.1's revenue table.
# MAGIC - **There is no wave to catch.** The genre mix moved by less than two points in five years. The
# MAGIC   concept has to win on execution inside a category that already pays, not on timing.
# MAGIC
# MAGIC The quality bar is section 3.6's: **90% positive is a good Ubisoft-scale result on Steam, and 95%
# MAGIC at scale is exceptional** — Portal 2 territory.
# MAGIC
# MAGIC ---
# MAGIC # 7. What this dataset cannot decide
# MAGIC
# MAGIC **It stops on 11 November 2022.** Every 2022 figure covers ten and a half months, and nothing
# MAGIC after that date exists — no Steam Deck effect, no post-2022 pricing.
# MAGIC
# MAGIC **There are no sales.** `owners` is SteamSpy's *estimate*, published as a bracket, and 68% of the
# MAGIC catalogue falls in the bottom one. The revenue proxy multiplies that bracket's midpoint by the
# MAGIC list price, so it ignores Steam's 30% cut, regional pricing, discounts, refunds, bundles, free
# MAGIC keys — and it values every free-to-play game at zero, which is why section 4.4 reads *Free to
# MAGIC Play: 0.02* for a business model that funds some of the largest games in the table.
# MAGIC
# MAGIC **Reviews are not players.** They correlate with owners at 0.76 on the log scale, which is enough
# MAGIC to rank and not enough to size.
# MAGIC
# MAGIC **Nothing here is causal.** Price, ports and localisation are all things a studio chooses because
# MAGIC it already expects the game to sell. Section 5.3 removes the most obvious confounder for porting
# MAGIC and cannot remove the rest. Every decision in section 6 is a reading of where successful games
# MAGIC are, not a recipe for becoming one.
# MAGIC
# MAGIC **Delisted games are absent.** The catalogue is what was on sale in November 2022, so failures
# MAGIC that were pulled never appear — the breakout rates are, if anything, optimistic.
# MAGIC
# MAGIC **One store, one region.** Steam is not consoles, not mobile, not the Epic store, and the prices
# MAGIC are US dollars.
