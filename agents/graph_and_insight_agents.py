
# graph_agent.py


import re
import calendar
import pandas as pd
import plotly.express as px
import numpy as np
import dateparser
from sqlalchemy import create_engine, text
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from dateparser.search import search_dates
import unicodedata
from typing import List, Tuple, Optional

# ----------------------- TUNABLE DEFAULTS -----------------------
TOPK_CORR_ROOMS_DEFAULT = 10       # scatter declutter limit
DEFAULT_CORR_AGG_FREQ = "d"        # 'd' (daily) unless "hourly" is requested
CONTINUOUS_DEFAULT_PCTL = 0.90     # dynamic threshold if user didn't give one
MIN_STREAK_GRANULARITY = "h"       # compute streaks in hourly buckets
MAX_BARS = 15                      # limit bars for readability
# ---------------------------------------------------------------

# DB connection
engine = create_engine("postgresql+psycopg2://postgres:SmartViz2025@localhost:5432/smartviz")

# LLM setup (used by text-insight only; charts/insight below are data-first)
llm = Ollama(model="mistral")
base_prompt = PromptTemplate.from_template("""
You are an AI data analyst. The user asked: "{query}"

Here are exact summary stats for the requested data:

{table}

Please provide a concise insight focusing on notable patterns, extremes, or comparisons.
""")
insight_chain: Runnable = base_prompt | llm | StrOutputParser()

# ------------------------- Metric Aliases -------------------------
# Map natural-language terms to canonical DB metric_name values (24 metrics).
METRIC_ALIASES = {
    # Environment (internal)
    "temperature": "temp", "temp": "temp", "indoor temperature": "temp",
    "humidity": "humidity", "rh": "humidity", "relative humidity": "humidity",
    "co₂": "co2", "co2": "co2", "carbon dioxide": "co2",
    "sound": "soundlevel", "sound level": "soundlevel", "noise": "soundlevel",
    "tvoc": "tvoc", "voc": "tvoc",

    # Occupancy / motion
    "occupancy": "occupancy", "people": "occupancy",
    "occupancy optimised": "occupancyoptimised", "occupancyoptimized": "occupancyoptimised",
    "people motion": "peoplemotion", "motion": "peoplemotion",
    "people motion optimised": "peoplemotionoptimised", "peoplemotion optimized": "peoplemotionoptimised",

    # External weather
    "external temperature": "exttemp", "ext temperature": "exttemp",
    "exttemp": "exttemp", "exterior temp": "exttemp", "outside temp": "exttemp",
    "external humidity": "exthumidity", "ext humidity": "exthumidity", "exthumidity": "exthumidity",
    "feels like": "feelslike", "feelslike": "feelslike",
    "cloud cover": "cloudcover", "cloudcover": "cloudcover",
    "precip": "precip", "precipitation": "precip", "rain": "precip",
    "wind": "windspeed", "wind speed": "windspeed", "windspeed": "windspeed",
    "wind gust": "windgust", "gust": "windgust", "windgust": "windgust",
    "wind direction": "winddir", "winddir": "winddir",

    # Preservation / materials
    "preservation index": "preservationindex", "preservationindex": "preservationindex",
    "days to mold": "daystomold", "daystomold": "daystomold",
    "equilibrium moisture content": "equilibriummoisturecontent", "emc": "equilibriummoisturecontent",
    "mechanical damage": "mechanicaldamage", "mechanicaldamage": "mechanicaldamage",
    "metal corrosion": "metalcorrosion", "metalcorrosion": "metalcorrosion",

    # Batteries / ambient
    "battery": "batterylevel", "battery level": "batterylevel", "batterylevel": "batterylevel",
    "ambient noise": "ambientnoise", "ambientnoise": "ambientnoise",

    # Exact DB keys (pass-through safety)
    "temp": "temp", "humidity": "humidity", "co2": "co2", "soundlevel": "soundlevel",
    "occupancyoptimised": "occupancyoptimised", "peoplemotion": "peoplemotion",
    "peoplemotionoptimised": "peoplemotionoptimised", "exttemp": "exttemp",
    "exthumidity": "exthumidity", "feelslike": "feelslike",
    "cloudcover": "cloudcover", "precip": "precip",
    "windspeed": "windspeed", "windgust": "windgust", "winddir": "winddir",
    "preservationindex": "preservationindex", "daystomold": "daystomold",
    "equilibriummoisturecontent": "equilibriummoisturecontent",
    "mechanicaldamage": "mechanicaldamage", "metalcorrosion": "metalcorrosion",
}

def _alias_metric_token(tok: str) -> str:
    if not tok:
        return tok
    t = str(tok).strip().lower().replace("—", "-").replace("–", "-")
    t = t.replace("co₂", "co2")
    return METRIC_ALIASES.get(t, t)

def _apply_metric_aliases_from_query(metrics: List[str], q_norm: str) -> List[str]:
    """
    Normalize parsed metrics and also scan the natural-language query for
    additional metric mentions. Returns a de-duplicated ordered list.
    """
    out = [_alias_metric_token(m) for m in metrics if m]
    # scan query for alias keys (longest first for multi-word matches)
    keys = sorted(METRIC_ALIASES.keys(), key=len, reverse=True)
    for k in keys:
        if re.search(rf"\b{re.escape(k)}\b", q_norm):
            mapped = METRIC_ALIASES[k]
            if mapped not in out:
                out.append(mapped)
    # de-dupe (preserve order)
    seen, deduped = set(), []
    for m in out:
        if m not in seen:
            deduped.append(m); seen.add(m)
    return deduped

# ------------------------- Helpers -------------------------

def extract_dates_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (start, end) from free text.
    Supports:
      - 'April 2025'  -> first..last of month
      - 'Apr–Jun 2025' / 'Apr-Jun 2025' / 'April to June 2025'
      - explicit 'YYYY-MM-DD' pairs
    """
    if not query:
        return None, None

    q = unicodedata.normalize("NFKD", query)
    q = q.replace("—", "-").replace("–", "-")  # normalize dashes

    # 1) YYYY-MM-DD pairs
    iso = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    if len(iso) >= 2:
        return iso[0], iso[1]

    # 2) Month-range like "Apr-Jun 2025" or "April-June 2025" (or "... to ... 2025")
    mon_pat = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
    m = re.search(rf"\b{mon_pat}\b\s*-\s*\b{mon_pat}\b\s+(\d{{4}})", q, flags=re.I)
    if not m:
        m = re.search(rf"\b{mon_pat}\b\s*(?:to|through)\s*\b{mon_pat}\b\s+(\d{{4}})", q, flags=re.I)
    if m:
        m1, m2, year = m.group(1), m.group(2), int(m.group(3))
        mm = {
            "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
            "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
            "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
        }
        m1n, m2n = mm[m1.lower()], mm[m2.lower()]
        first = pd.Timestamp(year=year, month=m1n, day=1)
        last  = pd.Timestamp(year=year, month=m2n, day=1) + pd.offsets.MonthEnd(1)
        return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")

    # 3) Single month like "April 2025"
    m_single = re.search(rf"\b{mon_pat}\b\s+(\d{{4}})", q, flags=re.I)
    if m_single:
        mon, year = m_single.group(1), int(m_single.group(2))
        mm = {
            "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
            "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
            "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
        }
        mn = mm[mon.lower()]
        first = pd.Timestamp(year=year, month=mn, day=1)
        last  = pd.Timestamp(year=year, month=mn, day=1) + pd.offsets.MonthEnd(1)
        return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")

    # 4) Fallback to dateparser’s fuzzy search
    found = search_dates(q)
    if found and len(found) >= 2:
        return found[0][1].strftime("%Y-%m-%d"), found[1][1].strftime("%Y-%m-%d")

    return None, None


def _end_exclusive(end: Optional[str]) -> Optional[str]:
    """Return next-day YYYY-MM-DD for end-exclusive filtering."""
    if not end:
        return None
    try:
        return (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return end


def _fetch_raw_rows(metric: str, room: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Pull raw rows with optional room / date filters.
    Case-insensitive metric match. Use start_time window [start, end_exclusive).
    """
    sql = """
        SELECT start_time, end_time, value, display_name, metric_name
        FROM metrics_app_merged
        WHERE LOWER(metric_name) = LOWER(:metric)
    """
    params = {"metric": metric}

    if room:
        sql += " AND display_name ILIKE :room"
        params["room"] = f"%{room}%"

    if start:
        sql += " AND start_time >= :start"
        params["start"] = start

    if end:
        sql += " AND start_time < :end"
        params["end"] = _end_exclusive(end)

    print(f"[SQL] Params: {params}")
    df = pd.read_sql(text(sql), con=engine, params=params)
    return _standardize_cols(df)


def _standardize_cols(df: pd.DataFrame, metric_name: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure downstream consistency:
    - rename display_name -> room (keep original if already present)
    - attach metric_name column if provided or missing
    - sort by start_time if present
    """
    d = df.copy()
    if "room" not in d.columns and "display_name" in d.columns:
        d = d.rename(columns={"display_name": "room"})
    if "metric_name" not in d.columns and metric_name is not None:
        d["metric_name"] = metric_name
    if "start_time" in d.columns:
        d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce", utc=True)
        d = d.sort_values("start_time")
    return d


def _list_available_metrics() -> List[str]:
    """Fetch distinct metric names from the DB (lowercased)."""
    try:
        dfm = pd.read_sql(text("SELECT DISTINCT metric_name FROM metrics_app_merged"), con=engine)
        return sorted({str(x).lower() for x in dfm["metric_name"].dropna().unique().tolist()})
    except Exception as e:
        print(f"[DEBUG] Failed to list metrics: {e}")
        return []


def _normalize_query(q: str) -> str:
    """Normalize unicode and aliases e.g. CO₂ -> co2, carbon dioxide -> co2."""
    if not q:
        return ""
    qn = unicodedata.normalize("NFKD", q)
    qn = qn.replace("co₂", "co2").replace("Co₂", "co2").replace("CO₂", "co2")
    qn = re.sub(r"\bcarbon\s+dioxide\b", "co2", qn, flags=re.I)
    qn = re.sub(r"\s+", " ", qn)
    return qn.lower().strip()


def _ensure_room(df: pd.DataFrame) -> pd.DataFrame:
    """Always expose a 'room' column."""
    if "room" in df.columns:
        return df
    if "display_name" in df.columns:
        return df.rename(columns={"display_name": "room"})
    return df


def _detect_right_now(q: str) -> bool:
    return any(k in q for k in ["right now", "current", "currently", "now"])


def _extract_threshold(q: str) -> Optional[float]:
    """Extract numeric threshold like '> 900' or 'above 21.5' if present."""
    m = re.search(r"[><=]\s*([\d.]+)", q)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m2 = re.search(r"\b(?:over|above|exceed(?:ing)?)\s+([\d.]+)\b", q)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def _longest_streak_above(series: pd.Series) -> int:
    """Compute the longest consecutive run of True in a boolean series."""
    s = series.astype(int)
    c = (s.groupby((s != s.shift()).cumsum()).cumsum()) * s
    return int(c.max()) if len(c) else 0


def _emit_eval_meta(metric: str, room: str, direction: str, time_window: str, value: float) -> str:
    """Machine-readable block consumed by the evaluator."""
    safe_metric = "" if metric is None else str(metric)
    safe_room = "" if room is None else str(room)
    safe_dir = "" if direction is None else str(direction)
    safe_tw = "" if time_window is None else str(time_window)
    try:
        val_str = f"{float(value):.6f}"
    except Exception:
        val_str = ""
    return (
        "\n[EVAL_META]\n"
        f"metric={safe_metric}\n"
        f"room={safe_room}\n"
        f"direction={safe_dir}\n"
        f"time_window={safe_tw}\n"
        f"value={val_str}\n"
        "[/EVAL_META]\n"
    )

def _fmt_window(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"{start} → {end}"
    return "the selected period"

def _fmt_unit(unit: str) -> str:
    return f" {unit}" if unit else ""

def _hide_in_html_comment(meta: str) -> str:
    # Invisible in Streamlit markdown; still present for evaluators.
    return f"\n<!-- {meta.strip()} -->\n"

def _strength_of_r(r: float) -> str:
    ar = abs(r)
    if ar >= 0.8: return "very strong"
    if ar >= 0.6: return "strong"
    if ar >= 0.4: return "moderate"
    if ar >= 0.2: return "weak"
    return "very weak or none"

# --------------------- NEW: Multi-metric helpers ---------------------

def _fetch_avg_by_room_for_metrics(metrics: List[str], start: str, end: str, room_like: Optional[str] = None) -> pd.DataFrame:
    """Return pivoted dataframe: columns = metrics, index = room."""
    if not metrics:
        return pd.DataFrame()
    placeholders = ",".join([f":m{i}" for i in range(len(metrics))])
    params = {f"m{i}": m for i, m in enumerate(metrics)}
    if start: params["start"] = start
    if end:   params["end"]   = _end_exclusive(end)
    room_clause = ""
    if room_like:
        room_clause = "AND display_name ILIKE :room"
        params["room"] = f"%{room_like}%"

    sql = f"""
        SELECT display_name AS room, LOWER(metric_name) AS metric_name, AVG(value) AS avg_value
        FROM metrics_app_merged
        WHERE LOWER(metric_name) IN ({placeholders})
          AND start_time >= :start AND start_time < :end
          {room_clause}
        GROUP BY room, metric_name
    """
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="room", columns="metric_name", values="avg_value").reset_index()
    return pivot


def _rank_multi_metric(pivot_df: pd.DataFrame, metrics: List[str], k: int = 10) -> pd.DataFrame:
    work = pivot_df.copy()
    # add missing columns as NaN to stay generic
    for m in metrics:
        if m not in work.columns:
            work[m] = np.nan
        work[f"rank_{m}"] = work[m].rank(method="min", ascending=False)
    work["rank_sum"] = work[[f"rank_{m}" for m in metrics]].sum(axis=1)
    topk = work.sort_values(["rank_sum"] + metrics, ascending=[True] + [False]*len(metrics)).head(min(k, MAX_BARS))
    return topk


def _plot_grouped_bar_multi_metric(topk: pd.DataFrame, metrics: List[str], title: str):
    # wide-form grouped bars (multiple y columns)
    fig = px.bar(topk, x="room", y=metrics, barmode="group", title=title)
    fig.update_layout(xaxis_tickangle=-45, height=600, width=1600, margin=dict(t=60, b=200))
    return fig

# --------------------- Graph + Data builder ---------------------

def generate_graph_from_query(parsed_query):
    """
    Return (fig, dataframe, chart_type)
    chart_type ∈ {"time_series","bar","pie","scatter","heatmap","box"}
    The returned dataframe is standardized to contain 'room' when applicable.
    """
    print(f"[GRAPH] Received parsed_query: {parsed_query}")
    try:
        # ---------------- Common fields ----------------
        metrics = parsed_query.get("metric_names") or []
        if not metrics and parsed_query.get("metric_name"):
            metrics = [parsed_query["metric_name"]]
        metrics = [str(m).lower() for m in metrics if m]  # normalize

        room       = parsed_query.get("room")
        start, end = parsed_query.get("start_date"), parsed_query.get("end_date")
        q_raw      = parsed_query.get("original_query") or ""
        chart_hint = parsed_query.get("chart_hint")
        q_norm     = _normalize_query(q_raw)

        # ---- Apply metric aliases & also scan query for more metrics (multi-metric trend support)
        metrics = _apply_metric_aliases_from_query(metrics, q_norm)

        # Heuristic default if user talks about "usage"
        if (not metrics) and re.search(
            r"\b(busiest|most used|most utilised|most utilized|highest usage|most crowded|underused|less utilised|least used|quietest|low usage)\b",
            q_norm
        ):
            metrics = ["occupancy"]

        # ---------------- Date handling ----------------
        # Prefer parser dates, but if the query contains an explicit month-range and
        # the parser returned a too-narrow span (<= 31 days), re-extract from text.
        month_range_pat = r"\b(jan|feb|mar|apr|may|jun|july?|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b\s*(?:-|–|—|\bto\b|\bthrough\b)\s*\b(jan|feb|mar|apr|may|jun|july?|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b"
        if start and end:
            try:
                span_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
            except Exception:
                span_days = None
            if re.search(month_range_pat, q_norm, flags=re.I) and (span_days is None or span_days <= 31):
                s2, e2 = extract_dates_from_query(q_raw)
                if s2 and e2:
                    start, end = s2, e2
        else:
            s2, e2 = extract_dates_from_query(q_raw)
            start = start or s2
            end   = end or e2

        # ---------------- Chart intent helpers ----------------
        def wants_scatter() -> bool:
            if chart_hint == "scatter":
                return True
            return any(w in q_norm for w in ("scatter", "correlation", "correlate", " vs ", "versus"))

        def wants_pie() -> bool:
            if chart_hint == "pie":
                return True
            return any(k in q_norm for k in ["pie", "pie chart", "donut", "doughnut", "proportion", "share", "composition"])

        def wants_heatmap() -> bool:
            if chart_hint == "heatmap":
                return True
            return any(s in q_norm for s in ("heatmap", "hourly pattern", "hour-of-day", "hour of day", "daily vs hourly", "by hour"))

        def wants_box() -> bool:
            if chart_hint == "box":
                return True
            return any(k in q_norm for k in ("box", "boxplot", "distribution", "quartile", "outliers"))

        # Stronger bar keywords so "bar chart comparing ..." routes to BAR
        bar_keywords = [
            "bar", "bar chart", "compare", "comparison", "rank", "top", "worst",
            "highest", "max", "minimum", "lowest", "min", "least",
            "less utilised", "less used", "underused", "under-utilised", "busiest",
            "across rooms", "by room"
        ]
        def wants_bar() -> bool:
            if chart_hint == "bar":
                return True
            return any(k in q_norm for k in bar_keywords)

        # ---------------- MULTI-METRIC BAR (combined rank) ----------------
        if wants_bar() and len(metrics) > 1:
            df_pivot = _fetch_avg_by_room_for_metrics(metrics, start, end, room_like=room)
            if df_pivot.empty:
                return None, df_pivot, "bar"
            topk = _rank_multi_metric(df_pivot, metrics, k=MAX_BARS)
            title = f"Top {len(topk)} Rooms by Combined Rank: {', '.join([m.upper() for m in metrics])}" + (f" ({start} → {end})" if start and end else "")
            fig = _plot_grouped_bar_multi_metric(topk, metrics, title)
            return fig, _ensure_room(topk), "bar"

        # ---------------- PIE ----------------
        if wants_pie():
            if not metrics:
                return None, pd.DataFrame(), "pie"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "pie"

            pie_df = (
                df_raw.groupby("room", as_index=False)["value"]
                      .sum()
                      .rename(columns={"value": "total"})
                      .sort_values("total", ascending=False)
            )
            pie_df["metric_name"] = metric
            fig = px.pie(
                pie_df, names="room", values="total",
                title=f"{metric.upper()} share by room" + (f" ({start} → {end})" if start and end else "")
            )
            return fig, _standardize_cols(df_raw, metric), "pie"

        # ---------------- SNAPSHOT: right now ----------------
        if _detect_right_now(q_norm) and not start and not end:
            if not metrics:
                return None, pd.DataFrame(), "bar"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric)
            if df_raw.empty:
                return None, df_raw, "bar"
            latest = (
                df_raw.sort_values("start_time")
                      .groupby("room", as_index=False)
                      .tail(1)
            )
            latest["metric_name"] = metric
            fig = px.bar(
                latest, x="room", y="value",
                title=f"Current {metric.upper()} across rooms"
            )
            return fig, _standardize_cols(latest, metric), "bar"

        # ---------------- THRESHOLD ----------------
        threshold_match = re.search(
            r"(temperature|temp|co2|occupancy|humidity|motion|people|peoplemotion)[^\d><=]*[><=]\s*([\d.]+)", q_norm
        )
        if threshold_match:
            if not metrics:
                return None, pd.DataFrame(), "bar"
            metric = metrics[0]
            threshold = float(threshold_match.groups()[1])
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "bar"
            grouped = (
                df_raw.groupby("room", as_index=False)["value"]
                      .mean()
                      .rename(columns={"value": "avg_val"})
                      .query(f"avg_val > {threshold}")
                      .sort_values("avg_val", ascending=False)
                      .head(MAX_BARS)
            )
            grouped["metric_name"] = metric
            fig = px.bar(
                grouped, x="room", y="avg_val",
                title=f"Rooms with Avg. {metric.upper()} > {threshold}"
            )
            return fig, _standardize_cols(grouped.rename(columns={"avg_val": "value"}), metric), "bar"

        # ---------------- CONTINUOUS / SUSTAINED ----------------
        continuous_words = ("continuous", "consist", "sustain", "persist")
        if any(w in q_norm for w in continuous_words):
            if not metrics:
                return None, pd.DataFrame(), "bar"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "bar"

            d = df_raw.copy()
            d["ts"] = pd.to_datetime(d["start_time"], errors="coerce", utc=True).dt.floor(MIN_STREAK_GRANULARITY)
            hourly = (
                d.groupby(["room", "ts"], as_index=False)["value"]
                 .mean()
                 .sort_values(["room", "ts"])
            )

            explicit = _extract_threshold(q_norm)
            if explicit is not None:
                thr = explicit
                thr_label = f">{thr:g}"
            else:
                thr = float(hourly["value"].quantile(CONTINUOUS_DEFAULT_PCTL))
                thr_label = f">P{int(CONTINUOUS_DEFAULT_PCTL*100)} ({thr:.0f})"

            def room_streak(g: pd.DataFrame) -> int:
                return _longest_streak_above(g["value"] > thr)

            streaks = (
                hourly.groupby("room")
                      .apply(room_streak)
                      .rename("longest_streak")
                      .reset_index()
                      .sort_values("longest_streak", ascending=False)
            )
            streaks["streak_hours"] = streaks["longest_streak"].astype(int)

            top = streaks.head(MAX_BARS).copy()
            top["metric_name"] = metric
            top = top.rename(columns={"streak_hours": "value"})
            fig = px.bar(
                top, x="room", y="value", color="room",
                title=f"Longest continuous {metric.upper()} {thr_label}" + (f" — {start} to {end}" if start and end else "")
            )
            return fig, _standardize_cols(top[["room", "value", "metric_name"]], metric), "bar"

        # ---------------- BAR (rank rooms by mean) ----------------
        if wants_bar():
            if not metrics:
                return None, pd.DataFrame(), "bar"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "bar"

            grouped = (
                df_raw.groupby("room", as_index=False)["value"]
                      .mean()
                      .rename(columns={"value": "value"})
            )
            grouped["metric_name"] = metric

            ascending = any(k in q_norm for k in [
                "worst", "minimum", "lowest", "min", "least",
                "less utilised", "less used", "underused", "under-utilised", "quietest", "low usage"
            ])
            title = ("Lowest " if ascending else "Top ") + f"{metric.upper()} Rooms"
            grouped = grouped.sort_values("value", ascending=ascending).head(MAX_BARS)

            fig = px.bar(
                grouped, x="room", y="value",
                title=title, labels={"value": f"Average {metric.upper()}"}
            )
            fig.update_traces(width=0.5)
            fig.update_layout(
                bargap=0.2, bargroupgap=0.1, xaxis_tickangle=-45,
                height=600, width=1600, margin=dict(t=60, b=200)
            )
            return fig, _standardize_cols(grouped, metric), "bar"

        # ---------------- SCATTER / CORRELATION ----------------
        if wants_scatter():
            if len(metrics) < 2:
                available = _list_available_metrics()
                mentioned = []
                for m_ in available:
                    if re.search(rf"\b{re.escape(m_)}\b", q_norm):
                        mentioned.append(m_)
                merged = []
                for m_ in metrics + mentioned:
                    if m_ and m_ not in merged:
                        merged.append(m_)
                metrics = merged[:2]

            if len(metrics) < 2:
                return None, pd.DataFrame(), "scatter"

            m1, m2 = metrics[0], metrics[1]
            agg_freq = "h" if any(k in q_norm for k in ["hour", "hourly"]) else DEFAULT_CORR_AGG_FREQ
            freq_label = "hourly" if agg_freq == "h" else "daily"

            def fetch_and_align(m: str, _room: Optional[str]) -> pd.DataFrame:
                d = _fetch_raw_rows(m, _room, start, end)
                if d.empty:
                    return pd.DataFrame(columns=["room", "ts", m])
                d = d.copy()
                d["ts"] = pd.to_datetime(d["start_time"], utc=True, errors="coerce").dt.floor("h")
                if agg_freq == "d":
                    d["ts"] = d["ts"].dt.floor("d")
                d = (
                    d.groupby(["room", "ts"], as_index=False)["value"]
                     .mean()
                     .rename(columns={"value": m})
                )
                return d

            d1 = fetch_and_align(m1, room)
            d2 = fetch_and_align(m2, room)

            if room and (d1.empty or d2.empty):
                d1_any = fetch_and_align(m1, None)
                d2_any = fetch_and_align(m2, None)
                if not d1_any.empty or not d2_any.empty:
                    print(f"[SCATTER] No data for room '{room}' in {start}→{end}. Showing building-wide scatter.")
                    d1, d2 = d1_any, d2_any

            if d1.empty or d2.empty:
                merged = pd.DataFrame(columns=["room", "ts", m1, m2])
            else:
                merged = pd.merge(d1, d2, on=["room", "ts"], how="inner").dropna()

            if merged.empty:
                return None, merged, "scatter"

            def safe_corr(g):
                try:
                    return g[[m1, m2]].corr(method="pearson").iloc[0, 1]
                except Exception:
                    return float("nan")

            by_room_r = (
                merged.groupby("room").apply(safe_corr)
                      .rename("pearson_r").reset_index()
                      .dropna(subset=["pearson_r"])
            )

            filtered = merged
            if not by_room_r.empty:
                by_room_r["_abs"] = by_room_r["pearson_r"].abs()
                keep_rooms = (
                    by_room_r.sort_values("_abs", ascending=False)
                             .head(TOPK_CORR_ROOMS_DEFAULT)["room"].tolist()
                )
                filtered = merged[merged["room"].isin(keep_rooms)].copy()

            if filtered.empty:
                return None, filtered, "scatter"

            title_room_suffix = "" if not room else f" — requested: {room}"
            fig = px.scatter(
                filtered, x=m1, y=m2, color="room",
                title=f"{m1.upper()} vs {m2.upper()} ({freq_label} means per room)"
                      + (f" — {start} to {end}" if start and end else "")
                      + title_room_suffix
            )
            fig.update_traces(marker=dict(size=6), opacity=0.35)
            out = filtered[["room", m1, m2, "ts"]].copy()
            return fig, _ensure_room(out), "scatter"

        # ---------------- HEATMAP ----------------
        if wants_heatmap():
            if not metrics:
                return None, pd.DataFrame(), "heatmap"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "heatmap"
            d = df_raw.copy()
            d["date"] = pd.to_datetime(d["start_time"]).dt.date
            d["hour"] = pd.to_datetime(d["start_time"]).dt.hour
            pivot = d.pivot_table(index="hour", columns="date", values="value", aggfunc="mean")
            fig = px.imshow(
                pivot, labels=dict(x="Date", y="Hour", color="Value"),
                title=f"Hourly Heatmap of {metric.upper()}" + (f" — {start} to {end}" if start and end else "")
            )
            d["metric_name"] = metric
            return fig, _standardize_cols(d, metric), "heatmap"

        # ---------------- BOX ----------------
        if wants_box():
            if not metrics:
                return None, pd.DataFrame(), "box"
            metric = metrics[0]
            df_raw = _fetch_raw_rows(metric, room, start, end)
            if df_raw.empty:
                return None, df_raw, "box"
            d = df_raw.copy()
            d["metric_name"] = metric
            fig = px.box(
                d, x="room", y="value", points="all",
                title=f"Distribution of {metric.upper()} by Room" + (f" — {start} to {end}" if start and end else "")
            )
            return fig, _standardize_cols(d, metric), "box"

        # ---------------- Default: TIME SERIES ----------------
        combined = pd.DataFrame()
        if not metrics:
            return None, combined, "time_series"

        for m_ in metrics:
            dfm = _fetch_raw_rows(m_, room, start, end)
            if dfm.empty:
                continue
            if room:
                dfm = dfm[dfm["room"].astype(str).str.lower().str.contains(str(room).lower())]
            dfm["metric_name"] = m_
            combined = pd.concat([combined, dfm], ignore_index=True)

        # Fallback to building-wide if requested room has no rows
        if combined.empty and room:
            combined_any = pd.DataFrame()
            for m_ in metrics:
                dfm = _fetch_raw_rows(m_, room=None, start=start, end=end)
                if dfm.empty:
                    continue
                dfm["metric_name"] = m_
                combined_any = pd.concat([combined_any, dfm], ignore_index=True)
            if not combined_any.empty:
                print(f"[TS] No data for room '{room}' in {start}→{end}. Showing building-wide trend.")
                combined = combined_any

        if combined.empty:
            return None, pd.DataFrame(), "time_series"

        combined = combined.sort_values("start_time")
        title_room_suffix = "" if not room else f" — requested: {room}"
        fig = px.line(
            combined, x="start_time", y="value", color="metric_name",
            title=f"{', '.join([m.upper() for m in metrics])} Trend"
                  + (f" — {start} to {end}" if start and end else "")
                  + title_room_suffix,
            markers=True
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Value",
                          margin=dict(t=50, b=40), height=500)
        return fig, _standardize_cols(combined), "time_series"

    except Exception as e:
        print(f"❌ Error in generate_graph_from_query: {e}")
        return None, pd.DataFrame(), "time_series"


# --------------------- Insight generator (unchanged interface, extended) ---------------------

def _fmt_window(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"{start} → {end}"
    if start and not end:
        return f"from {start}"
    if end and not start:
        return f"until {end}"
    return "full range"

def _fmt_unit(unit: str) -> str:
    return f" {unit}" if unit else ""


def generate_insight_from_data(
    df: pd.DataFrame,
    query: str,
    start: str = None,
    end: str = None,
    metric_hint: str = None
) -> str:
    """
    Generate human-readable insight text from the SAME data used for the chart.

    Fixes:
    - Clamp uses UTC consistently (avoids tz-aware vs tz-naive comparison).
    - Insight window text prefers user-provided dates; falls back to data span.
    - EVAL_META is printed to console only.
    """
    try:
        print(f"[INSIGHT] got df: rows={0 if df is None else len(df)} cols={[] if df is None else list(df.columns)}")
        print(f"[INSIGHT] start={start} end={end} query='{(query or '')[:120]}'")

        q_norm = _normalize_query(query or "")

        # ---- Decide if we need to rebuild the dataframe
        need_rebuild = (
            df is None or df.empty or
            (("value" not in df.columns) and (df.select_dtypes(include="number").shape[1] < 1))
        )

        # ---- Metric resolution (hint > df > query aliases > heuristic > default)
        metric_key = None
        if metric_hint:
            metric_key = _alias_metric_token(metric_hint)

        if metric_key is None and (not need_rebuild) and "metric_name" in df.columns and not df["metric_name"].isna().all():
            metric_key = str(df["metric_name"].dropna().astype(str).value_counts().idxmax()).lower()

        if metric_key is None:
            for k in sorted(METRIC_ALIASES.keys(), key=len, reverse=True):
                if re.search(rf"\b{re.escape(k)}\b", q_norm):
                    metric_key = METRIC_ALIASES[k]
                    break

        if metric_key is None and re.search(r"\b(utilis|utiliz|usage|used|underused|under-utilis|under-utiliz)\w*\b", q_norm):
            metric_key = "occupancy"

        if metric_key is None:
            metric_key = "temperature"

        # ---- Resolve dates from query if missing
        if not start or not end:
            s2, e2 = extract_dates_from_query(query or "")
            start = start or s2
            end = end or e2

        # ---- (Re)build if needed
        if need_rebuild:
            raw = _fetch_raw_rows(metric_key, start=start, end=end)
            raw = _standardize_cols(raw, metric_key)
            if raw.empty:
                return "No data found for that time window."
            df = raw

        # ---- Clamp to requested window in UTC (fixes tz comparison error)
        if start and end and ("start_time" in df.columns):
            ts = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
            st = pd.to_datetime(start, utc=True, errors="coerce")
            et_excl = pd.to_datetime(_end_exclusive(end), utc=True, errors="coerce")
            df = df.loc[(ts >= st) & (ts < et_excl)].copy()

        # ---------- Correlation / Scatter narrative ----------
        num_cols = df.select_dtypes(include="number").columns.tolist()
        candidate_cols = [c for c in num_cols if c != "value"]
        if (("ts" in df.columns) or ("start_time" in df.columns)) and len(candidate_cols) >= 2:
            m1, m2 = candidate_cols[:2]
            key_pair = "_vs_".join(sorted([m1, m2])) + "_pearson_r"

            try:
                overall_r = df[[m1, m2]].corr(method="pearson").iloc[0, 1]
            except Exception:
                overall_r = float("nan")

            per_room = pd.DataFrame()
            if "room" in df.columns:
                def safe_corr(g):
                    try:
                        return g[[m1, m2]].corr(method="pearson").iloc[0, 1]
                    except Exception:
                        return float("nan")
                per_room = (
                    df.groupby("room").apply(safe_corr).rename("pearson_r")
                      .reset_index().dropna(subset=["pearson_r"])
                )

            if start and end:
                window_txt = f"{start} → {end}"
                eval_window = f"{start}→{end}"
            else:
                if "start_time" in df.columns:
                    smin = pd.to_datetime(df["start_time"]).min().date()
                    smax = pd.to_datetime(df["start_time"]).max().date()
                    window_txt = f"{smin} → {smax}"
                    eval_window = window_txt.replace(" → ", "→")
                else:
                    window_txt = _fmt_window(start, end)
                    eval_window = "full"

            strength = _strength_of_r(overall_r)
            sign_word = "positive" if (pd.notna(overall_r) and overall_r > 0) else ("negative" if pd.notna(overall_r) and overall_r < 0 else "neutral")
            r_str  = f"{overall_r:.4f}" if pd.notna(overall_r) else "nan"
            r2_val = (overall_r ** 2) if pd.notna(overall_r) else float("nan")
            r2_str = f"{r2_val:.4f}" if pd.notna(r2_val) else "nan"

            lines = [
                f"From {window_txt}, the overall correlation **Pearson r = {r_str}** ({sign_word}, {strength}). "
                f"R² = **{r2_str}**."
            ]

            print(_emit_eval_meta(metric=key_pair, room="", direction="corr", time_window=eval_window, value=overall_r))

            if not per_room.empty:
                top_pos1 = per_room.sort_values("pearson_r", ascending=False).head(1)
                if not top_pos1.empty:
                    pr = top_pos1.iloc[0]
                    lines.append(f"Strongest positive: **{pr['room']}** (r = **{pr['pearson_r']:.4f}**).")
                    print(_emit_eval_meta(metric=key_pair, room=str(pr["room"]), direction="max", time_window=eval_window, value=float(pr["pearson_r"])))
                top_neg1 = per_room.sort_values("pearson_r", ascending=True).head(1)
                if not top_neg1.empty:
                    nr = top_neg1.iloc[0]
                    lines.append(f"Strongest negative: **{nr['room']}** (r = **{nr['pearson_r']:.4f}**).")
                    print(_emit_eval_meta(metric=key_pair, room=str(nr["room"]), direction="min", time_window=eval_window, value=float(nr["pearson_r"])) )

            return " ".join(lines)

        # ---------- Multi-metric pivot insight ----------
        known_metrics = sorted(set(METRIC_ALIASES.values()))
        metric_cols = [c for c in df.columns if c in known_metrics]
        if len(metric_cols) >= 2 and "room" in df.columns:
            unit_map = {"temp": "°C", "temperature": "°C", "humidity": "%", "co2": "ppm", "occupancy": "people"}
            def fmt_val(m, v):
                if pd.isna(v): return "n/a"
                u = unit_map.get(m, "")
                return f"{v:.2f}{(' ' + u) if u else ''}"

            window_txt = f"{start} → {end}" if (start and end) else _fmt_window(start, end)

            parts = []
            for m in metric_cols:
                row = df.sort_values(m, ascending=False).iloc[0]
                parts.append(f"{m.title()} {window_txt}: highest in **{row['room']}** at **{fmt_val(m, row[m])}**.")
                print(_emit_eval_meta(metric=m, room=str(row["room"]), direction="max",
                                      time_window=("full" if not (start and end) else f"{start}→{end}"),
                                      value=float(row[m]) if pd.notna(row[m]) else float("nan")))
            if "rank_sum" in df.columns:
                leader = df.sort_values("rank_sum", ascending=True).iloc[0]
                combo = ", ".join([f"{m}={fmt_val(m, leader.get(m, np.nan))}" for m in metric_cols])
                parts.append(f"Overall leader by combined rank: **{leader['room']}** ({combo}).")

            head = df.head(5)
            preview = "; ".join([
                f"{r['room']} (" + "; ".join([f"{m}={fmt_val(m, r.get(m, np.nan))}" for m in metric_cols]) + ")"
                for _, r in head.iterrows()
            ])
            parts.append(f"Top rooms: {preview}.")
            return "  \n".join(parts)

        # ---------- Friendly labels/units ----------
        label_map = {
            "temp": ("temperature", "°C"),
            "temperature": ("temperature", "°C"),
            "co2": ("CO₂", "ppm"),
            "occupancy": ("occupancy", "people"),
            "peoplemotion": ("motion", "counts"),
            "humidity": ("humidity", "%"),
        }
        friendly_label, unit = label_map.get(metric_key.lower(), (metric_key.replace("_", " "), ""))

        wants_min = any(k in q_norm for k in ["lowest", "min", "minimum", "coolest", "least", "less utilised", "less used", "underused", "under-utilised", "low usage"])
        wants_max = any(k in q_norm for k in ["highest", "max", "maximum", "top", "worst", "hottest", "busiest", "most crowded", "peak"])
        right_now = any(k in q_norm for k in ["right now", "current", "currently", "now"])
        target_name = "min" if wants_min else ("max" if (wants_max or right_now) else "avg")

        # ---------- Bar-style ----------
        if "start_time" not in df.columns and {"room", "value"}.issubset(df.columns):
            per_room = df.groupby("room", as_index=False)["value"].mean().rename(columns={"value": "avg_val"})
            per_room = per_room.sort_values("avg_val", ascending=(target_name == "min"))
            best = per_room.iloc[0]
            u = f" {unit}" if unit else ""
            window_txt = f"{start} → {end}" if (start and end) else _fmt_window(start, end)
            headline = (
                f"{friendly_label.title()} {window_txt}: "
                f"{'lowest' if target_name=='min' else ('highest' if target_name=='max' else 'highest average')} "
                f"in **{best['room']}** at **{best['avg_val']:.2f}{u}**."
            )
            top3 = per_room.head(3)
            bullets = ", ".join([f"{r} ({v:.2f}{u})" for r, v in zip(top3['room'], top3['avg_val'])])
            print(_emit_eval_meta(metric=metric_key, room=str(best["room"]), direction=target_name,
                                  time_window=("full" if not (start and end) else f"{start}→{end}"),
                                  value=float(best["avg_val"])))
            return f"{headline} Top by average: {bullets}."

        # ---------- Time series ----------
        if {"start_time", "value", "room"}.issubset(df.columns):
            stats = df.groupby("room", as_index=False)["value"].agg(min_val="min", avg_val="mean", max_val="max").round(2)
            pick = {"min": "min_val", "avg": "avg_val", "max": "max_val"}[target_name]
            best = stats.sort_values(pick, ascending=(pick == "min_val")).iloc[0]

            if start and end:
                window_txt = f"{start} → {end}"
            else:
                try:
                    smin = pd.to_datetime(df["start_time"]).min().date()
                    smax = pd.to_datetime(df["start_time"]).max().date()
                    window_txt = f"{smin} → {smax}"
                except Exception:
                    window_txt = _fmt_window(start, end)

            u = f" {unit}" if unit else ""
            headline = (
                f"{friendly_label.title()} {window_txt}: "
                f"{'lowest' if target_name=='min' else ('highest' if target_name=='max' else 'highest average')} "
                f"in **{best['room']}** at **{best[pick]:.2f}{u}**."
            )
            print(_emit_eval_meta(metric=metric_key, room=str(best["room"]), direction=target_name,
                                  time_window=window_txt.replace(" → ", "→"),
                                  value=float(best[pick])))
            return headline

        return "Could not compute an insight from the available data."

    except Exception as e:
        print(f"❌ Error in generate_insight_from_data: {e}")
        return "An error occurred while generating insights."


