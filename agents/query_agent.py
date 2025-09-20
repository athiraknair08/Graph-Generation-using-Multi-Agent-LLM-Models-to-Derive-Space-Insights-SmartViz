
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import dateparser
from dateparser.search import search_dates
from datetime import datetime
import calendar
import unicodedata

# -------------------------------
# Paths & Data
# -------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH     = PROJECT_ROOT / "data" / "metrics_app_merged_cleaned.csv"

# Load embedding model for semantic room matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load known room names from cleaned dataset
df = pd.read_csv(CSV_PATH)
room_list = df['display_name'].dropna().astype(str).unique().tolist()
room_embeddings = model.encode(room_list, convert_to_tensor=True)

# Available metrics in cleaned dataset
available_metrics = df['metric_name'].dropna().astype(str).unique().tolist()
available_metrics_lower = [m.lower() for m in available_metrics]
lower_to_original_metric = {m.lower(): m for m in available_metrics}

# -------------------------------
# Keyword → metric candidates
# (keys are LOWERCASE; we map to dataset's exact casing via lower_to_original_metric)
# -------------------------------
metric_keywords = {
    "occupancy": [
        "utilised", "utilized", "usage", "used",
        "less utilised", "less used", "underused",
        "under-utilised", "under-utilized", "empty",
        "vacant", "busiest", "most used", "busy",
        "least crowded", "most crowded", "peak usage"
    ],
    "occupancyoptimised": ["optimised occupancy", "optimized occupancy"],
    "peoplemotion": ["movement", "motion", "footfall", "people motion"],
    "peoplemotionoptimised": ["optimised motion", "optimized motion"],
    "co2": ["stuffy", "air quality", "ventilation", "co₂", "carbon dioxide", "co2"],
    "temp": ["temperature", "hot", "cold", "warmest", "hottest", "coldest", "temp"],
    "humidity": ["humidity", "moisture"],
    "feelslike": ["feels like", "feelslike"],
    "soundlevel": ["noise", "sound", "sound level"],
    "ambientnoise": ["ambient noise", "background noise"],
    "tvoc": ["tvoc", "voc"],
    "exttemp": ["external temperature", "outside temperature", "ext temp"],
}

# -------------------------------
# Normalizers & helpers
# -------------------------------
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
    flags=re.I
)

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("CO₂", "co2").replace("Co₂", "co2").replace("co₂", "co2")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _get_month_range(dt: datetime) -> Tuple[str, str]:
    y, m = dt.year, dt.month
    start = datetime(y, m, 1).date().isoformat()
    end_day = calendar.monthrange(y, m)[1]
    end = datetime(y, m, end_day).date().isoformat()
    return start, end


def _parse_dates(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try month phrases, then explicit YYYY-MM-DD, then general 'between ... and ...' via search_dates.
    Returns (start_date, end_date) in YYYY-MM-DD or (None, None).
    """
    ql = q.lower()

    # 1) Month phrases (handles one or two months)
    months = [dateparser.parse(m, settings={'PREFER_DAY_OF_MONTH': 'first'}) for m in MONTH_RE.findall(ql)]
    months = [m for m in months if m]
    if len(months) == 1:
        return _get_month_range(months[0])
    elif len(months) >= 2:
        s1, _ = _get_month_range(months[0])
        _, e2 = _get_month_range(months[1])
        return s1, e2

    # 2) Strict YYYY-MM-DD (one or two)
    strict = re.findall(r"\d{4}-\d{2}-\d{2}", ql)
    if len(strict) == 1:
        return strict[0], None
    elif len(strict) >= 2:
        return strict[0], strict[1]

    # 3) Loose: search_dates to find two dates in order
    try:
        found = search_dates(q)
        if found and len(found) >= 2:
            s = found[0][1].date().isoformat()
            e = found[1][1].date().isoformat()
            return s, e
    except Exception:
        pass

    return None, None

def _detect_chart_hint(ql: str) -> Optional[str]:
    """
    Return one of: 'scatter','pie','heatmap','box','bar','line' (None if unknown).
    """
    if any(k in ql for k in ["scatter", "correlation", "correlate", " vs ", "versus"]):
        return "scatter"
    if any(k in ql for k in ["pie", "donut", "doughnut", "proportion", "share", "composition"]):
        return "pie"
    if any(k in ql for k in ["heatmap", "hourly pattern", "hour-of-day", "hour of day", "by hour"]):
        return "heatmap"
    if any(k in ql for k in ["box", "boxplot", "distribution", "quartile", "outliers"]):
        return "box"
    if any(k in ql for k in ["compare", "top", "worst", "highest", "lowest", "min", "max", "busiest", "least utilised", "underused"]):
        return "bar"
    if any(k in ql for k in ["trend", "over time", "time series", "timeline", "evolution"]):
        return "line"
    return None

def _map_keywords_to_existing_metrics(ql: str) -> List[str]:
    hits: List[str] = []
    for meta_key, keywords in metric_keywords.items():
        if any(kw in ql for kw in keywords):
            # map to actual dataset metric casing if present
            if meta_key in lower_to_original_metric:
                hits.append(lower_to_original_metric[meta_key])
    # de-dup but keep order
    seen = set()
    ordered = []
    for m in hits:
        ml = m.lower()
        if ml not in seen:
            ordered.append(m)
            seen.add(ml)
    return ordered

def _explicit_metric_mentions(ql: str) -> List[str]:
    """
    If the query literally contains a dataset metric token (any case), return those.
    """
    hits: List[str] = []
    for m in available_metrics:
        tok = re.escape(m.lower())
        if re.search(rf"\b{tok}\b", ql):
            hits.append(m)
    # de-dup
    seen = set()
    out = []
    for m in hits:
        ml = m.lower()
        if ml not in seen:
            out.append(m)
            seen.add(ml)
    return out

def _detect_room(user_query: str) -> Optional[str]:
    """
    Prefer exact/substring match first (case-insensitive).
    Fallback to embeddings with a slightly lower threshold.
    """
    q = _norm(user_query)
    ql = q.lower()

    # Exact/substring match
    candidates = [r for r in room_list if r and r.lower() in ql]
    if candidates:
        # choose the longest matched name to avoid short collisions
        return sorted(candidates, key=lambda s: len(s), reverse=True)[0]

    # Embedding fallback
    try:
        q_emb = model.encode(user_query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, room_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        if best_score > 0.45:  
            return room_list[best_idx]
    except Exception:
        pass
    return None

# -------------------------------
# Main Parser
# -------------------------------
def detect_intent_and_entities(user_query: str) -> Dict:
    q_norm = _norm(user_query)
    ql = q_norm.lower()

    # --- chart hint (so graph_agent can route correctly) ---
    chart_hint = _detect_chart_hint(ql)

    # --- intent detection ---
    if any(k in ql for k in ["plot", "graph", "visual", "trend", "compare"]) or chart_hint in {"scatter","pie","heatmap","box","bar","line"}:
        intent = "graph"
    elif any(k in ql for k in ["insight", "summary", "interpret", "explain"]):
        intent = "insight"
    elif any(k in ql for k in ["text", "ocr", "image", "pdf", "document"]):
        intent = "ocr"
    else:
        intent = "unknown"

    # --- metric detection: explicit mentions first, then keyword mapping ---
    detected_metrics: List[str] = []
    explicit = _explicit_metric_mentions(ql)
    if explicit:
        detected_metrics.extend(explicit)

    if not detected_metrics:
        detected_metrics.extend(_map_keywords_to_existing_metrics(ql))

    # De-dup while keeping order
    seen = set()
    ordered_metrics = []
    for m in detected_metrics:
        ml = m.lower()
        if ml not in seen:
            ordered_metrics.append(m)
            seen.add(ml)

    detected_metric = ordered_metrics[0] if ordered_metrics else None

    # If usage language present but no metric found, prefer occupancy if available
    usage_words = ["busiest", "most used", "most utilised", "most utilized", "highest usage",
                   "most crowded", "underused", "under-utilised", "under-utilized", "less utilised",
                   "least used", "quietest", "low usage"]
    if (not detected_metric) and any(w in ql for w in usage_words) and "occupancy" in lower_to_original_metric:
        detected_metric = lower_to_original_metric["occupancy"]
        ordered_metrics.insert(0, detected_metric)

    # Resolve intent if unknown but we found a metric
    if intent == "unknown" and detected_metric:
        intent = "insight"  # default to insight; graph_agent chooses chart from chart_hint/text

    # --- date parsing ---
    start_date, end_date = _parse_dates(user_query)

    # --- room detection ---
    detected_room = _detect_room(user_query)

    return {
        "intent": intent,
        "metric_name": detected_metric,
        "metric_names": ordered_metrics,
        "room": detected_room,
        "start_date": start_date,
        "end_date": end_date,
        "chart_hint": chart_hint,        
        "original_query": user_query,    # always carry forward
    }
