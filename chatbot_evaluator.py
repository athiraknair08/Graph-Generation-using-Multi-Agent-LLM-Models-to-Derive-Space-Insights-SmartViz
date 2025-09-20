# chatbot_evaluator.py
from __future__ import annotations

import os
import re
import math
import tempfile
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

# ------------------ Safe DeepEval cache location (optional) ------------------
if "DEEPEVAL_HOME" not in os.environ:
    safe_home = os.path.join(tempfile.gettempdir(), "deepeval_cache")
    try:
        os.makedirs(safe_home, exist_ok=True)
        os.environ["DEEPEVAL_HOME"] = safe_home
    except Exception:
        pass

# ------------------ Optional imports (never hard-fail) ------------------
_trulens_ok = False
_deepeval_ok = False

try:
   
    from trulens.core import Tru  # type: ignore
    _trulens_ok = True
except Exception as e:
    print(f"[EVAL] TruLens not available: {e}")

try:
    
    import deepeval  # noqa: F401
    _deepeval_ok = True
except Exception as e:
    print(f"[EVAL] DeepEval not available: {e}")

# ------------------ Helpers to infer reference facts from DF ------------------

_LABELS = {
    "temp":         ("temperature", "°C"),
    "temperature":  ("temperature", "°C"),
    "co2":          ("CO₂", "ppm"),
    "occupancy":    ("occupancy", "people"),
    "peoplemotion": ("motion", "counts"),
    "humidity":     ("humidity", "%"),
}

def _normalize_query(q: str) -> str:
    if not q:
        return ""
    q = q.lower()
    q = q.replace("co₂", "co2")
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _detect_target_stat(q_norm: str) -> Tuple[str, bool]:
    """
    Returns (target_col, ascending?) among {"min_val","avg_val","max_val"}.
    ascending? means the 'best' is the smallest (for min).
    """
    # Add utilisation synonyms and British/US spellings
    wants_min = any(k in q_norm for k in [
        "lowest", "min", "minimum", "coolest", "least",
        "less utilised", "less utilized", "less used",
        "underused", "under-utilised", "under-utilized",
        "low usage", "quietest"
    ])

    wants_max = any(k in q_norm for k in [
        "highest", "max", "maximum", "top", "worst", "hottest",
        "busiest", "most crowded", "peak"
    ])

    wants_avg = any(k in q_norm for k in ["average", "avg", "mean"])

    if wants_min:
        return "min_val", True
    if wants_max:
        return "max_val", False
    # default to average if unspecified
    return "avg_val", False


def _pick_metric_from_df(df: pd.DataFrame) -> str:
    if "metric_name" in df.columns and not df["metric_name"].isna().all():
        # use the most frequent non-null
        return str(df["metric_name"].dropna().astype(str).value_counts().idxmax()).lower()
    # fallback: try to guess from columns present
    for candidate in _LABELS.keys():
        if candidate in (c.lower() for c in df.columns):
            return candidate
    # last resort
    return "temperature"

def _compute_room_stats_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns: room, min_val, avg_val, max_val
    Works whether source has 'room' or 'display_name'.
    """
    room_col = "room" if "room" in df.columns else ("display_name" if "display_name" in df.columns else None)
    value_col = "value" if "value" in df.columns else None
    if room_col is None or value_col is None:
        # Try to detect numeric column if value missing.
        value_candidates = df.select_dtypes(include="number").columns.tolist()
        if value_candidates:
            value_col = value_candidates[0]
        else:
            raise ValueError("No numeric value column to evaluate.")
        # detect room-like column
        for c in df.columns:
            if str(c).lower() in ("room", "display_name", "space", "area"):
                room_col = c
                break
        if room_col is None:
            raise ValueError("No room/display_name column to evaluate.")

    stats = (
        df.groupby(room_col, as_index=False)[value_col]
          .agg(min_val="min", avg_val="mean", max_val="max")
          .round(2)
          .rename(columns={room_col: "room"})
    )
    return stats

def _reference_from_df_and_query(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    qn = _normalize_query(query)
    metric_key = _pick_metric_from_df(df)
    friendly, unit = _LABELS.get(metric_key, (metric_key.replace("_", " ").title(), ""))

    stats = _compute_room_stats_from_df(df)
    target_col, ascending = _detect_target_stat(qn)
    stats_sorted = stats.sort_values(by=target_col, ascending=ascending, ignore_index=True)
    best_row = stats_sorted.iloc[0]
    best_room = str(best_row["room"])
    best_val = float(best_row[target_col])

    # Collect room list for entity detection
    rooms = [str(x) for x in stats["room"].astype(str).unique().tolist()]

    # Time window presence 
    start = None
    end = None
    if "start_time" in df.columns:
        try:
            s = pd.to_datetime(df["start_time"])
            if len(s.dropna()) > 0:
                start = str(pd.to_datetime(s.min()).date())
                end = str(pd.to_datetime(s.max()).date())
        except Exception:
            pass

    return dict(
        metric_key=metric_key,
        friendly_label=friendly,
        unit=unit,
        target_col=target_col,
        ascending=ascending,
        best_room=best_room,
        best_val=best_val,
        rooms=rooms,
        stats=stats,       # for top3 fallback checks
        start=start,
        end=end
    )

# ------------------ Extract facts from model’s text ------------------

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _extract_number_near_unit(text: str, unit: str) -> Optional[float]:
    """
    Try to find a number; if a unit is given, prefer numbers close to that unit token.
    """
    t = text or ""
    # If unit present, search around it
    if unit:
        m = re.search(rf"({ _NUM_RE.pattern })\s*{re.escape(unit)}", t, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # Fallback: take a reasonable number (skip years like 1900..2100 if other numbers exist)
    nums = []
    for tok in _NUM_RE.findall(t):
        try:
            nums.append(float(tok))
        except Exception:
            pass
    if not nums:
        return None
    filtered = [n for n in nums if not (1900 <= n <= 2100)]
    return (filtered or nums)[0]

def _find_room_mentions(text: str, rooms: List[str]) -> Tuple[Optional[str], bool]:
    """
    Return (mentioned_room, is_top3_ok). We do exact substring match (case-insensitive).
    """
    if not text:
        return None, False
    low = text.lower()
    # Try bolded markdown first (often the chosen entity is bolded)
    bolded = re.findall(r"\*\*([^*]+)\*\*", text)
    bolded_lower = [b.lower() for b in bolded]
    for r in rooms:
        rlow = r.lower()
        if any(rlow in b for b in bolded_lower):
            return r, False
    # Else plain containment with word-boundary preference
    for r in rooms:
        rlow = r.lower()
        if re.search(rf"\b{re.escape(rlow)}\b", low):
            return r, False
    return None, False

def _detect_directionality(text: str) -> Optional[str]:
    """
    Returns 'min' / 'max' / 'avg' / None based on phrasing.
    """
    t = (text or "").lower()
    if any(k in t for k in ["lowest", "min", "minimum", "coolest", "least", "best"]):
        return "min"
    if any(k in t for k in ["highest", "max", "maximum", "top", "worst", "hottest"]):
        return "max"
    if any(k in t for k in ["average", "avg", "mean"]):
        return "avg"
    return None

def _mentions_time_window(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in t for k in ["now", "right now", "current", "currently"]):
        return True
    return any(k in t for k in [" to ", "→", "from", "between", "range", "window", "today", "yesterday"])

# ------------------ Parse machine-readable EVAL META ------------------

def _parse_eval_meta(text: str) -> Dict[str, Any]:
    """
    Parse blocks like:
      [EVAL_META]
      metric=temperature
      room=PES Seminar Room 20
      direction=max
      time_window=now
      value=25.070000
      [/EVAL_META]
    Returns empty dict if not present.
    """
    if not text:
        return {}
    m = re.search(r"\[EVAL_META\](.*?)\[/EVAL_META\]", text, flags=re.S | re.I)
    if not m:
        return {}
    body = m.group(1)

    meta: Dict[str, Any] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k:
            continue
        if k == "value":
            try:
                meta[k] = float(v)
            except Exception:
                meta[k] = None
        else:
            meta[k] = v
    return meta

# ------------------ Scoring (0–5) ------------------

def _relative_error(pred: float, ref: float) -> float:
    denom = max(1.0, abs(ref))
    return abs(pred - ref) / denom

def _score_numeric(pred_val: Optional[float], ref_val: float) -> float:
    """
    Returns 0..1 for numeric correctness (tolerance bands):
      <=2% → 1.0
      <=5% → 0.8
      <=10% → 0.6
      <=20% → 0.3
      else 0
    """
    if pred_val is None or not math.isfinite(ref_val):
        return 0.0
    r = _relative_error(pred_val, ref_val)
    if r <= 0.02: return 1.0
    if r <= 0.05: return 0.8
    if r <= 0.10: return 0.6
    if r <= 0.20: return 0.3
    return 0.0

def _score_entity(mentioned_room: Optional[str], ref_room: str, stats: pd.DataFrame, target_col: str, ascending: bool) -> float:
    """
    1.0 if mentioned room is among the tied 'best' set (min or max, depending on target_col).
    0.6 if within top-3 of the correct ranking.
    else 0.0
    """
    if not mentioned_room:
        return 0.0

    eps = 1e-9
    if ascending:
        best_val = float(stats[target_col].min())
        best_set = [str(r) for r in stats.loc[(stats[target_col] - best_val).abs() <= eps, "room"].tolist()]
    else:
        best_val = float(stats[target_col].max())
        best_set = [str(r) for r in stats.loc[(stats[target_col] - best_val).abs() <= eps, "room"].tolist()]

    if any(mentioned_room.lower() == r.lower() for r in best_set):
        return 1.0

    ordered = stats.sort_values(by=target_col, ascending=ascending, ignore_index=True)
    top3 = [str(x) for x in ordered["room"].head(3).tolist()]
    if any(mentioned_room.lower() == r.lower() for r in top3):
        return 0.6
    return 0.0


def _score_directionality(text_dir: Optional[str], target_col: str) -> float:
    """
    1.0 if the narrative direction matches ('max' with max_val, 'min' with min_val, 'avg' with avg_val).
    0.5 if text direction missing (don’t punish too hard).
    0.0 if it contradicts.
    """
    if text_dir is None:
        return 0.5
    if target_col == "max_val" and text_dir == "max":
        return 1.0
    if target_col == "min_val" and text_dir == "min":
        return 1.0
    if target_col == "avg_val" and text_dir == "avg":
        return 1.0
    return 0.0

def _score_timewindow(mentioned: bool) -> float:
    # Small bonus for mentioning time window explicitly
    return 1.0 if mentioned else 0.4

def _aggregate_to_5(entity: float, numeric: float, direction: float, timewin: float) -> float:
    """
    Weighted average → 0..5
      entity:     0.35
      numeric:    0.45
      direction:  0.10
      timewindow: 0.10
    """
    score01 = (
        0.35 * entity +
        0.45 * numeric +
        0.10 * direction +
        0.10 * timewin
    )
    return round(5.0 * score01, 2)

# ------------------ Public API ------------------

def log_and_score(
    user_query: str,
    df: pd.DataFrame,
    insight_text: str
) -> Optional[float]:
    """
    Scores factual correctness against the actual data in df (0..5).
    - Uses [EVAL_META] block if present for deterministic evaluation.
    - No OpenAI calls.
    - TruLens/DeepEval are OPTIONAL; if present, we report the same fact score through them.
    Returns the 0..5 score (float) or None on failure.
    """
    try:
        ref = _reference_from_df_and_query(df, user_query)
    except Exception as e:
        print(f"[EVAL] Could not build reference from data: {e}")
        return None

    # ---------- Prefer machine-readable meta, then fall back ----------
    meta = _parse_eval_meta(insight_text or "")

    # Entity (room)
    if meta.get("room"):
        mentioned_room = str(meta["room"])
    else:
        mentioned_room, _ = _find_room_mentions(insight_text or "", ref["rooms"])

    # Numeric value
    if "value" in meta and meta["value"] is not None:
        pred_val = float(meta["value"])
    else:
        pred_val = _extract_number_near_unit(insight_text or "", ref["unit"])

    # Direction
    if meta.get("direction") in {"min", "max", "avg"}:
        text_dir = meta["direction"]
    else:
        text_dir = _detect_directionality(insight_text or "")

    # Time window
    if "time_window" in meta:
        # Treat presence of a concrete time window in meta as a valid mention.
        time_mentioned = True
    else:
        time_mentioned = _mentions_time_window(insight_text or "")

    # Per-axis scores (0..1)
    s_entity = _score_entity(mentioned_room, ref["best_room"], ref["stats"], ref["target_col"], ref["ascending"])
    s_numeric = _score_numeric(pred_val, ref["best_val"])
    s_dir = _score_directionality(text_dir, ref["target_col"])
    s_time = _score_timewindow(time_mentioned)

    overall = _aggregate_to_5(s_entity, s_numeric, s_dir, s_time)

    print(f"[EVAL] Entity(Room): {s_entity:.2f}  "
          f"Numeric: {s_numeric:.2f}  "
          f"Direction: {s_dir:.2f}  "
          f"TimeWindow: {s_time:.2f}")
    print(f"[EVAL] Fact-based Accuracy (0–5): {overall:.2f}")

    # ---------- Optional: Report via TruLens / DeepEval without breaking ----------
    if _trulens_ok:
        try:
            tru = Tru()  # lightweight; if configured will log
            print(f"[EVAL] TruLens fact_score: {overall:.2f}")
        except Exception as e:
            print(f"[EVAL] TruLens logging skipped: {e}")

    if _deepeval_ok:
        try:
            print(f"[EVAL] DeepEval fact_score: {overall:.2f}")
        except Exception as e:
            print(f"[EVAL] DeepEval logging skipped: {e}")

    return overall
