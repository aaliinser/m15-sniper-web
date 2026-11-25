import os
import re
import json
import time
import datetime
import threading
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import requests
from requests.adapters import HTTPAdapter
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="M15 Sniper PRO")

# ==========================================
# 1. API Keys & Core Configuration
# ==========================================
TWELVE_DATA_KEYS = [
    "3db4b82facc84c5a84dd9abc262a5acb",
    "5d43d790ef114a7994876c0829e9fe4a",
    "5c929836e14d412182bbc5fce8f61b70"
]
FINNHUB_API_KEY = "d75bfl9r01qk56kc51dgd75bfl9r01qk56kc51e0"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash"
]

try:
    DISPLAY_TZ = ZoneInfo("Asia/Hebron")
except Exception:
    DISPLAY_TZ = datetime.timezone(datetime.timedelta(hours=2))

STRATEGY_NAMES = {
    "1": "1. M15 Sniper",
    "2": "2. V1_Original",
    "3": "3. S3_Calm",
    "4": "4. F4_ZScore",
    "5": "5. Stoch_9_5"
}

key_lock = threading.Lock()
cache_lock = threading.Lock()
autopilot_lock = threading.Lock()

current_key_index = 0
key_cooldowns = {0: 0.0, 1: 0.0, 2: 0.0}

AUTO_PILOT_DURATION_SECONDS = 6 * 3600
AUTO_PILOT_STATE = {
    "active": False,
    "expires_at": 0.0
}

CACHE_STORE = {}
CACHE_TTL_BY_TF = {
    "1M": 20,
    "5M": 45,
    "15M": 120,
    "1H": 120
}

thread_local = threading.local()

def get_http_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "M15-Sniper-PRO/6.3-SHIELD"})
        thread_local.session = s
    return thread_local.session

PAIRS_22 = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/NZD",
    "GBP/JPY", "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/NZD",
    "AUD/JPY", "AUD/CAD", "CAD/JPY", "CHF/JPY"
]

TIMEFRAME_MAP = {
    "1M": "1min",
    "5M": "5min",
    "15M": "15min",
    "1H": "1h"
}

TIMEFRAME_SECONDS = {
    "1M": 60,
    "5M": 300,
    "15M": 900,
    "1H": 3600
}

def normalize_symbol(raw_sym):
    sym = str(raw_sym or "EUR/USD").strip().upper()
    if "/" not in sym and len(sym) == 6:
        sym = f"{sym[:3]}/{sym[3:]}"
    return sym

def normalize_timeframe(raw_tf):
    tf = str(raw_tf or "15M").strip().upper()
    return tf if tf in TIMEFRAME_MAP else "15M"

def get_currencies(symbol):
    clean = str(symbol).replace("/", "").upper()
    return {clean[:3], clean[3:6]} if len(clean) >= 6 else set()

def normalize_strategy(raw_strat):
    s = str(raw_strat or "0").strip().lower()
    if s in ("0", "all", "multi") or "all" in s or "0." in s:
        return "0"
    if s in ("1", "sniper", "m15", "default") or "sniper" in s or "1." in s:
        return "1"
    if s in ("2", "v1", "v1_original", "calm") or "v1" in s or "2." in s:
        return "2"
    if s in ("3", "s3", "scalp") or "s3" in s or "3." in s:
        return "3"
    if s in ("4", "f4", "zscore") or "f4" in s or "4." in s:
        return "4"
    if s in ("5", "stoch", "stochastic", "h1") or "stoch" in s or "5." in s:
        return "5"
    return "0"

def get_price_decimals(symbol):
    return 3 if "JPY" in symbol else 5

def get_available_key_index(now_ts):
    global current_key_index
    with key_lock:
        num_keys = len(TWELVE_DATA_KEYS)
        for offset in range(num_keys):
            idx = (current_key_index + offset) % num_keys
            if now_ts >= key_cooldowns.get(idx, 0.0):
                current_key_index = (idx + 1) % num_keys
                return idx
        return None

def mark_key_cooldown(idx, seconds=55.0):
    with key_lock:
        key_cooldowns[idx] = time.time() + seconds

def cleanup_cache(now_ts):
    with cache_lock:
        expired = [k for k, (ts, ttl, _) in CACHE_STORE.items() if now_ts - ts >= ttl]
        for k in expired:
            CACHE_STORE.pop(k, None)

def check_autopilot_state():
    now_ts = time.time()
    with autopilot_lock:
        if AUTO_PILOT_STATE["active"] and now_ts >= AUTO_PILOT_STATE["expires_at"]:
            AUTO_PILOT_STATE["active"] = False
            AUTO_PILOT_STATE["expires_at"] = 0.0
        active = AUTO_PILOT_STATE["active"]
        remaining = max(0, int(AUTO_PILOT_STATE["expires_at"] - now_ts)) if active else 0
        return active, remaining

# ==========================================
# 2. Fetch Candles (UTC End-to-End)
# ==========================================
def fetch_candles(symbol, timeframe="15M", outputsize=300):
    symbol = normalize_symbol(symbol)
    tf_upper = normalize_timeframe(timeframe)
    interval = TIMEFRAME_MAP[tf_upper]
    tf_sec = TIMEFRAME_SECONDS[tf_upper]
    ttl = CACHE_TTL_BY_TF.get(tf_upper, 120)
    now = time.time()
    current_bucket = int(now // tf_sec)
    cache_key = f"{symbol}_{interval}_{outputsize}_{current_bucket}"

    cleanup_cache(now)

    with cache_lock:
        if cache_key in CACHE_STORE:
            cached_time, cached_ttl, cached_df = CACHE_STORE[cache_key]
            if now - cached_time < cached_ttl:
                return cached_df.copy(), None

    last_error = "تم الوصول للحد المؤقت للطلبات (24 طلب/دقيقة) — انتظر 45 ثانية ثم أعد المحاولة."
    num_keys = len(TWELVE_DATA_KEYS)
    session = get_http_session()
    min_needed = 60 if outputsize < 200 else 205

    for _ in range(num_keys):
        idx = get_available_key_index(time.time())
        if idx is None:
            break

        api_key = TWELVE_DATA_KEYS[idx]
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}&interval={interval}&outputsize={outputsize}"
            f"&apikey={api_key}&timezone=UTC"
        )
        try:
            resp = session.get(url, timeout=7)
            if resp.status_code == 429:
                mark_key_cooldown(idx, 55.0)
                continue

            data = resp.json()
            if not isinstance(data, dict):
                continue

            if data.get("status") == "error":
                code = data.get("code", 0)
                msg = str(data.get("message", ""))
                if code == 429 or "run out of API credits" in msg:
                    mark_key_cooldown(idx, 55.0)
                    continue
                return None, f"تنبيه من مزود البيانات: {msg or 'خطأ غير معروف'}"

            values = data.get("values")
            if not values or len(values) < min_needed:
                return None, "بيانات الشموع غير كافية لحساب المؤشرات بدقة."

            df = pd.DataFrame(values)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["open_dt_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "open_dt_utc"])

            if len(df) < min_needed:
                return None, "البيانات الرقمية للشموع غير صالحة."

            df["open_ts"] = [int(ts.timestamp()) for ts in df["open_dt_utc"]]
            df = df.drop_duplicates(subset=["open_ts"]).sort_values("open_ts").reset_index(drop=True)

            if len(df) < min_needed:
                return None, "بيانات الشموع غير كافية بعد إزالة التكرار."

            with cache_lock:
                CACHE_STORE[cache_key] = (time.time(), ttl, df.copy())

            return df, None

        except Exception:
            last_error = "انقطع الاتصال المؤقت بالسيرفر — أعد المحاولة بعد ثوانٍ."

    return None, last_error

def filter_closed_candles(df, timeframe):
    if df is None or len(df) == 0:
        return pd.DataFrame(), 0.0
    tf_sec = TIMEFRAME_SECONDS.get(normalize_timeframe(timeframe), 900)
    now_ts = time.time()
    live_price = float(df["close"].iloc[-1])
    closed_df = df[(df["open_ts"] + tf_sec) <= now_ts].copy().reset_index(drop=True)
    return closed_df, live_price

# ==========================================
# 3. Audited Indicators
# ==========================================
def calculate_indicators(closed_df):
    df = closed_df.copy()
    if len(df) == 0:
        for col in [
            "bb_mid", "bb_upper", "bb_lower", "bb20_upper", "bb20_lower",
            "zscore", "ema200", "rsi", "rsi7", "atr", "adx", "stoch_k", "stoch_d"
        ]:
            df[col] = np.nan
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]

    sma15 = close.rolling(window=15, min_periods=15).mean()
    std15 = close.rolling(window=15, min_periods=15).std(ddof=0)
    df["bb_mid"] = sma15
    df["bb_upper"] = sma15 + (2.0 * std15)
    df["bb_lower"] = sma15 - (2.0 * std15)

    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    df["bb20_upper"] = sma20 + (2.0 * std20)
    df["bb20_lower"] = sma20 - (2.0 * std20)
    df["zscore"] = (close - sma20) / std20.replace(0.0, np.nan)

    df["ema200"] = close.ewm(span=200, min_periods=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))

    avg_gain14 = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss14 = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs14 = avg_gain14 / avg_loss14.replace(0.0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs14))
    df.loc[(avg_loss14 == 0.0) & (avg_gain14 > 0.0), "rsi"] = 100.0
    df.loc[(avg_loss14 == 0.0) & (avg_gain14 == 0.0), "rsi"] = 50.0

    avg_gain7 = gain.ewm(alpha=1.0 / 7.0, min_periods=7, adjust=False).mean()
    avg_loss7 = loss.ewm(alpha=1.0 / 7.0, min_periods=7, adjust=False).mean()
    rs7 = avg_gain7 / avg_loss7.replace(0.0, np.nan)
    df["rsi7"] = 100.0 - (100.0 / (1.0 + rs7))
    df.loc[(avg_loss7 == 0.0) & (avg_gain7 > 0.0), "rsi7"] = 100.0
    df.loc[(avg_loss7 == 0.0) & (avg_gain7 == 0.0), "rsi7"] = 50.0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    df["atr"] = atr

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr_safe
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr_safe
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    df["adx"] = dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    lowest_low = low.rolling(window=9, min_periods=9).min()
    highest_high = high.rolling(window=9, min_periods=9).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    df["stoch_k"] = 100.0 * ((close - lowest_low) / denom)
    df["stoch_d"] = df["stoch_k"].rolling(window=5, min_periods=5).mean()

    return df

def detect_sr_and_fib(closed_df, symbol):
    if closed_df is None or len(closed_df) == 0:
        return [], [], {}

    decimals = get_price_decimals(symbol)
    recent = closed_df.tail(120)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    diff = max(swing_high - swing_low, 1e-6)

    fib_levels = {
        "0.0": round(swing_high, decimals),
        "0.236": round(swing_high - 0.236 * diff, decimals),
        "0.382": round(swing_high - 0.382 * diff, decimals),
        "0.5": round(swing_high - 0.5 * diff, decimals),
        "0.618": round(swing_high - 0.618 * diff, decimals),
        "0.786": round(swing_high - 0.786 * diff, decimals),
        "1.0": round(swing_low, decimals),
    }

    atr_raw = closed_df["atr"].iloc[-1] if "atr" in closed_df.columns else np.nan
    atr_val = float(atr_raw) if not pd.isna(atr_raw) else (diff * 0.02)
    min_tol = 0.015 if "JPY" in symbol else 0.00015
    tol = max(atr_val * 0.35, min_tol)

    candidates = list(recent["high"].values) + list(recent["low"].values)
    levels = []
    for price in candidates:
        matched = False
        for lvl in levels:
            if abs(lvl["price"] - price) <= tol:
                lvl["touches"] += 1
                matched = True
                break
        if not matched:
            levels.append({"price": float(price), "touches": 1})

    current_close = float(closed_df["close"].iloc[-1])
    resistances = sorted([l for l in levels if l["price"] > current_close and l["touches"] >= 3], key=lambda x: x["price"])[:2]
    supports = sorted([l for l in levels if l["price"] <= current_close and l["touches"] >= 3], key=lambda x: -x["price"])[:2]

    return resistances, supports, fib_levels

# ==========================================
# 4. AI Agents Engine (Fast Multi-Model Fallback)
# ==========================================
def call_gemini_json(prompt_text, timeout_sec=8):
    if not GEMINI_API_KEY:
        return {"_error": "مفتاح GEMINI_API_KEY غير مضاف في Render"}

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    session = get_http_session()
    last_error = "تأخر استجابة سيرفرات Gemini"

    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        gen_config = {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
        if "2.5" in model or "latest" in model:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": gen_config
        }
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                last_error = "رد غير متوقع من Gemini"
                continue
            last_error = f"HTTP {resp.status_code} ({model})"
        except Exception as e:
            last_error = f"{model}: {str(e)[:90]}"

    return {"_error": last_error}

def run_agent_court(symbol, timeframe, closed_df, analysis):
    """Agent 1: محكمة الصفقة (Sniper vs Trap Hunter -> Judge)"""
    if analysis["signal"] not in ("CALL", "PUT"):
        return None

    last_candles = []
    for _, r in closed_df.tail(8).iterrows():
        last_candles.append({
            "o": round(float(r["open"]), 5),
            "h": round(float(r["high"]), 5),
            "l": round(float(r["low"]), 5),
            "c": round(float(r["close"]), 5)
        })

    prompt = (
        f"أنت نظام محكمة تداول كمي من 3 وكلاء لصفقة خيارات ثنائية على {symbol} فريم {timeframe}.\n"
        f"الإشارة المقترحة: {analysis['signal']} | الاستراتيجية: {analysis.get('strategy_label', '')} | السبب الفني: {analysis['reason']}\n"
        f"السعر الحالي: {analysis['live_price']} | EMA200: {analysis.get('ema200')} | "
        f"BB Upper: {analysis.get('bb_upper')} | BB Lower: {analysis.get('bb_lower')}\n"
        f"أقرب مقاومات: {analysis.get('resistances')} | أقرب دعوم: {analysis.get('supports')}\n"
        f"آخر 8 شموع مغلقة: {json.dumps(last_candles)}\n\n"
        "أرجع JSON فقط بهذه المفاتيح بالضبط وباللغة العربية المختصرة:\n"
        '{"sniper": "مرافعة القناص المؤيدة في جملة قصيرة", '
        '"trap_hunter": "مرافعة صائد الفخوخ المحذرة في جملة قصيرة", '
        '"verdict": "A+ APPROVED أو HALF RISK أو VETO / TRAP", '
        '"judge_note": "قرار القاضي النهائي في جملة واحدة"}'
    )

    res = call_gemini_json(prompt, timeout_sec=8)
    if not isinstance(res, dict) or "verdict" not in res:
        err_detail = res.get("_error", "تأخر استجابة الـ API") if isinstance(res, dict) else "تأخر استجابة الـ API"
        return {
            "verdict": "HALF RISK",
            "sniper": "الإشارة مطابقة لشروط الاستراتيجية الكمية.",
            "trap_hunter": f"تعذر الاتصال بـ Gemini ({err_detail})",
            "judge_note": "حكم احتياطي تلقائي."
        }
    return {
        "verdict": str(res.get("verdict", "HALF RISK")).upper(),
        "sniper": str(res.get("sniper", "")),
        "trap_hunter": str(res.get("trap_hunter", "")),
        "judge_note": str(res.get("judge_note", ""))
    }

def run_agent_mtf_guard(symbol, timeframe, closed_df, analysis):
    """Agent 2: حارس فريم الساعة والأرقام النفسية (MTF Liquidity & Round-Number Guard)"""
    if analysis["signal"] not in ("CALL", "PUT"):
        return None

    price = float(analysis["live_price"])
    is_jpy = "JPY" in symbol
    decimals = 3 if is_jpy else 5
    pip_size = 0.01 if is_jpy else 0.0001
    step = 0.50 if is_jpy else 0.0050

    nearest_round = round(round(price / step) * step, decimals)
    dist_round_pips = round(abs(price - nearest_round) / pip_size, 1)

    direction = analysis["signal"]
    round_ahead = (direction == "CALL" and nearest_round > price) or (direction == "PUT" and nearest_round < price)

    h1_wall_info = "طريق فريم الساعة (1H) مفتوح بدون جدار قريب"
    h1_wall_blocked = False

    res_1h, sup_1h = [], []
    if timeframe == "1H":
        res_1h = analysis.get("resistances") or []
        sup_1h = analysis.get("supports") or []
    else:
        h1_df, err = fetch_candles(symbol, "1H", outputsize=120)
        if not err and h1_df is not None and len(h1_df) >= 30:
            h1_closed, _ = filter_closed_candles(h1_df, "1H")
            if len(h1_closed) >= 20:
                res_1h, sup_1h, _ = detect_sr_and_fib(h1_closed, symbol)
        else:
            h1_wall_info = "لم يتم فحص جدار الساعة مؤقتاً"

    if direction == "CALL" and res_1h:
        wall = res_1h[0]["price"]
        dist_pips = round((wall - price) / pip_size, 1)
        h1_wall_info = f"مقاومة 1H عند {wall} (تبعد {dist_pips} نقطة)"
        if dist_pips <= 6.0:
            h1_wall_blocked = True
    elif direction == "PUT" and sup_1h:
        wall = sup_1h[0]["price"]
        dist_pips = round((price - wall) / pip_size, 1)
        h1_wall_info = f"دعم 1H عند {wall} (يبعد {dist_pips} نقطة)"
        if dist_pips <= 6.0:
            h1_wall_blocked = True

    if h1_wall_blocked or (round_ahead and dist_round_pips <= 4.0):
        status = "BLOCKED"
        summary = f"⛔ جدار قريب أمام السعر! رقم نفسي {nearest_round} (يبعد {dist_round_pips} نقطة) | {h1_wall_info}"
    elif round_ahead and dist_round_pips <= 10.0:
        status = "CAUTION"
        summary = f"⚠️ انتباه: رقم نفسي {nearest_round} أمام السعر ({dist_round_pips} نقطة) | {h1_wall_info}"
    else:
        status = "CLEAR"
        summary = f"✅ الطريق مفتوح (إشارة ذهبية) | أقرب رقم نفسي {nearest_round} ({dist_round_pips} نقطة) | {h1_wall_info}"

    return {
        "status": status,
        "nearest_round": nearest_round,
        "dist_round_pips": dist_round_pips,
        "h1_wall": h1_wall_info,
        "summary": summary
    }

def run_agent_correlation_assassin(signals_list):
    """Agent 3: صائد الارتباطات ومنع الفخ المزدوج (Correlation & Exposure Assassin)"""
    if len(signals_list) <= 1:
        return signals_list, []

    ordered = sorted(
        signals_list,
        key=lambda x: (x.get("confluence", 1), x.get("strength", 0.0)),
        reverse=True
    )

    if GEMINI_API_KEY and len(ordered) >= 2:
        brief = [
            {
                "symbol": s["symbol"],
                "signal": s["signal"],
                "confidence": s["confidence"],
                "confluence": s.get("confluence", 1),
                "strategy": s.get("strategy_label", ""),
                "reason": s["reason"]
            }
            for s in ordered
        ]
        prompt = (
            "أنت وكيل إدارة مخاطر وارتباطات فوركس. هذه إشارات ظهرت في نفس الشمعة:\n"
            f"{json.dumps(brief, ensure_ascii=False)}\n"
            "اختر أفضل الأزواج التي لا تشترك في نفس العملة (ممنوع تكرار USD أو EUR أو GBP أو JPY في أكثر من صفقة)، "
            "وارجع JSON فقط بالشكل:\n"
            '{"keep_symbols": ["EUR/USD"], "note": "سبب الاختيار في جملة قصيرة بالعربية"}'
        )
        ai_res = call_gemini_json(prompt, timeout_sec=7)
        if isinstance(ai_res, dict) and isinstance(ai_res.get("keep_symbols"), list):
            ai_keep = {normalize_symbol(sym) for sym in ai_res["keep_symbols"] if isinstance(sym, str) and sym.strip()}
            ai_note = str(ai_res.get("note", "تمت التصفية لمنع تداخل العملات."))
            kept, killed, used_currencies = [], [], set()
            for item in ordered:
                currs = get_currencies(item["symbol"])
                if item["symbol"] in ai_keep and not (currs & used_currencies):
                    used_currencies |= currs
                    item["corr_note"] = ai_note
                    kept.append(item)
                else:
                    item["killed_reason"] = f"تم استبعاده لتداخل العملات ({ai_note})"
                    killed.append(item)
            if kept:
                return kept, killed

    kept, killed, used_currencies = [], [], set()
    for item in ordered:
        currs = get_currencies(item["symbol"])
        if currs & used_currencies:
            item["killed_reason"] = "تداخل عملة مع إشارة أقوى في نفس الشمعة"
            killed.append(item)
        else:
            used_currencies |= currs
            kept.append(item)

    return kept, killed

# ==========================================
# 5. Strategy Evaluation (Single + All 5)
# ==========================================
def evaluate_single_strategy(closed_df, live_price, symbol, strategy_id):
    decimals = get_price_decimals(symbol)
    min_req = 201 if strategy_id in ("1", "5") else 35
    strat_name = STRATEGY_NAMES.get(strategy_id, f"استراتيجية {strategy_id}")

    if len(closed_df) < min_req:
        return {
            "signal": "WAIT",
            "reason": "بيانات الشموع المغلقة غير كافية لحساب المؤشرات.",
            "confidence": 0,
            "strength": 0.0,
            "confluence": 0,
            "strategy_label": strat_name,
            "live_price": round(live_price, decimals),
            "bb_upper": None, "bb_lower": None, "ema200": None,
            "resistances": [], "supports": [], "fib": {}
        }

    c = closed_df.iloc[-1]
    prev = closed_df.iloc[-2]

    required_cols = []
    if strategy_id == "1":
        required_cols = ["bb_upper", "bb_lower", "ema200", "rsi", "atr"]
    elif strategy_id == "2":
        required_cols = ["bb20_upper", "bb20_lower", "rsi7", "adx", "atr"]
    elif strategy_id == "3":
        required_cols = ["bb_upper", "bb_lower", "rsi", "adx", "atr"]
    elif strategy_id == "4":
        required_cols = ["zscore", "rsi", "adx", "atr"]
    elif strategy_id == "5":
        required_cols = ["ema200", "stoch_k", "stoch_d"]

    if any(pd.isna(c[col]) for col in required_cols) or (
        strategy_id == "5" and (pd.isna(prev["stoch_k"]) or pd.isna(prev["stoch_d"]))
    ):
        return {
            "signal": "WAIT",
            "reason": "مؤشرات التهيئة غير مكتملة (NaN) — تم تخطي التقييم.",
            "confidence": 0,
            "strength": 0.0,
            "confluence": 0,
            "strategy_label": strat_name,
            "live_price": round(live_price, decimals),
            "bb_upper": None, "bb_lower": None, "ema200": None,
            "resistances": [], "supports": [], "fib": {}
        }

    close_p = float(c["close"])
    open_p = float(c["open"])
    high_p = float(c["high"])
    low_p = float(c["low"])

    bb_u = float(c["bb_upper"]) if not pd.isna(c["bb_upper"]) else None
    bb_l = float(c["bb_lower"]) if not pd.isna(c["bb_lower"]) else None
    ema200 = float(c["ema200"]) if not pd.isna(c["ema200"]) else None
    rsi = float(c["rsi"]) if not pd.isna(c["rsi"]) else 50.0
    atr = float(c["atr"]) if not pd.isna(c["atr"]) else 0.0

    resistances, supports, fib_levels = detect_sr_and_fib(closed_df, symbol)

    signal = "WAIT"
    reason = "بانتظار إشارة مطابقة للشروط"
    confidence = 0
    strength = 0.0

    # الاستراتيجية 1 (الأساسية) — بدون أي تعديل إطلاقاً
    if strategy_id == "1":
        candle_range = max(high_p - low_p, 1e-9)
        body_size = abs(close_p - open_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)
        not_blowout = (atr <= 0.0) or (body_size <= 2.2 * atr)

        if close_p > ema200 and close_p <= bb_l and rsi <= 42.0 and not_blowout:
            signal = "CALL"
            reason = f"[A+] إغلاق مؤكد <= البولينجر السفلي وفوق EMA 200 (RSI={rsi:.1f})"
            confidence = 62
            strength = round(abs(50.0 - rsi), 2)
        elif close_p < ema200 and close_p >= bb_u and rsi >= 58.0 and not_blowout:
            signal = "PUT"
            reason = f"[A+] إغلاق مؤكد >= البولينجر العلوي وتحت EMA 200 (RSI={rsi:.1f})"
            confidence = 62
            strength = round(abs(50.0 - rsi), 2)
        elif close_p > ema200 and low_p <= bb_l and (lower_wick / candle_range) >= 0.35 and rsi <= 42.0 and not_blowout:
            signal = "CALL"
            reason = f"[A] ذيل رفض سفلي قوي عند البولينجر السفلي وفوق EMA 200 (RSI={rsi:.1f})"
            confidence = 58
            strength = round(abs(50.0 - rsi), 2)
        elif close_p < ema200 and high_p >= bb_u and (upper_wick / candle_range) >= 0.35 and rsi >= 58.0 and not_blowout:
            signal = "PUT"
            reason = f"[A] ذيل رفض علوي قوي عند البولينجر العلوي وتحت EMA 200 (RSI={rsi:.1f})"
            confidence = 58
            strength = round(abs(50.0 - rsi), 2)
        else:
            reason = "بانتظار إغلاق أو ذيل رفض عند BB(15, 2.0) مع توافق EMA 200"

    # الاستراتيجية 2 (24 ساعة + مرنة + درع الحماية الثلاثي)
    elif strategy_id == "2":
        bb20_u = float(c["bb20_upper"])
        bb20_l = float(c["bb20_lower"])
        rsi7 = float(c["rsi7"])
        adx = float(c["adx"])
        adx_prev = float(prev["adx"]) if not pd.isna(prev["adx"]) else adx
        bb_u, bb_l = bb20_u, bb20_l

        candle_range = max(high_p - low_p, 1e-9)
        body_size = abs(close_p - open_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)

        adx_rising_fast = (adx - adx_prev) > 1.5
        is_rocket = (atr > 0.0) and (body_size > 1.5 * atr)
        has_call_wick = (lower_wick / candle_range) >= 0.25
        has_put_wick = (upper_wick / candle_range) >= 0.25

        if "JPY" in symbol:
            reason = "استراتيجية V1_Original تستبعد أزواج الين (JPY)."
        elif adx >= 32.0:
            reason = f"السوق في ترند قوي (ADX={adx:.1f} >= 32) — تم تخطي الدخول."
        elif adx_rising_fast:
            reason = f"🛡️ درع الحماية: تسارع ترند مفاجئ (ADX صعد من {adx_prev:.1f} إلى {adx:.1f})."
        elif is_rocket:
            reason = "🛡️ درع الحماية: شمعة اندفاع صاروخي ضخمة — ممنوع الدخول عكسها."
        elif close_p < open_p and close_p <= bb20_l and rsi7 <= 28.0 and has_call_wick:
            signal = "CALL"
            reason = f"[V1_Original | 🛡️ درع مفعّل] كسر BB(20,2) السفلي + ذيل رفض + RSI7={rsi7:.1f} + ADX={adx:.1f}"
            confidence = 68
            strength = round(abs(50.0 - rsi7), 2)
        elif close_p > open_p and close_p >= bb20_u and rsi7 >= 72.0 and has_put_wick:
            signal = "PUT"
            reason = f"[V1_Original | 🛡️ درع مفعّل] كسر BB(20,2) العلوي + ذيل رفض + RSI7={rsi7:.1f} + ADX={adx:.1f}"
            confidence = 68
            strength = round(abs(50.0 - rsi7), 2)
        else:
            reason = f"مراقبة V1_Original (RSI7={rsi7:.1f} | ADX={adx:.1f})"

    # الاستراتيجية 3 (24 ساعة + مرنة + درع الحماية الثلاثي)
    elif strategy_id == "3":
        adx = float(c["adx"])
        adx_prev = float(prev["adx"]) if not pd.isna(prev["adx"]) else adx

        candle_range = max(high_p - low_p, 1e-9)
        body_size = abs(close_p - open_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)

        adx_rising_fast = (adx - adx_prev) > 1.5
        is_rocket = (atr > 0.0) and (body_size > 1.5 * atr)
        has_call_wick = (lower_wick / candle_range) >= 0.25
        has_put_wick = (upper_wick / candle_range) >= 0.25

        if "JPY" in symbol:
            reason = "استراتيجية S3_Calm تستبعد أزواج الين (JPY)."
        elif adx >= 32.0:
            reason = f"السوق في ترند قوي (ADX={adx:.1f} >= 32) — تم تخطي الدخول."
        elif adx_rising_fast:
            reason = f"🛡️ درع الحماية: تسارع ترند مفاجئ (ADX صعد من {adx_prev:.1f} إلى {adx:.1f})."
        elif is_rocket:
            reason = "🛡️ درع الحماية: شمعة اندفاع صاروخي ضخمة — ممنوع الدخول عكسها."
        elif rsi <= 34.0 and close_p <= bb_l and has_call_wick:
            signal = "CALL"
            reason = f"[S3 | 🛡️ درع مفعّل] تشبع بيعي RSI14={rsi:.1f} <= BB(15,2) + ذيل رفض + ADX={adx:.1f}"
            confidence = 66
            strength = round(abs(50.0 - rsi), 2)
        elif rsi >= 66.0 and close_p >= bb_u and has_put_wick:
            signal = "PUT"
            reason = f"[S3 | 🛡️ درع مفعّل] تشبع شرائي RSI14={rsi:.1f} >= BB(15,2) + ذيل رفض + ADX={adx:.1f}"
            confidence = 66
            strength = round(abs(50.0 - rsi), 2)
        else:
            reason = f"مراقبة S3_Calm (RSI14={rsi:.1f} | ADX={adx:.1f})"

    # الاستراتيجية 4 (24 ساعة + مرنة + درع الحماية الثلاثي)
    elif strategy_id == "4":
        zscore = float(c["zscore"])
        adx = float(c["adx"])
        adx_prev = float(prev["adx"]) if not pd.isna(prev["adx"]) else adx

        candle_range = max(high_p - low_p, 1e-9)
        body_size = abs(close_p - open_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)

        adx_rising_fast = (adx - adx_prev) > 1.5
        is_rocket = (atr > 0.0) and (body_size > 1.5 * atr)
        has_call_wick = (lower_wick / candle_range) >= 0.25
        has_put_wick = (upper_wick / candle_range) >= 0.25

        if adx >= 32.0:
            reason = f"السوق في ترند قوي (ADX={adx:.1f} >= 32) — تم تخطي الدخول."
        elif adx_rising_fast:
            reason = f"🛡️ درع الحماية: تسارع ترند مفاجئ (ADX صعد من {adx_prev:.1f} إلى {adx:.1f})."
        elif is_rocket:
            reason = "🛡️ درع الحماية: شمعة اندفاع صاروخي ضخمة — ممنوع الدخول عكسها."
        elif close_p < open_p and zscore <= -2.0 and rsi <= 32.0 and has_call_wick:
            signal = "CALL"
            reason = f"[F4 | 🛡️ درع مفعّل] انحراف Z-Score={zscore:.2f} + ذيل رفض + RSI14={rsi:.1f} + ADX={adx:.1f}"
            confidence = 67
            strength = round(abs(zscore) * 10.0, 2)
        elif close_p > open_p and zscore >= 2.0 and rsi >= 68.0 and has_put_wick:
            signal = "PUT"
            reason = f"[F4 | 🛡️ درع مفعّل] انحراف Z-Score={zscore:.2f} + ذيل رفض + RSI14={rsi:.1f} + ADX={adx:.1f}"
            confidence = 67
            strength = round(abs(zscore) * 10.0, 2)
        else:
            reason = f"مراقبة F4 (Z-Score={zscore:.2f} | RSI14={rsi:.1f} | ADX={adx:.1f})"

    # الاستراتيجية 5 (استراتيجية الساعة Stoch) — بدون أي تعديل إطلاقاً
    elif strategy_id == "5":
        k_curr = float(c["stoch_k"])
        d_curr = float(c["stoch_d"])
        k_prev = float(prev["stoch_k"])
        d_prev = float(prev["stoch_d"])

        if close_p > ema200 and k_prev <= d_prev and k_curr > d_curr and k_curr < 10.0:
            signal = "CALL"
            reason = f"تقاطع صاعد Stoch(9,5) تحت 10 (K={k_curr:.1f}, D={d_curr:.1f}) فوق EMA 200"
            confidence = 60
            strength = round(100.0 - k_curr, 2)
        elif close_p < ema200 and k_prev >= d_prev and k_curr < d_curr and k_curr > 90.0:
            signal = "PUT"
            reason = f"تقاطع هابط Stoch(9,5) فوق 90 (K={k_curr:.1f}, D={d_curr:.1f}) تحت EMA 200"
            confidence = 60
            strength = round(k_curr, 2)
        else:
            reason = f"مراقبة Stoch(9,5) (K={k_curr:.1f}, D={d_curr:.1f}) مع توافق EMA 200"

    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "strength": strength,
        "confluence": 1 if signal in ("CALL", "PUT") else 0,
        "strategy_label": strat_name,
        "live_price": round(live_price, decimals),
        "bb_upper": round(bb_u, decimals) if bb_u is not None else None,
        "bb_lower": round(bb_l, decimals) if bb_l is not None else None,
        "ema200": round(ema200, decimals) if ema200 is not None else None,
        "resistances": [{"price": round(r["price"], decimals), "touches": r["touches"]} for r in resistances],
        "supports": [{"price": round(s["price"], decimals), "touches": s["touches"]} for s in supports],
        "fib": fib_levels
    }

def evaluate_strategy(closed_df, live_price, symbol, strategy_id):
    strategy_id = normalize_strategy(strategy_id)
    if strategy_id != "0":
        return evaluate_single_strategy(closed_df, live_price, symbol, strategy_id)

    results = {}
    calls = []
    puts = []
    for sid in ("1", "2", "3", "4", "5"):
        res = evaluate_single_strategy(closed_df, live_price, symbol, sid)
        results[sid] = res
        if res["signal"] == "CALL":
            calls.append((sid, res))
        elif res["signal"] == "PUT":
            puts.append((sid, res))

    base = results["1"]

    if calls and puts:
        base["signal"] = "WAIT"
        base["confidence"] = 0
        base["strength"] = 0.0
        base["confluence"] = 0
        base["strategy_label"] = "تضارب استراتيجيات"
        base["reason"] = "تم استبعاد الزوج لتضارب إشارات الصعود والهبوط بين الاستراتيجيات في نفس الشمعة."
        return base

    winners = calls if calls else puts
    if not winners:
        base["signal"] = "WAIT"
        base["confidence"] = 0
        base["strength"] = 0.0
        base["confluence"] = 0
        base["strategy_label"] = "فحص شامل (5 استراتيجيات)"
        base["reason"] = "لا توجد إشارة مطابقة في أي من الاستراتيجيات الخمس على هذا الفريم."
        return base

    winners.sort(key=lambda item: item[1]["strength"], reverse=True)
    primary = winners[0][1].copy()
    matched_names = [STRATEGY_NAMES[sid] for sid, _ in winners]
    confluence_count = len(winners)

    primary["confluence"] = confluence_count
    primary["strategy_label"] = " + ".join(matched_names)
    primary["confidence"] = min(90, max(r["confidence"] for _, r in winners) + (confluence_count - 1) * 6)
    primary["strength"] = round(primary["strength"] + (confluence_count - 1) * 20.0, 2)
    if confluence_count > 1:
        primary["reason"] = f"🔥 تأكيد مزدوج ({confluence_count} استراتيجيات: {primary['strategy_label']}) | {primary['reason']}"
    else:
        primary["reason"] = f"[{primary['strategy_label']}] {primary['reason']}"

    return primary

def scan_single_pair(sym, timeframe, strategy):
    df, err = fetch_candles(sym, timeframe)
    if err or df is None:
        return None
    closed_df, live_price = filter_closed_candles(df, timeframe)
    closed_df = calculate_indicators(closed_df)
    res = evaluate_strategy(closed_df, live_price, sym, strategy)
    return sym, res, closed_df

# ==========================================
# 6. Autopilot Routes
# ==========================================
@app.api_route("/api/autopilot/start", methods=["GET", "POST"])
@app.api_route("/autopilot/start", methods=["GET", "POST"])
def api_autopilot_start():
    now_ts = time.time()
    with autopilot_lock:
        AUTO_PILOT_STATE["active"] = True
        AUTO_PILOT_STATE["expires_at"] = now_ts + AUTO_PILOT_DURATION_SECONDS
    return JSONResponse({
        "ok": True,
        "active": True,
        "remaining_seconds": AUTO_PILOT_DURATION_SECONDS
    })

@app.api_route("/api/autopilot/stop", methods=["GET", "POST"])
@app.api_route("/autopilot/stop", methods=["GET", "POST"])
def api_autopilot_stop():
    with autopilot_lock:
        AUTO_PILOT_STATE["active"] = False
        AUTO_PILOT_STATE["expires_at"] = 0.0
    return JSONResponse({
        "ok": True,
        "active": False,
        "remaining_seconds": 0
    })

@app.get("/api/autopilot/status")
@app.get("/autopilot/status")
def api_autopilot_status():
    active, remaining = check_autopilot_state()
    return JSONResponse({
        "ok": True,
        "active": active,
        "remaining_seconds": remaining
    })

# ==========================================
# 7. Main API & Frontend Routes
# ==========================================
@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/api/analyze")
@app.get("/analyze")
def api_analyze(
    symbol: str = Query(default="EUR/USD"),
    timeframe: str = Query(default="15M"),
    strategy: str = Query(default="0"),
    agent_court: int = Query(default=0),
    agent_mtf: int = Query(default=0)
):
    sym = normalize_symbol(symbol)
    tf = normalize_timeframe(timeframe)
    strat_id = normalize_strategy(strategy)
    decimals = get_price_decimals(sym)

    df, err = fetch_candles(sym, tf)
    if err:
        return JSONResponse({"ok": False, "error": err})

    closed_df, live_price = filter_closed_candles(df, tf)
    closed_df = calculate_indicators(closed_df)
    analysis = evaluate_strategy(closed_df, live_price, sym, strat_id)

    agents_report = {}
    if analysis["signal"] in ("CALL", "PUT"):
        if agent_mtf == 1:
            agents_report["mtf"] = run_agent_mtf_guard(sym, tf, closed_df, analysis)
        if agent_court == 1:
            agents_report["court"] = run_agent_court(sym, tf, closed_df, analysis)

    analysis["agents"] = agents_report

    bb_u_col = "bb20_upper" if strat_id == "2" else "bb_upper"
    bb_l_col = "bb20_lower" if strat_id == "2" else "bb_lower"

    chart_df = closed_df.tail(90).copy()
    candles = []
    for _, row in chart_df.iterrows():
        local_dt = row["open_dt_utc"].to_pydatetime().astimezone(DISPLAY_TZ)
        candles.append({
            "time": local_dt.strftime("%H:%M"),
            "open": round(float(row["open"]), decimals),
            "high": round(float(row["high"]), decimals),
            "low": round(float(row["low"]), decimals),
            "close": round(float(row["close"]), decimals),
            "bb_upper": round(float(row[bb_u_col]), decimals) if not pd.isna(row[bb_u_col]) else None,
            "bb_lower": round(float(row[bb_l_col]), decimals) if not pd.isna(row[bb_l_col]) else None,
            "ema200": round(float(row["ema200"]), decimals) if not pd.isna(row["ema200"]) else None,
        })

    return JSONResponse({
        "ok": True,
        "symbol": sym,
        "timeframe": tf,
        "finnhub_key": FINNHUB_API_KEY,
        "analysis": analysis,
        "candles": candles
    })

@app.get("/api/scan")
@app.get("/scan")
def api_scan(
    timeframe: str = Query(default="15M"),
    strategy: str = Query(default="0"),
    mode: str = Query(default="manual"),
    agent_court: int = Query(default=0),
    agent_mtf: int = Query(default=0),
    agent_corr: int = Query(default=0)
):
    tf = normalize_timeframe(timeframe)
    m = str(mode).strip().lower()
    strat_id = normalize_strategy(strategy)

    if m == "auto":
        active, _ = check_autopilot_state()
        if not active:
            return JSONResponse({
                "ok": False,
                "auto_stopped": True,
                "error": "توقف الرادار التلقائي بعد انتهاء مدة الـ 6 ساعات أو تم إيقافه يدوياً."
            })

    target_pairs = list(PAIRS_22)
    raw_confirmed = []
    pairs_summary = []
    closed_map = {}
    scanned_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_single_pair, sym, tf, strat_id): sym for sym in target_pairs}
        try:
            for future in as_completed(futures, timeout=25):
                try:
                    outcome = future.result(timeout=1)
                except Exception:
                    continue
                if outcome is None:
                    continue
                sym, res, c_df = outcome
                scanned_count += 1
                pairs_summary.append({
                    "symbol": sym,
                    "signal": res.get("signal", "WAIT"),
                    "price": res.get("live_price", 0.0),
                    "reason": res.get("reason", ""),
                    "strategy_label": res.get("strategy_label", "")
                })
                if res.get("signal") in ("CALL", "PUT") and len(c_df) > 0:
                    closed_map[sym] = (c_df, res)
                    raw_confirmed.append({
                        "symbol": sym,
                        "signal": res["signal"],
                        "confidence": res["confidence"],
                        "strength": res.get("strength", 0.0),
                        "confluence": res.get("confluence", 1),
                        "strategy_label": res.get("strategy_label", ""),
                        "reason": res["reason"],
                        "price": res["live_price"]
                    })
        except FuturesTimeoutError:
            pass

    if scanned_count == 0:
        return JSONResponse({
            "ok": False,
            "error": "تم الوصول للحد المؤقت لمزود البيانات (24 طلب/دقيقة) — انتظر 45 ثانية ثم أعد المسح."
        })

    raw_confirmed.sort(key=lambda x: (x.get("confluence", 1), x.get("strength", 0.0)), reverse=True)
    pairs_summary.sort(key=lambda x: x["symbol"])
    killed_by_corr = []

    if agent_corr == 1 and len(raw_confirmed) > 1:
        confirmed, killed_by_corr = run_agent_correlation_assassin(raw_confirmed)
        if strat_id in ("2", "3") and len(confirmed) > 2:
            killed_by_corr.extend(confirmed[2:])
            confirmed = confirmed[:2]
    elif strat_id in ("2", "3"):
        confirmed = []
        used_currencies = set()
        for item in raw_confirmed:
            if len(confirmed) >= 2:
                break
            currs = get_currencies(item["symbol"])
            if currs & used_currencies:
                continue
            used_currencies |= currs
            confirmed.append(item)
    else:
        confirmed = raw_confirmed

    for item in confirmed[:3]:
        sym = item["symbol"]
        if sym in closed_map:
            c_df, full_res = closed_map[sym]
            if agent_mtf == 1:
                item["mtf"] = run_agent_mtf_guard(sym, tf, c_df, full_res)
            if agent_court == 1:
                item["court"] = run_agent_court(sym, tf, c_df, full_res)

    return JSONResponse({
        "ok": True,
        "mode": m,
        "scanned": scanned_count,
        "total": len(target_pairs),
        "confirmed": confirmed,
        "killed_by_corr": killed_by_corr,
        "pairs_summary": pairs_summary
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
