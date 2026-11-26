import os
import time
import datetime
import threading
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
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

UTC_TZ = datetime.timezone.utc
try:
    DISPLAY_TZ = ZoneInfo("Asia/Hebron")
except Exception:
    DISPLAY_TZ = datetime.timezone(datetime.timedelta(hours=2))

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
CACHE_TTL_SECONDS = 150

thread_local = threading.local()

def get_http_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "M15-Sniper-PRO/3.2"})
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

def normalize_strategy(raw_strat):
    s = str(raw_strat or "1").strip().lower()
    if s in ("1", "sniper", "m15", "default") or "sniper" in s or "1." in s:
        return "1"
    if s in ("2", "trend") or "trend" in s or "2." in s:
        return "2"
    if s in ("3", "scalp") or "scalp" in s or "3." in s:
        return "3"
    if s in ("4", "pa", "sr", "price_action") or "4." in s:
        return "4"
    if s in ("5", "stoch", "stochastic", "h1") or "stoch" in s or "5." in s:
        return "5"
    return "1"

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
        expired = [k for k, (ts, _) in CACHE_STORE.items() if now_ts - ts >= CACHE_TTL_SECONDS]
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
# 2. Fetch Candles (UTC End-to-End, 300 Warm-up Candles)
# ==========================================
def fetch_candles(symbol, timeframe="15M", outputsize=300):
    symbol = normalize_symbol(symbol)
    tf_upper = str(timeframe or "15M").strip().upper()
    interval = TIMEFRAME_MAP.get(tf_upper, "15min")
    cache_key = f"{symbol}_{interval}"
    now = time.time()

    cleanup_cache(now)

    with cache_lock:
        if cache_key in CACHE_STORE:
            cached_time, cached_df = CACHE_STORE[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                return cached_df.copy(), None

    last_error = "تم الوصول للحد المؤقت للطلبات (24 طلب/دقيقة) — انتظر 45 ثانية ثم أعد المحاولة."
    num_keys = len(TWELVE_DATA_KEYS)
    session = get_http_session()

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
            resp = session.get(url, timeout=12)
            if resp.status_code == 429:
                mark_key_cooldown(idx, 55.0)
                continue

            data = resp.json()
            if data.get("status") == "error":
                code = data.get("code", 0)
                msg = str(data.get("message", ""))
                if code == 429 or "run out of API credits" in msg:
                    mark_key_cooldown(idx, 55.0)
                    continue
                return None, f"تنبيه من مزود البيانات: {msg or 'خطأ غير معروف'}"

            values = data.get("values")
            if not values or len(values) < 205:
                return None, "بيانات الشموع غير كافية لحساب EMA 200 بدقة."

            df = pd.DataFrame(values)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["open_dt_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "open_dt_utc"])

            if len(df) < 205:
                return None, "البيانات الرقمية للشموع غير صالحة."

            df["open_ts"] = (df["open_dt_utc"].astype("int64") // 10**9).astype(int)
            df = df.sort_values("open_ts").reset_index(drop=True)

            with cache_lock:
                CACHE_STORE[cache_key] = (time.time(), df.copy())

            return df, None

        except Exception:
            last_error = "انقطع الاتصال المؤقت بالسيرفر — أعد المحاولة بعد ثوانٍ."

    return None, last_error

def filter_closed_candles(df, timeframe):
    """Strictly drop any in-progress candle where open_ts + tf_sec > now_utc."""
    tf_sec = TIMEFRAME_SECONDS.get(str(timeframe).upper(), 900)
    now_ts = time.time()
    live_price = float(df["close"].iloc[-1])
    closed_df = df[(df["open_ts"] + tf_sec) <= now_ts].copy().reset_index(drop=True)
    return closed_df, live_price

# ==========================================
# 3. Audited Indicators (Strict Closed Candles)
# ==========================================
def calculate_indicators(closed_df):
    df = closed_df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Smart Bollinger Bands (15, 2.0, ddof=0)
    sma15 = close.rolling(window=15, min_periods=15).mean()
    std15 = close.rolling(window=15, min_periods=15).std(ddof=0)
    df["bb_mid"] = sma15
    df["bb_upper"] = sma15 + (2.0 * std15)
    df["bb_lower"] = sma15 - (2.0 * std15)

    # Validated EMA 200 (adjust=False) + EMA 50 & EMA 20
    df["ema200"] = close.ewm(span=200, min_periods=200, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, min_periods=50, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, min_periods=20, adjust=False).mean()

    # Wilder RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[(avg_loss == 0.0) & (avg_gain > 0.0), "rsi"] = 100.0
    df.loc[(avg_loss == 0.0) & (avg_gain == 0.0), "rsi"] = 50.0

    # Wilder ATR (14) - No bfill()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    # Strategy 5: Stochastic Oscillator (%K=9, %D=5)
    lowest_low = low.rolling(window=9, min_periods=9).min()
    highest_high = high.rolling(window=9, min_periods=9).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    df["stoch_k"] = 100.0 * ((close - lowest_low) / denom)
    df["stoch_d"] = df["stoch_k"].rolling(window=5, min_periods=5).mean()

    return df

def detect_sr_and_fib(closed_df, symbol):
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

    atr_raw = closed_df["atr"].iloc[-1]
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

def evaluate_strategy(closed_df, live_price, symbol, strategy_id):
    decimals = get_price_decimals(symbol)
    strategy_id = normalize_strategy(strategy_id)

    if len(closed_df) < 201:
        return {
            "signal": "WAIT",
            "reason": "بيانات الشموع المغلقة غير كافية لحساب EMA 200.",
            "confidence": 0,
            "live_price": round(live_price, decimals),
            "bb_upper": None, "bb_lower": None, "ema200": None,
            "resistances": [], "supports": [], "fib": {}
        }

    c = closed_df.iloc[-1]     # Strictly last CLOSED candle
    prev = closed_df.iloc[-2]  # Previous CLOSED candle

    required_cols = ["bb_upper", "bb_lower", "ema200"]
    if strategy_id == "1":
        required_cols.extend(["rsi", "atr"])
    elif strategy_id == "2":
        required_cols.extend(["ema50", "ema20"])
    elif strategy_id == "3":
        required_cols.append("rsi")
    elif strategy_id == "4":
        required_cols.append("atr")
    elif strategy_id == "5":
        required_cols.extend(["stoch_k", "stoch_d"])

    # Strict NaN guard: never fabricate indicator values
    if any(pd.isna(c[col]) for col in required_cols) or (
        strategy_id == "5" and (pd.isna(prev["stoch_k"]) or pd.isna(prev["stoch_d"]))
    ):
        return {
            "signal": "WAIT",
            "reason": "مؤشرات التهيئة غير مكتملة (NaN) — تم تخطي التقييم.",
            "confidence": 0,
            "live_price": round(live_price, decimals),
            "bb_upper": None, "bb_lower": None, "ema200": None,
            "resistances": [], "supports": [], "fib": {}
        }

    close_p = float(c["close"])
    open_p = float(c["open"])
    high_p = float(c["high"])
    low_p = float(c["low"])

    bb_u = float(c["bb_upper"])
    bb_l = float(c["bb_lower"])
    ema200 = float(c["ema200"])
    ema50 = float(c["ema50"]) if not pd.isna(c["ema50"]) else ema200
    ema20 = float(c["ema20"]) if not pd.isna(c["ema20"]) else ema200
    rsi = float(c["rsi"]) if not pd.isna(c["rsi"]) else 50.0
    atr = float(c["atr"]) if not pd.isna(c["atr"]) else 0.0

    resistances, supports, fib_levels = detect_sr_and_fib(closed_df, symbol)

    signal = "WAIT"
    reason = "بانتظار إغلاق أو ذيل رفض عند BB(15, 2.0) مع توافق EMA 200"
    confidence = 0

    if strategy_id == "1":  # Smart Rejection Engine (BB 15/2.0 + EMA 200 + ATR + RSI)
        candle_range = max(high_p - low_p, 1e-9)
        body_size = abs(close_p - open_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)
        not_blowout = (atr <= 0.0) or (body_size <= 2.2 * atr)

        if close_p > ema200 and close_p <= bb_l and rsi <= 42.0 and not_blowout:
            signal = "CALL"
            reason = f"[A+] إغلاق مؤكد <= البولينجر السفلي وفوق EMA 200 (RSI={rsi:.1f})"
            confidence = 62
        elif close_p < ema200 and close_p >= bb_u and rsi >= 58.0 and not_blowout:
            signal = "PUT"
            reason = f"[A+] إغلاق مؤكد >= البولينجر العلوي وتحت EMA 200 (RSI={rsi:.1f})"
            confidence = 62
        elif close_p > ema200 and low_p <= bb_l and (lower_wick / candle_range) >= 0.35 and rsi <= 42.0 and not_blowout:
            signal = "CALL"
            reason = f"[A] ذيل رفض سفلي قوي عند البولينجر السفلي وفوق EMA 200 (RSI={rsi:.1f})"
            confidence = 58
        elif close_p < ema200 and high_p >= bb_u and (upper_wick / candle_range) >= 0.35 and rsi >= 58.0 and not_blowout:
            signal = "PUT"
            reason = f"[A] ذيل رفض علوي قوي عند البولينجر العلوي وتحت EMA 200 (RSI={rsi:.1f})"
            confidence = 58

    elif strategy_id == "2":  # Trend Following
        if close_p > ema200 and ema20 > ema50 and low_p <= ema20 and close_p > open_p:
            signal = "CALL"
            reason = "ارتداد صاعد مغلق من EMA 20 مع اتجاه صاعد فوق EMA 200"
            confidence = 55
        elif close_p < ema200 and ema20 < ema50 and high_p >= ema20 and close_p < open_p:
            signal = "PUT"
            reason = "ارتداد هابط مغلق من EMA 20 مع اتجاه هابط تحت EMA 200"
            confidence = 55
        else:
            reason = "بانتظار إعادة اختبار EMA 20 في اتجاه EMA 200"

    elif strategy_id == "3":  # Scalp Reversal (Wilder RSI)
        if rsi <= 28.0 and close_p <= bb_l:
            signal = "CALL"
            reason = f"تشبع بيعي Wilder (RSI={rsi:.1f}) مع إغلاق عند البولينجر السفلي"
            confidence = 55
        elif rsi >= 72.0 and close_p >= bb_u:
            signal = "PUT"
            reason = f"تشبع شرائي Wilder (RSI={rsi:.1f}) مع إغلاق عند البولينجر العلوي"
            confidence = 55
        else:
            reason = f"مراقبة تشبع Wilder RSI ({rsi:.1f}) مع إغلاق البولينجر"

    elif strategy_id == "4":  # Price Action S/R
        near_sup = any(abs(low_p - s["price"]) <= atr * 0.3 for s in supports)
        near_res = any(abs(high_p - r["price"]) <= atr * 0.3 for r in resistances)
        wick_down = (min(open_p, close_p) - low_p) > abs(close_p - open_p) * 1.3
        wick_up = (high_p - max(open_p, close_p)) > abs(close_p - open_p) * 1.3

        if near_sup and wick_down:
            signal = "CALL"
            reason = "ذيل رفض سفلي مغلق عند منطقة دعم تاريخية"
            confidence = 55
        elif near_res and wick_up:
            signal = "PUT"
            reason = "ذيل رفض علوي مغلق عند منطقة مقاومة تاريخية"
            confidence = 55
        else:
            reason = "بانتظار ذيل رفض واضح عند مناطق الدعم أو المقاومة"

    elif strategy_id == "5":  # H1 Stochastic Extreme Reversal (K=9, D=5, 10/90 + EMA200)
        k_curr = float(c["stoch_k"])
        d_curr = float(c["stoch_d"])
        k_prev = float(prev["stoch_k"])
        d_prev = float(prev["stoch_d"])

        if close_p > ema200 and k_prev <= d_prev and k_curr > d_curr and k_curr < 10.0:
            signal = "CALL"
            reason = f"تقاطع صاعد Stoch(9,5) تحت 10 (K={k_curr:.1f}, D={d_curr:.1f}) فوق EMA 200"
            confidence = 60
        elif close_p < ema200 and k_prev >= d_prev and k_curr < d_curr and k_curr > 90.0:
            signal = "PUT"
            reason = f"تقاطع هابط Stoch(9,5) فوق 90 (K={k_curr:.1f}, D={d_curr:.1f}) تحت EMA 200"
            confidence = 60
        else:
            reason = f"مراقبة Stoch(9,5) (K={k_curr:.1f}, D={d_curr:.1f}) مع توافق EMA 200"

    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "live_price": round(live_price, decimals),
        "bb_upper": round(bb_u, decimals),
        "bb_lower": round(bb_l, decimals),
        "ema200": round(ema200, decimals),
        "resistances": [{"price": round(r["price"], decimals), "touches": r["touches"]} for r in resistances],
        "supports": [{"price": round(s["price"], decimals), "touches": s["touches"]} for s in supports],
        "fib": fib_levels
    }

def scan_single_pair(sym, timeframe, strategy):
    df, err = fetch_candles(sym, timeframe)
    if err or df is None:
        return None
    closed_df, live_price = filter_closed_candles(df, timeframe)
    closed_df = calculate_indicators(closed_df)
    res = evaluate_strategy(closed_df, live_price, sym, strategy)
    return sym, res

# ==========================================
# 4. Autopilot Routes (6 Hours)
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
# 5. Main API & Frontend Routes
# ==========================================
@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/api/analyze")
@app.get("/analyze")
def api_analyze(
    symbol: str = Query(default="EUR/USD"),
    timeframe: str = Query(default="15M"),
    strategy: str = Query(default="1")
):
    sym = normalize_symbol(symbol)
    tf = str(timeframe).upper()
    decimals = get_price_decimals(sym)

    df, err = fetch_candles(sym, tf)
    if err:
        return JSONResponse({"ok": False, "error": err})

    closed_df, live_price = filter_closed_candles(df, tf)
    closed_df = calculate_indicators(closed_df)
    analysis = evaluate_strategy(closed_df, live_price, sym, strategy)

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
            "bb_upper": round(float(row["bb_upper"]), decimals) if not pd.isna(row["bb_upper"]) else None,
            "bb_lower": round(float(row["bb_lower"]), decimals) if not pd.isna(row["bb_lower"]) else None,
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
    strategy: str = Query(default="1"),
    mode: str = Query(default="manual")
):
    tf = str(timeframe).upper()
    m = str(mode).strip().lower()

    if m == "auto":
        active, _ = check_autopilot_state()
        if not active:
            return JSONResponse({
                "ok": False,
                "auto_stopped": True,
                "error": "توقف الرادار التلقائي بعد انتهاء مدة الـ 6 ساعات أو تم إيقافه يدوياً."
            })

    target_pairs = PAIRS_22
    confirmed = []
    scanned_count = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(scan_single_pair, sym, tf, strategy): sym for sym in target_pairs}
        for future in as_completed(futures):
            try:
                outcome = future.result()
            except Exception:
                continue
            if outcome is None:
                continue
            sym, res = outcome
            scanned_count += 1

            if res["signal"] in ("CALL", "PUT"):
                confirmed.append({
                    "symbol": sym,
                    "signal": res["signal"],
                    "confidence": res["confidence"],
                    "reason": res["reason"],
                    "price": res["live_price"]
                })

    return JSONResponse({
        "ok": True,
        "mode": m,
        "scanned": scanned_count,
        "total": len(target_pairs),
        "confirmed": confirmed
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
