import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="M15 Sniper PRO")

# ==========================================
# 1. إعدادات مفاتيح API والتوزيع المتوازن (2400 طلب/يوم | 24 طلب/دقيقة)
# ==========================================
TWELVE_DATA_KEYS = [
    "3db4b82facc84c5a84dd9abc262a5acb",
    "5d43d790ef114a7994876c0829e9fe4a",
    "5c929836e14d412182bbc5fce8f61b70"
]
FINNHUB_API_KEY = "d75bfl9r01qk56kc51dgd75bfl9r01qk56kc51e0"

key_lock = threading.Lock()
cache_lock = threading.Lock()
autopilot_lock = threading.Lock()

current_key_index = 0
key_cooldowns = {0: 0.0, 1: 0.0, 2: 0.0}

# حالة الرادار التلقائي على السيرفر (حماية 6 ساعات)
AUTO_PILOT_DURATION_SECONDS = 6 * 3600  # 21600 ثانية
AUTO_PILOT_STATE = {
    "active": False,
    "expires_at": 0.0
}

CACHE_STORE = {}
CACHE_TTL_SECONDS = 150

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "M15-Sniper-PRO/2.3"})

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
# 2. جلب البيانات مع التوزيع الذكي والتبديل الفوري بين 3 مفاتيح
# ==========================================
def fetch_candles(symbol, timeframe="15M", outputsize=250):
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
            resp = HTTP_SESSION.get(url, timeout=12)
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
            if not values or len(values) < 50:
                return None, "بيانات الشموع غير مكتملة لهذا الزوج حالياً."

            df = pd.DataFrame(values)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

            if len(df) < 50:
                return None, "البيانات الرقمية للشموع غير صالحة."

            # تحويل التوقيت إلى توقيت غزة (UTC+2)
            df["datetime"] = pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=2)
            df = df.iloc[::-1].reset_index(drop=True)

            with cache_lock:
                CACHE_STORE[cache_key] = (time.time(), df.copy())

            return df, None

        except Exception:
            last_error = "انقطع الاتصال المؤقت بالسيرفر — أعد المحاولة بعد ثوانٍ."

    return None, last_error

# ==========================================
# 3. الحسابات الفنية والاستراتيجيات الأربع (بدون أي تعديل)
# ==========================================
def calculate_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Bands (15, 2.3, ddof=0)
    sma15 = close.rolling(window=15).mean()
    std15 = close.rolling(window=15).std(ddof=0)
    df["bb_mid"] = sma15
    df["bb_upper"] = sma15 + (2.3 * std15)
    df["bb_lower"] = sma15 - (2.3 * std15)

    # EMA 200 & EMA 50 & EMA 20
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # ATR 14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean().bfill()

    return df

def detect_sr_and_fib(df, symbol):
    decimals = get_price_decimals(symbol)
    recent = df.tail(120)
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

    atr_raw = df["atr"].iloc[-2]
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
                lvl["price"] = (lvl["price"] + price) / 2.0
                matched = True
                break
        if not matched:
            levels.append({"price": float(price), "touches": 1})

    current_close = float(df["close"].iloc[-2])
    resistances = sorted([l for l in levels if l["price"] > current_close and l["touches"] >= 3], key=lambda x: x["price"])[:2]
    supports = sorted([l for l in levels if l["price"] <= current_close and l["touches"] >= 3], key=lambda x: -x["price"])[:2]

    return resistances, supports, fib_levels

def evaluate_strategy(df, symbol, strategy_id):
    decimals = get_price_decimals(symbol)
    strategy_id = normalize_strategy(strategy_id)

    c = df.iloc[-2]
    live = df.iloc[-1]

    close_p = float(c["close"])
    open_p = float(c["open"])
    high_p = float(c["high"])
    low_p = float(c["low"])

    bb_u = float(c["bb_upper"]) if not pd.isna(c["bb_upper"]) else high_p * 1.001
    bb_l = float(c["bb_lower"]) if not pd.isna(c["bb_lower"]) else low_p * 0.999
    ema200 = float(c["ema200"])
    ema50 = float(c["ema50"])
    ema20 = float(c["ema20"])
    rsi = float(c["rsi"]) if not pd.isna(c["rsi"]) else 50.0
    default_atr = 0.05 if "JPY" in symbol else 0.0005
    atr = float(c["atr"]) if not pd.isna(c["atr"]) and float(c["atr"]) > 0 else default_atr

    resistances, supports, fib_levels = detect_sr_and_fib(df, symbol)

    signal = "WAIT"
    reason = "بانتظار ملامسة حدود البولينجر مع توافق EMA 200"
    confidence = 0

    if strategy_id == "1":  # M15 Sniper PRO (الأساسية)
        if (low_p <= bb_l or close_p <= bb_l) and close_p > ema200:
            signal = "CALL"
            reason = "ملامسة حد البولينجر السفلي فوق EMA 200 (ارتداد صاعد)"
            confidence = 86
        elif (high_p >= bb_u or close_p >= bb_u) and close_p < ema200:
            signal = "PUT"
            reason = "ملامسة حد البولينجر العلوي تحت EMA 200 (ارتداد هابط)"
            confidence = 86
        elif low_p <= bb_l and close_p > open_p:
            signal = "CALL"
            reason = "اختراق البولينجر السفلي مع شمعة انعكاس خضراء"
            confidence = 78
        elif high_p >= bb_u and close_p < open_p:
            signal = "PUT"
            reason = "اختراق البولينجر العلوي مع شمعة انعكاس حمراء"
            confidence = 78

    elif strategy_id == "2":  # مع الاتجاه (Trend Following)
        if close_p > ema200 and ema20 > ema50 and low_p <= ema20 and close_p > open_p:
            signal = "CALL"
            reason = "ارتداد صاعد من EMA 20 مع اتجاه صاعد قوي فوق EMA 200"
            confidence = 82
        elif close_p < ema200 and ema20 < ema50 and high_p >= ema20 and close_p < open_p:
            signal = "PUT"
            reason = "ارتداد هابط من EMA 20 مع اتجاه هابط تحت EMA 200"
            confidence = 82
        else:
            reason = "بانتظار إعادة اختبار EMA 20 في اتجاه EMA 200"

    elif strategy_id == "3":  # سكالبينج سريع (Scalp Reversal)
        if rsi <= 28 and low_p <= bb_l:
            signal = "CALL"
            reason = f"تشبع بيعي حاد (RSI={rsi:.1f}) عند البولينجر السفلي"
            confidence = 84
        elif rsi >= 72 and high_p >= bb_u:
            signal = "PUT"
            reason = f"تشبع شرائي حاد (RSI={rsi:.1f}) عند البولينجر العلوي"
            confidence = 84
        else:
            reason = f"مراقبة تشبع RSI ({rsi:.1f}) مع حدود البولينجر"

    elif strategy_id == "4":  # برايس أكشن ودعوم ومقاومات
        near_sup = any(abs(low_p - s["price"]) <= atr * 0.3 for s in supports)
        near_res = any(abs(high_p - r["price"]) <= atr * 0.3 for r in resistances)
        wick_down = (min(open_p, close_p) - low_p) > abs(close_p - open_p) * 1.3
        wick_up = (high_p - max(open_p, close_p)) > abs(close_p - open_p) * 1.3

        if near_sup and wick_down:
            signal = "CALL"
            reason = "ذيل رفض سفلي قوي عند منطقة دعم تاريخية"
            confidence = 83
        elif near_res and wick_up:
            signal = "PUT"
            reason = "ذيل رفض علوي قوي عند منطقة مقاومة تاريخية"
            confidence = 83
        else:
            reason = "بانتظار ذيل رفض واضح عند مناطق الدعم أو المقاومة"

    live_price = round(float(live["close"]), decimals)

    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "live_price": live_price,
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
    df = calculate_indicators(df)
    res = evaluate_strategy(df, sym, strategy)
    return sym, res

# ==========================================
# 4. مسارات التحكم بالرادار التلقائي (6 ساعات)
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
# 5. مسارات API الرئيسية والواجهة
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

    df = calculate_indicators(df)
    analysis = evaluate_strategy(df, sym, strategy)

    chart_df = df.tail(90).copy()
    candles = []
    for _, row in chart_df.iterrows():
        candles.append({
            "time": row["datetime"].strftime("%H:%M"),
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
