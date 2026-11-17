#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="M15 Sniper PRO Ultimate Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

try:
    GAZA_TZ = ZoneInfo("Asia/Hebron")
except Exception:
    GAZA_TZ = timezone.utc

TRADING_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "NZD/USD",
    "USD/CAD", "USD/CHF", "EUR/GBP", "EUR/AUD", "EUR/CAD",
    "EUR/CHF", "EUR/NZD", "GBP/AUD", "GBP/CAD", "GBP/CHF",
    "GBP/NZD", "AUD/CAD", "AUD/CHF", "AUD/NZD", "CAD/CHF",
    "CAD/JPY", "CHF/JPY"
]

# ثوابت الاستراتيجية الأساسية (محمية 100% بدون أي تعديل)
BB_PERIOD = 15
BB_STD_DEV = 2.3
EMA_TREND_PERIOD = 200

# ذاكرة مؤقتة لحماية رصيد Twelve Data (45 ثانية)
SCAN_CACHE = {}
SCAN_CACHE_TTL = 45.0


def map_interval(tf: str) -> str:
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h"
    }
    return mapping.get(tf, "15min")


def tf_to_seconds(tf: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600
    }
    return mapping.get(tf, 900)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # 1. مؤشرات الاستراتيجية الأساسية (BB 15, 2.3, ddof=0 + EMA 200)
    sma = c.rolling(window=BB_PERIOD).mean()
    std = c.rolling(window=BB_PERIOD).std(ddof=0)
    upper_band = sma + (BB_STD_DEV * std)
    lower_band = sma - (BB_STD_DEV * std)
    ema_200 = c.ewm(span=EMA_TREND_PERIOD, adjust=False).mean()

    # 2. مؤشرات الاستراتيجيات المساعدة وقراءة السوق (معزولة تماماً)
    ema_21 = c.ewm(span=21, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()

    # ATR (14) لقياس متوسط حركة الشمعة في محرك التنبؤ المستقبلي
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().bfill()

    # RSI (7) للسكالبينج السريع + RSI (14) لقراءة السوق
    delta = c.diff()
    gain7 = delta.clip(lower=0).rolling(window=7).mean()
    loss7 = (-delta.clip(upper=0)).rolling(window=7).mean()
    rs7 = gain7 / loss7.replace(0, np.nan)
    rsi_7 = (100.0 - (100.0 / (1.0 + rs7))).fillna(50.0)

    gain14 = delta.clip(lower=0).rolling(window=14).mean()
    loss14 = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs14 = gain14 / loss14.replace(0, np.nan)
    rsi_14 = (100.0 - (100.0 / (1.0 + rs14))).fillna(50.0)

    # MACD (12, 26, 9)
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    return df.assign(
        bb_upper=upper_band,
        bb_lower=lower_band,
        ema_200=ema_200,
        ema_21=ema_21,
        ema_50=ema_50,
        atr_14=atr_14,
        rsi_7=rsi_7,
        rsi_14=rsi_14,
        macd_line=macd_line,
        macd_signal=macd_signal
    )


def calculate_support_resistance(df: pd.DataFrame, pip_mult: float, lookback: int = 120):
    sub = df.tail(lookback).copy()
    if len(sub) < 20:
        return []

    highs = sub["high"].values
    lows = sub["low"].values
    closes = sub["close"].values
    curr_price = float(closes[-1])

    pivots = []
    window = 4
    for i in range(window, len(sub) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            pivots.append(float(highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            pivots.append(float(lows[i]))

    if not pivots:
        return []

    tolerance = 4.0 / pip_mult
    clusters = []
    for p in sorted(pivots):
        matched = False
        for c in clusters:
            if abs(c["price"] - p) <= tolerance:
                c["prices"].append(p)
                c["price"] = float(np.mean(c["prices"]))
                matched = True
                break
        if not matched:
            clusters.append({"price": p, "prices": [p]})

    levels = []
    touch_tol = 2.5 / pip_mult
    digits = 3 if pip_mult == 100.0 else 5
    for c in clusters:
        lvl = c["price"]
        touches = int(np.sum((lows <= lvl + touch_tol) & (highs >= lvl - touch_tol)))
        if touches >= 2:
            lvl_type = "RESISTANCE" if lvl > curr_price else "SUPPORT"
            levels.append({
                "price": round(lvl, digits),
                "touches": touches,
                "type": lvl_type
            })

    levels.sort(key=lambda x: (-x["touches"], abs(x["price"] - curr_price)))
    return levels[:5]


def calculate_auto_fibonacci(df: pd.DataFrame, digits: int, lookback: int = 90):
    sub = df.tail(lookback)
    if len(sub) < 20:
        return []

    swing_high = float(sub["high"].max())
    swing_low = float(sub["low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return []

    high_idx = sub["high"].idxmax()
    low_idx = sub["low"].idxmin()
    is_uptrend = high_idx > low_idx

    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = []
    for r in ratios:
        if is_uptrend:
            price = swing_high - (diff * r)
        else:
            price = swing_low + (diff * r)
        fib_levels.append({
            "ratio": f"{r:.3f}" if r not in (0.0, 0.5, 1.0) else f"{r:.1f}",
            "price": round(price, digits),
            "is_golden": r == 0.618
        })
    return fib_levels


def calculate_predictive_eta(df: pd.DataFrame, tf: str, pip_mult: float, digits: int, sr_levels=None, fib_levels=None):
    """
    محرك التنبؤ المستقبلي العلمي (Smart Predictive Engine):
    يحسب المسافة المتبقية، سرعة الاقتراب الفعلية، ومتوسط المدى الحقيقي ATR
    لتقدير زمن وصول السعر للهدف السعري ونسبة الثقة بدقة كمية.
    """
    if len(df) < 20:
        return None

    live_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    older_bar = df.iloc[-4]

    if (
        pd.isna(live_bar["close"])
        or pd.isna(live_bar["bb_upper"])
        or pd.isna(live_bar["bb_lower"])
        or pd.isna(live_bar["ema_200"])
        or pd.isna(live_bar["atr_14"])
    ):
        return None

    price = float(live_bar["close"])
    bb_upper = float(live_bar["bb_upper"])
    bb_lower = float(live_bar["bb_lower"])
    ema_200 = float(live_bar["ema_200"])
    atr_pips = max(float(live_bar["atr_14"]) * pip_mult, 0.5)

    # تحديد الاتجاه والهدف السعري المتوافق مع EMA 200
    if price > ema_200 and bb_lower > ema_200:
        bias = "CALL"
        target_price = bb_lower
        dist_pips = (price - target_price) * pip_mult
        # سرعة الهبوط نحو البولينجر السفلي في آخر 3 شموع
        approach_move = (float(older_bar["close"]) - price) * pip_mult
        last_step = (float(prev_bar["close"]) - price) * pip_mult
    elif price < ema_200 and bb_upper < ema_200:
        bias = "PUT"
        target_price = bb_upper
        dist_pips = (target_price - price) * pip_mult
        # سرعة الصعود نحو البولينجر العلوي في آخر 3 شموع
        approach_move = (price - float(older_bar["close"])) * pip_mult
        last_step = (price - float(prev_bar["close"])) * pip_mult
    else:
        return None

    # إذا كان السعر بعيداً جداً (أكثر من 3 أضعاف ATR) لا نصدر توقعاً
    if dist_pips < 0 or dist_pips > (atr_pips * 3.0):
        return None

    # حساب السرعة الفعالة للشمعة الواحدة (دمج ATR مع زخم الاقتراب الفعلي)
    avg_approach_per_bar = max(approach_move / 3.0, 0.0)
    effective_velocity = (atr_pips * 0.65) + (avg_approach_per_bar * 0.35)
    if effective_velocity <= 0.1:
        effective_velocity = atr_pips * 0.5

    est_candles = int(np.ceil(dist_pips / effective_velocity))
    est_candles = max(1, min(est_candles, 4))

    # حساب التوقيت المستقبلي الدقيق لإغلاق الشمعة المتوقعة بتوقيت غزة
    tf_sec = tf_to_seconds(tf)
    now_utc = datetime.now(timezone.utc)
    now_ts = int(now_utc.timestamp())
    rem_current_sec = tf_sec - (now_ts % tf_sec)
    target_utc = now_utc + timedelta(seconds=rem_current_sec + ((est_candles - 1) * tf_sec))
    eta_gaza_str = target_utc.astimezone(GAZA_TZ).strftime("%H:%M")

    # حساب درجة الثقة الكمية (Confidence Score %)
    confidence = 55
    # 1. زخم الاقتراب الفعلي في آخر شمعة
    if last_step > 0:
        confidence += 12
    if approach_move > 0:
        confidence += 8

    # 2. القرب النسبي مقارنة بـ ATR
    if dist_pips <= atr_pips:
        confidence += 10

    # 3. تزامن الهدف السعري مع دعم/مقاومة قوي (S/R)
    confluences = []
    tol_price = 4.0 / pip_mult
    if sr_levels:
        for lvl in sr_levels:
            if abs(lvl["price"] - target_price) <= tol_price:
                confidence += 10
                confluences.append(f"تزامن مع {lvl['type']} ({lvl['touches']} لمسات)")
                break

    # 4. تزامن الهدف السعري مع مستوى فيبوناتشي
    if fib_levels:
        for fib in fib_levels:
            if abs(fib["price"] - target_price) <= tol_price:
                confidence += 8 if fib["is_golden"] else 5
                confluences.append(f"مستوى Fib {fib['ratio']}")
                break

    confidence = min(confidence, 94)
    conf_text = " + ".join(confluences) if confluences else "توافق اتجاه EMA 200 وزخم الاقتراب"

    return {
        "bias": bias,
        "target_price": round(target_price, digits),
        "distance_pips": round(dist_pips, 1),
        "est_candles": est_candles,
        "eta_time": eta_gaza_str,
        "confidence": confidence,
        "notes": conf_text
    }


def evaluate_strategy_row(df: pd.DataFrame, idx: int, strategy: str, digits: int, sr_levels=None):
    if idx < 1 or idx >= len(df):
        return "WAIT", 0.0, "بيانات غير كافية"

    bar = df.iloc[idx]
    if (
        pd.isna(bar["close"])
        or pd.isna(bar["bb_upper"])
        or pd.isna(bar["bb_lower"])
        or pd.isna(bar["ema_200"])
    ):
        return "WAIT", 0.0, "المؤشرات غير جاهزة بعد"

    c_close = float(bar["close"])
    c_open = float(bar["open"])
    c_high = float(bar["high"])
    c_low = float(bar["low"])
    bb_upper = float(bar["bb_upper"])
    bb_lower = float(bar["bb_lower"])
    ema_200 = float(bar["ema_200"])

    # 1. الاستراتيجية الأساسية (M15 Sniper PRO - نقية 100%)
    if strategy == "sniper_m15":
        if c_close > ema_200 and c_close <= bb_lower:
            return (
                "CALL",
                c_close,
                f"M15 Sniper CALL: إغلاق ({c_close:.{digits}f}) عند/أسفل BB السفلي ({bb_lower:.{digits}f}) وأعلى EMA200 ({ema_200:.{digits}f})."
            )
        if c_close < ema_200 and c_close >= bb_upper:
            return (
                "PUT",
                c_close,
                f"M15 Sniper PUT: إغلاق ({c_close:.{digits}f}) عند/أعلى BB العلوي ({bb_upper:.{digits}f}) وأسفل EMA200 ({ema_200:.{digits}f})."
            )
        return "WAIT", c_close, "بانتظار ملامسة حدود البولينجر مع توافق EMA 200"

    # 2. استراتيجية مع الاتجاه (Trend Following)
    if strategy == "trend":
        ema_21 = float(bar["ema_21"])
        ema_50 = float(bar["ema_50"])
        macd_l = float(bar["macd_line"])
        macd_s = float(bar["macd_signal"])
        if c_close > ema_50 > ema_200 and c_low <= ema_21 and c_close > ema_21 and macd_l > macd_s:
            return (
                "CALL",
                c_close,
                "Trend CALL: ترند صاعد (EMA50 > EMA200) مع ارتداد من EMA21 وتقاطع MACD إيجابي."
            )
        if c_close < ema_50 < ema_200 and c_high >= ema_21 and c_close < ema_21 and macd_l < macd_s:
            return (
                "PUT",
                c_close,
                "Trend PUT: ترند هابط (EMA50 < EMA200) مع ارتداد من EMA21 وتقاطع MACD سلبي."
            )
        return "WAIT", c_close, "بانتظار ارتداد سعري متوافق مع EMA 21 و 50 و 200"

    # 3. استراتيجية السكالبينج السريع (Scalp Reversal)
    if strategy == "scalp":
        rsi7 = float(bar["rsi_7"])
        if c_close <= bb_lower and rsi7 <= 25.0:
            return (
                "CALL",
                c_close,
                f"Scalp CALL: تشبع بيعي حاد RSI(7)={rsi7:.1f} مع اختراق البولينجر السفلي."
            )
        if c_close >= bb_upper and rsi7 >= 75.0:
            return (
                "PUT",
                c_close,
                f"Scalp PUT: تشبع شرائي حاد RSI(7)={rsi7:.1f} مع اختراق البولينجر العلوي."
            )
        return "WAIT", c_close, "بانتظار تشبع RSI(7) عند حدود البولينجر للارتداد السريع"

    # 4. استراتيجية الارتداد والبرايس أكشن (Price Action & S/R)
    if strategy == "price_action":
        body = abs(c_close - c_open)
        upper_wick = c_high - max(c_open, c_close)
        lower_wick = min(c_open, c_close) - c_low
        min_body = max(body, 1e-6)

        if lower_wick >= (min_body * 2.0) and c_close > c_open and c_low <= bb_lower:
            return (
                "CALL",
                c_close,
                "Price Action CALL: شمعة رفض شرائية بذيل سفلي طويل عند منطقة دعم/بولينجر."
            )
        if upper_wick >= (min_body * 2.0) and c_close < c_open and c_high >= bb_upper:
            return (
                "PUT",
                c_close,
                "Price Action PUT: شمعة رفض بيعية بذيل علوي طويل عند منطقة مقاومة/بولينجر."
            )
        return "WAIT", c_close, "بانتظار شمعة رفض واضحة (Pin Bar) عند حدود البولينجر"

    return "WAIT", c_close, "لا توجد إشارة حالياً"


def extract_historical_markers(df: pd.DataFrame, strategy: str, digits: int):
    markers = []
    if len(df) < BB_PERIOD + 2:
        return markers

    for i in range(BB_PERIOD, len(df) - 1):
        sig, _, _ = evaluate_strategy_row(df, i, strategy, digits)
        ts = int(df.iloc[i]["datetime"].timestamp())
        if sig == "CALL":
            markers.append({
                "time": ts,
                "position": "belowBar",
                "color": "#10b981",
                "shape": "arrowUp",
                "text": "CALL"
            })
        elif sig == "PUT":
            markers.append({
                "time": ts,
                "position": "aboveBar",
                "color": "#ef4444",
                "shape": "arrowDown",
                "text": "PUT"
            })
    return markers


def build_market_summary(df: pd.DataFrame, digits: int):
    bar = df.iloc[-1]
    price = float(bar["close"])
    ema_200 = float(bar["ema_200"])
    ema_50 = float(bar["ema_50"])
    rsi_14 = float(bar["rsi_14"])
    macd_l = float(bar["macd_line"])
    macd_s = float(bar["macd_signal"])

    structure = "صاعد (Bullish)" if price > ema_200 and ema_50 > ema_200 else (
        "هابط (Bearish)" if price < ema_200 and ema_50 < ema_200 else "عرضي / متذبذب"
    )
    macd_state = "إيجابي (صاعد)" if macd_l > macd_s else "سلبي (هابط)"
    rsi_state = "تشبع شرائي" if rsi_14 >= 70 else ("تشبع بيعي" if rsi_14 <= 30 else "متوازن")

    return {
        "structure": structure,
        "rsi_14": round(rsi_14, 1),
        "rsi_state": rsi_state,
        "macd_state": macd_state,
        "bb_upper": round(float(bar["bb_upper"]), digits),
        "bb_lower": round(float(bar["bb_lower"]), digits),
        "ema_200": round(ema_200, digits)
    }


def fetch_twelve_data(symbol: str, tf: str):
    if not TWELVE_DATA_KEY:
        return None, "مفتاح TWELVE_DATA_KEY غير مضاف في إعدادات الاستضافة."

    interval = map_interval(tf)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 250,
        "apikey": TWELVE_DATA_KEY,
        "timezone": "UTC"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok" or "values" not in data:
            return None, data.get("message", "فشل جلب البيانات من Twelve Data")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("datetime").reset_index(drop=True)
        df = calculate_indicators(df)
        return df, None
    except Exception as e:
        return None, f"خطأ في الاتصال بمزود البيانات: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/pairs")
async def get_pairs():
    return {"pairs": TRADING_PAIRS}


@app.get("/api/analyze")
async def analyze_pair(
    symbol: str = Query("EUR/USD"),
    tf: str = Query("15m"),
    strategy: str = Query("sniper_m15")
):
    symbol = symbol.strip().upper()
    if symbol not in TRADING_PAIRS:
        return JSONResponse(status_code=400, content={"error": f"زوج العملات غير مدعوم: {symbol}"})

    valid_strategies = {"sniper_m15", "trend", "scalp", "price_action"}
    if strategy not in valid_strategies:
        strategy = "sniper_m15"

    df, err = fetch_twelve_data(symbol, tf)
    if err or df is None or len(df) < 205:
        return JSONResponse(status_code=400, content={"error": err or "بيانات غير كافية للتحليل"})

    pip_mult = 100.0 if "JPY" in symbol else 10000.0
    digits = 3 if "JPY" in symbol else 5

    candles = []
    bb_upper_line = []
    bb_lower_line = []
    ema200_line = []

    for _, row in df.iterrows():
        ts = int(row["datetime"].timestamp())
        candles.append({
            "time": ts,
            "open": round(float(row["open"]), digits),
            "high": round(float(row["high"]), digits),
            "low": round(float(row["low"]), digits),
            "close": round(float(row["close"]), digits),
        })
        if pd.notna(row["bb_upper"]):
            bb_upper_line.append({"time": ts, "value": round(float(row["bb_upper"]), digits)})
        if pd.notna(row["bb_lower"]):
            bb_lower_line.append({"time": ts, "value": round(float(row["bb_lower"]), digits)})
        if pd.notna(row["ema_200"]):
            ema200_line.append({"time": ts, "value": round(float(row["ema_200"]), digits)})

    sr_levels = calculate_support_resistance(df, pip_mult)
    fib_levels = calculate_auto_fibonacci(df, digits)
    signal, _, reason = evaluate_strategy_row(df, len(df) - 2, strategy, digits, sr_levels)
    predictive_eta = calculate_predictive_eta(df, tf, pip_mult, digits, sr_levels, fib_levels)
    markers = extract_historical_markers(df, strategy, digits)
    summary = build_market_summary(df, digits)

    live_bar = df.iloc[-1]
    gaza_now = datetime.now(timezone.utc).astimezone(GAZA_TZ)

    return {
        "symbol": symbol,
        "timeframe": tf,
        "strategy": strategy,
        "live_price": round(float(live_bar["close"]), digits),
        "signal": signal,
        "reason": reason,
        "predictive_eta": predictive_eta,
        "gaza_time": gaza_now.strftime("%H:%M:%S"),
        "sr_levels": sr_levels,
        "fib_levels": fib_levels,
        "markers": markers,
        "summary": summary,
        "candles": candles,
        "indicators": {
            "bb_upper": bb_upper_line,
            "bb_lower": bb_lower_line,
            "ema_200": ema200_line
        }
    }


@app.get("/api/scan")
async def scan_market(
    tf: str = Query("15m"),
    strategy: str = Query("sniper_m15")
):
    if not TWELVE_DATA_KEY:
        return JSONResponse(status_code=400, content={"error": "مفتاح TWELVE_DATA_KEY غير مضاف."})

    valid_strategies = {"sniper_m15", "trend", "scalp", "price_action"}
    if strategy not in valid_strategies:
        strategy = "sniper_m15"

    cache_key = f"{tf}_{strategy}"
    now_ts = time.time()
    if cache_key in SCAN_CACHE:
        cached_time, cached_payload = SCAN_CACHE[cache_key]
        if (now_ts - cached_time) < SCAN_CACHE_TTL:
            resp_copy = dict(cached_payload)
            resp_copy["from_cache"] = True
            return resp_copy

    active_signals = []
    near_setups = []
    scanned_count = 0

    batch_size = 4
    for i in range(0, len(TRADING_PAIRS), batch_size):
        batch = TRADING_PAIRS[i:i + batch_size]
        tasks = [asyncio.to_thread(fetch_twelve_data, sym, tf) for sym in batch]
        results = await asyncio.gather(*tasks)

        for sym, (df, err) in zip(batch, results):
            if err or df is None or len(df) < 205:
                continue

            scanned_count += 1
            pip_mult = 100.0 if "JPY" in sym else 10000.0
            digits = 3 if "JPY" in sym else 5

            sig, entry_price, reason = evaluate_strategy_row(df, len(df) - 2, strategy, digits)
            if sig in ("CALL", "PUT"):
                active_signals.append({
                    "symbol": sym,
                    "signal": sig,
                    "price": round(entry_price, digits),
                    "reason": reason
                })
            else:
                sr = calculate_support_resistance(df, pip_mult)
                fib = calculate_auto_fibonacci(df, digits)
                eta = calculate_predictive_eta(df, tf, pip_mult, digits, sr, fib)
                if eta is not None:
                    near_setups.append({
                        "symbol": sym,
                        "bias": eta["bias"],
                        "distance_pips": eta["distance_pips"],
                        "eta_time": eta["eta_time"],
                        "est_candles": eta["est_candles"],
                        "confidence": eta["confidence"],
                        "target_price": eta["target_price"]
                    })

        if i + batch_size < len(TRADING_PAIRS):
            await asyncio.sleep(11)

    near_setups.sort(key=lambda x: (-x["confidence"], x["distance_pips"]))
    gaza_now = datetime.now(timezone.utc).astimezone(GAZA_TZ)

    payload = {
        "timeframe": tf,
        "strategy": strategy,
        "scanned_count": scanned_count,
        "total_pairs": len(TRADING_PAIRS),
        "scan_time": gaza_now.strftime("%H:%M:%S"),
        "active_signals": active_signals,
        "near_setups": near_setups[:3],
        "from_cache": False
    }
    SCAN_CACHE[cache_key] = (now_ts, payload)
    return payload
