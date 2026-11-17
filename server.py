#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="M15 Sniper Pro Live Dashboard")

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

BB_PERIOD = 15
BB_STD_DEV = 2.3
EMA_TREND_PERIOD = 200


def map_interval(tf: str) -> str:
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h"
    }
    return mapping.get(tf, "15min")


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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    sma = c.rolling(window=BB_PERIOD).mean()
    std = c.rolling(window=BB_PERIOD).std(ddof=0)
    upper_band = sma + (BB_STD_DEV * std)
    lower_band = sma - (BB_STD_DEV * std)
    ema_200 = c.ewm(span=EMA_TREND_PERIOD, adjust=False).mean()

    return df.assign(
        bb_upper=upper_band,
        bb_lower=lower_band,
        ema_200=ema_200
    )


def check_signal_conditions(df: pd.DataFrame, digits: int):
    if len(df) < 3:
        return "WAIT", 0.0, "بيانات غير كافية"

    closed_bar = df.iloc[-2]
    if (
        pd.isna(closed_bar["close"])
        or pd.isna(closed_bar["bb_upper"])
        or pd.isna(closed_bar["bb_lower"])
        or pd.isna(closed_bar["ema_200"])
    ):
        return "WAIT", 0.0, "المؤشرات غير جاهزة بعد"

    curr_close = float(closed_bar["close"])
    bb_upper = float(closed_bar["bb_upper"])
    bb_lower = float(closed_bar["bb_lower"])
    ema_200 = float(closed_bar["ema_200"])

    if curr_close > ema_200 and curr_close <= bb_lower:
        reason = (
            f"Uptrend Pullback: السعر ({curr_close:.{digits}f}) أغلق عند أو أسفل "
            f"البولينجر السفلي ({bb_lower:.{digits}f}) وأعلى EMA200 ({ema_200:.{digits}f})."
        )
        return "CALL", curr_close, reason

    if curr_close < ema_200 and curr_close >= bb_upper:
        reason = (
            f"Downtrend Rally: السعر ({curr_close:.{digits}f}) أغلق عند أو أعلى "
            f"البولينجر العلوي ({bb_upper:.{digits}f}) وأسفل EMA200 ({ema_200:.{digits}f})."
        )
        return "PUT", curr_close, reason

    return "WAIT", curr_close, "لا توجد إشارة دخول مطابقة لشروط الاستراتيجية حالياً"


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
    tf: str = Query("15m")
):
    symbol = symbol.strip().upper()
    if symbol not in TRADING_PAIRS:
        return JSONResponse(status_code=400, content={"error": f"زوج العملات غير مدعوم: {symbol}"})

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

    signal, _, reason = check_signal_conditions(df, digits)
    live_bar = df.iloc[-1]
    sr_levels = calculate_support_resistance(df, pip_mult)
    gaza_now = datetime.now(timezone.utc).astimezone(GAZA_TZ)

    return {
        "symbol": symbol,
        "timeframe": tf,
        "live_price": round(float(live_bar["close"]), digits),
        "signal": signal,
        "reason": reason,
        "gaza_time": gaza_now.strftime("%H:%M:%S"),
        "night_ban_active": False,
        "sr_levels": sr_levels,
        "candles": candles,
        "indicators": {
            "bb_upper": bb_upper_line,
            "bb_lower": bb_lower_line,
            "ema_200": ema200_line
        }
    }
