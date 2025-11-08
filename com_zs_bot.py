# com_zs_combo.py
# Telegram bot: Two strategies in one (PSAR+BB) and (AO+RSI)
# Timeframes: 30s or 1m only
# Optional real-candle feed via Finnhub (free key) else synthetic candles
# Lifetime unlock password via env UNLOCK_PASSWORD (default: "com.zs")

import os, asyncio, math, random, time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import aiohttp

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

# ====================== ENV ======================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
FINNHUB = os.getenv("FINNHUB_API_KEY", "").strip()  # optional
UNLOCK_PASSWORD = os.getenv("UNLOCK_PASSWORD", "com.zs").strip()
BRAND = "com_zs💳"

# ================= UTILS =================
def esc(text: str) -> str:
    for c in "_*`[]()~>#+-=|{}.!":
        text = text.replace(c, "\\" + c)
    return text

def ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

# Keep simple, limit choices to avoid spam
TIMEFRAMES = ["30s", "1m"]

# Your earlier OTC/stock picks + forex from slides
ASSETS = [
    # your 10 list
    "UKBrent (OTC)", "Silver (OTC)", "USCrude (OTC)", "Gold (OTC)",
    "NZD/CHF (OTC)", "NZD/CAD (OTC)", "EUR/AUD (OTC)", "EUR/USD (OTC)",
    "USD/ARS (OTC)", "MICROSOFT (OTC)",
    # slide assets (forex)
    "EUR/USD", "AUD/JPY", "USD/CAD", "GBP/USD", "GBP/AUD", "AUD/USD",
    "AUD/JPY", "EUR/JPY", "USD/JPY", "GBP/JPY"
]

# Per-user state and history
USER_UNLOCKED = set()
USER_STATE = {}  # chat_id -> dict(selection)
USER_HISTORY = defaultdict(lambda: deque(maxlen=20))

# ================= INDICATORS =================
def rsi(series, period=14):
    arr = np.asarray(series, dtype=float)
    if len(arr) < period + 1:
        return None
    delta = np.diff(arr)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = np.convolve(up, np.ones(period), 'valid') / period
    roll_down = np.convolve(down, np.ones(period), 'valid') / period
    if roll_down[-1] == 0:
        return 100.0
    rs = roll_up[-1] / roll_down[-1]
    return 100 - (100 / (1 + rs))

def bbands(series, period=20, std=2.0):
    arr = np.asarray(series, dtype=float)
    if len(arr) < period:
        return None, None, None
    sma = np.mean(arr[-period:])
    sd = np.std(arr[-period:], ddof=0)
    upper = sma + std * sd
    lower = sma - std * sd
    return lower, sma, upper

def awesome_oscillator(high, low, fast=5, slow=34):
    if len(high) < slow or len(low) < slow:
        return None, None
    median = (np.asarray(high) + np.asarray(low)) / 2.0
    sma_fast = np.convolve(median, np.ones(fast)/fast, mode='valid')
    sma_slow = np.convolve(median, np.ones(slow)/slow, mode='valid')
    # align to the end
    ao_series = sma_fast[-(len(sma_slow)):] - sma_slow
    return ao_series[-1], ao_series

def psar(high, low, step=0.02, max_step=0.2):
    # Very compact PSAR for last value only (not full series)
    if len(high) < 5 or len(low) < 5:
        return None
    trend_up = True
    af = step
    ep = high[0]
    sar = low[0]
    for i in range(1, len(high)):
        prev_sar = sar
        if trend_up:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low[i-1], low[i-2] if i >= 2 else low[i-1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < sar:
                trend_up = False
                sar = ep
                ep = low[i]
                af = step
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high[i-1], high[i-2] if i >= 2 else high[i-1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > sar:
                trend_up = True
                sar = ep
                ep = high[i]
                af = step
    return sar, trend_up  # last sar and current trend flag

# ================= DATA FEEDS =================
async def fetch_candles_finnhub(session, symbol: str, tf: str):
    # Map tf to seconds
    res_map = {"30s": 30, "1m": 60}
    resolution = res_map.get(tf, 60)

    # Very basic tickers mapping
    fin_sym = symbol.replace(" (OTC)", "").replace("/", "")
    url = (
        f"https://finnhub.io/api/v1/crypto/candle?symbol=BINANCE:{fin_sym}USDT"
        f"&resolution={resolution}&count=200&token={FINNHUB}"
    )
    # NOTE: This mapping is simplistic; real broker symbols differ.
    # If this returns empty, we’ll fall back to synthetic data.

    try:
        async with session.get(url, timeout=8) as r:
            j = await r.json()
            if not j or j.get("s") != "ok":
                return None
            close = j["c"]
            high = j["h"]
            low = j["l"]
            return close, high, low
    except:
        return None

def synth_candles(n=200, start=100.0):
    close = []
    high = []
    low = []
    last = start
    for _ in range(n):
        drift = random.uniform(-0.3, 0.3)
        vol = random.uniform(0.05, 0.5)
        price = max(0.1, last + drift)
        c = round(price, 4)
        h = round(c + vol, 4)
        l = round(max(0.1, c - vol), 4)
        close.append(c); high.append(h); low.append(l)
        last = c
    return close, high, low

async def get_candles(symbol: str, tf: str):
    if FINNHUB:
        async with aiohttp.ClientSession() as s:
            data = await fetch_candles_finnhub(s, symbol, tf)
            if data:
                return data
    # fallback
    seed = abs(hash(symbol + tf)) % 100 + 50
    return synth_candles(200, start=float(seed))

# ================= RULES =================
def signal_psar_bb(close, high, low):
    # BB(20,2); PSAR(0.02→0.2); midline cross logic
    if len(close) < 35:
        return None, {}
    lower, mid, upper = bbands(close, 20, 2.0)
    ps, trend_up = psar(high, low, 0.02, 0.2)
    if lower is None or ps is None:
        return None, {}

    last = close[-1]; prev = close[-2]
    mid_cross_up = (prev < mid) and (last >= mid)
    mid_cross_down = (prev > mid) and (last <= mid)

    if trend_up and mid_cross_up:
        side = "BUY"
    elif (not trend_up) and mid_cross_down:
        side = "SELL"
    else:
        side = None

    debug = {
        "PSAR_trend_up": trend_up,
        "BB_mid": round(mid, 4),
        "Close": round(last, 4)
    }
    return side, debug

def signal_ao_rsi(close, high, low):
    # RSI 14 centerline 50; AO fast 5 / slow 34 with zero-line cross
    if len(close) < 50:
        return None, {}
    r = rsi(close, 14)
    ao_last, ao_series = awesome_oscillator(high, low, 5, 34)
    if r is None or ao_last is None:
        return None, {}

    # AO zero-line cross check (last two bars)
    if len(ao_series) < 2:
        return None, {}
    prev_ao = ao_series[-2]
    ao_cross_up = (prev_ao <= 0) and (ao_last > 0)
    ao_cross_down = (prev_ao >= 0) and (ao_last < 0)

    # RSI centerline cross (last two closes)
    # compute RSI on last two windows
    r_prev = rsi(close[:-1], 14)
    r_cross_up = (r_prev is not None and r_prev <= 50 and r > 50)
    r_cross_down = (r_prev is not None and r_prev >= 50 and r < 50)

    if r_cross_up and ao_cross_up:
        side = "BUY"
    elif r_cross_down and ao_cross_down:
        side = "SELL"
    else:
        side = None

    debug = {"RSI": round(r, 2), "AO": round(float(ao_last), 5)}
    return side, debug

# =============== TELEGRAM FLOW ===============
def kb(rows): return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in rows])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_chat.id
    USER_STATE[uid] = {}
    rows = [
        [("🔓 Unlock Premium", "unlock")],
        [("🧠 Strategy A: PSAR + BB", "strat_psar_bb")],
        [("🧠 Strategy B: AO + RSI", "strat_ao_rsi")]
    ]
    await update.message.reply_text(
        esc(f"{BRAND} — Choose a strategy or unlock with lifetime password."),
        reply_markup=kb(rows)
    )

async def unlock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.reply_text(esc("Send password here (single message):"))
    context.user_data["awaiting_pass"] = True

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_chat.id
    if context.user_data.get("awaiting_pass"):
        context.user_data["awaiting_pass"] = False
        if update.message.text.strip() == UNLOCK_PASSWORD:
            USER_UNLOCKED.add(uid)
            await update.message.reply_text(esc("✅ Unlocked for lifetime."))
        else:
            await update.message.reply_text(esc("❌ Wrong password."))
        return

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; uid = q.message.chat.id
    data = q.data

    if data == "unlock":
        await unlock(update, context); return

    state = USER_STATE.setdefault(uid, {})
    if data in ("strat_psar_bb", "strat_ao_rsi"):
        state["strategy"] = data
        # assets menu (paged small)
        rows = []
        for a in ASSETS[:10]:
            rows.append([(a, f"asset::{a}")])
        rows.append([("More assets ▶️", "assets_more")])
        await q.answer()
        await q.edit_message_text(
            esc("Select asset:"), reply_markup=kb(rows)
        )
        return

    if data == "assets_more":
        rows = []
        for a in ASSETS[10:]:
            rows.append([(a, f"asset::{a}")])
        rows.append([("◀️ Back", "assets_back")])
        await q.answer()
        await q.edit_message_text(esc("Select asset (more):"), reply_markup=kb(rows))
        return

    if data == "assets_back":
        rows = []
        for a in ASSETS[:10]:
            rows.append([(a, f"asset::{a}")])
        rows.append([("More assets ▶️", "assets_more")])
        await q.answer()
        await q.edit_message_text(esc("Select asset:"), reply_markup=kb(rows))
        return

    if data.startswith("asset::"):
        asset = data.split("::", 1)[1]
        state["asset"] = asset
        rows = [[("30s", "tf::30s"), ("1m", "tf::1m")]]
        await q.answer()
        await q.edit_message_text(
            esc(f"Asset: {asset}\nPick timeframe:"), reply_markup=kb(rows)
        )
        return

    if data.startswith("tf::"):
        tf = data.split("::", 1)[1]
        if tf not in TIMEFRAMES:
            await q.answer("Invalid timeframe"); return
        state["tf"] = tf
        rows = [[("⚡ Generate Signal", "gen")], [("Change asset", "assets_back")]]
        await q.answer()
        await q.edit_message_text(
            esc(f"Strategy: {state['strategy']} | Asset: {state['asset']} | TF: {tf}\n"
                "Tap to generate ONE signal."),
            reply_markup=kb(rows)
        )
        return

    if data == "gen":
        await q.answer()
        st = USER_STATE.get(uid, {})
        if not st.get("asset") or not st.get("tf") or not st.get("strategy"):
            await q.edit_message_text(esc("Missing selection. Use /start.")); return

        # Require unlock for signals
        if uid not in USER_UNLOCKED:
            await q.edit_message_text(esc("🔒 Premium only. Tap Unlock or send password.")); return

        asset = st["asset"]; tf = st["tf"]; strategy = st["strategy"]

        # Load candles
        close, high, low = await get_candles(asset, tf)

        if strategy == "strat_psar_bb":
            side, dbg = signal_psar_bb(close, high, low)
            strat_name = "PSAR + BB (20,2)"
            expiry_hint = "≤ 5 minutes"  # from slide
        else:
            side, dbg = signal_ao_rsi(close, high, low)
            strat_name = "AO(5/34) + RSI(14)"
            expiry_hint = "1 minute"

        if not side:
            msg = (f"⏳ No clean setup right now.\n"
                   f"Strategy: {strat_name}\nAsset: {asset} | TF: {tf}\n"
                   f"Try again later.")
            await q.edit_message_text(esc(msg))
            return

        signal_msg = (
            f"🟦 {BRAND} — Auto Signal\n"
            f"• Strategy: {strat_name}\n"
            f"• Asset: {asset}\n"
            f"• TF: {tf}\n"
            f"• Direction: *{side}*\n"
            f"• Suggested Expiry: {expiry_hint}\n"
            f"• Time: {ts()}\n"
            f"• Notes: {dbg}\n"
            f"_Powered by {BRAND}_"
        )
        USER_HISTORY[uid].appendleft({
            "time": ts(), "asset": asset, "tf": tf,
            "strategy": strat_name, "side": side
        })

        rows = [[("Next signal", "gen")], [("Change timeframe", f"tf::{tf}"), ("Change asset", "assets_back")]]
        await q.edit_message_text(esc(signal_msg), reply_markup=kb(rows))
        return

async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_chat.id
    hist = list(USER_HISTORY[uid])
    if not hist:
        await update.message.reply_text(esc("No history yet."))
        return
    lines = []
    for h in hist[:20]:
        lines.append(f"{h['time']} | {h['asset']} {h['tf']} | {h['strategy']} | {h['side']}")
    await update.message.reply_text(esc("Last signals:\n" + "\n".join(lines)))

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        f"{BRAND} — How to use:\n"
        "1) /start → pick Strategy → pick Asset → pick TF (30s or 1m)\n"
        "2) Tap ⚡ Generate Signal to get ONE signal\n"
        "3) /history to view last 20 signals\n"
        "Unlock (lifetime): send password when asked.\n"
        "Optional real data: add FINNHUB_API_KEY env.\n"
        "Educational only. No guarantees."
    )
    await update.message.reply_text(esc(txt))

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(CommandHandler("unlock", help_cmd))
    app.add_handler(CommandHandler("signal", help_cmd))
    app.add_handler(CommandHandler("menu", start))
    app.add_handler(CommandHandler("his", history_cmd))
    app.add_handler(CommandHandler("dash", history_cmd))
    app.add_handler(CommandHandler("upgrade", help_cmd))
    app.add_handler(CommandHandler("unlock_com_zs", help_cmd))
    app.add_handler(CommandHandler("premium", help_cmd))
    app.add_handler(CommandHandler("about", help_cmd))
    app.add_handler(CommandHandler("support", help_cmd))
    app.add_handler(CommandHandler("id", help_cmd))
    app.add_handler(CommandHandler("ping", help_cmd))
    app.add_handler(CommandHandler("helpme", help_cmd))
    app.add_handler(CommandHandler("ps", help_cmd))
    app.add_handler(CommandHandler("aorsi", help_cmd))
    app.add_handler(CommandHandler("psar", help_cmd))
    app.add_handler(CommandHandler("bb", help_cmd))
    app.add_handler(CommandHandler("ao", help_cmd))
    app.add_handler(CommandHandler("rsi", help_cmd))
    app.add_handler(CommandHandler("assets", start))
    app.add_handler(CommandHandler("tf", start))
    app.add_handler(CommandHandler("next", help_cmd))
    app.add_handler(CommandHandler("restart", start))
    app.add_handler(CommandHandler("stop", help_cmd))
    app.add_handler(CommandHandler("terms", help_cmd))
    app.add_handler(CommandHandler("privacy", help_cmd))
    app.add_handler(CommandHandler("license", help_cmd))
    app.add_handler(CommandHandler("contact", help_cmd))
    app.add_handler(CommandHandler("owner", help_cmd))
    app.add_handler(CommandHandler("admin", help_cmd))
    app.add_handler(CommandHandler("version", help_cmd))
    app.add_handler(CommandHandler("v", help_cmd))
    app.add_handler(CommandHandler("info", help_cmd))
    app.add_handler(CommandHandler("status", help_cmd))
    app.add_handler(CommandHandler("startagain", start))
    app.add_handler(CommandHandler("clear", help_cmd))
    app.add_handler(CommandHandler("reset", start))
    app.add_handler(CommandHandler("guide", help_cmd))
    app.add_handler(CommandHandler("how", help_cmd))
    app.add_handler(CommandHandler("tutorial", help_cmd))
    app.add_handler(CommandHandler("faq", help_cmd))
    app.add_handler(CommandHandler("cmds", help_cmd))
    app.add_handler(CommandHandler("commands", help_cmd))
    app.add_handler(CommandHandler("help_", help_cmd))
    app.add_handler(CommandHandler("menu_", start))
    # text handler for password
    app.add_handler(CommandHandler("unlocknow", help_cmd))
    app.add_handler(CommandHandler("premiumnow", help_cmd))
    app.add_handler(CommandHandler("history_", history_cmd))
    app.add_handler(CommandHandler("dashboard", history_cmd))
    app.add_handler(CommandHandler("signals", help_cmd))
    app.add_handler(CommandHandler("generate", help_cmd))
    # Any plain text (for password entry)
    from telegram.ext import MessageHandler, filters
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print(f"✅ {BRAND} bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
