"""
bot.py
------
Core chatbot logic for the WhatsApp sales analytics bot.
This module processes user queries, interacts with the Gemini LLM,
executes SQL queries on the SQLite database, and formats responses.

Main responsibilities:
- Parse user intent
- Generate SQL queries using Gemini
- Fetch insights from SQLite sales database
- Return human-readable analytics responses
"""


import sqlite3
import pandas as pd
import re
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = "data/sales.db"
TABLE_NAME = "sales"
MODEL = "gpt-3.5-turbo"

MAX_ROWS = 5
MAX_TEXT_LEN = 28
DEFAULT_PERIOD = "90d"

conn = None
llm = None
COLUMNS = None

# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_context = {
    "last_intent": None,
    "last_dealer": None,
    "last_period": DEFAULT_PERIOD,
    "last_dataframe": None,
    "pagination_index": 0,
    "last_snapshot": None
}

# ============================================================
# GREETING + MENU (TEXT VERSION)
# ============================================================

def is_greeting(msg: str) -> bool:
    return msg.lower().strip() in ["hi", "hello", "hey", "start", "help", "menu"]

def menu_message() -> str:
    return (
        "👋 *Welcome!*\n\n"
        "What would you like to do?\n\n"
        "1️⃣ Dealer Snapshot\n"
        "2️⃣ Compare Dealers\n"
        "3️⃣ Top Dealers by Sales\n"
        "4️⃣ Sales by State\n"
        "5️⃣ Sales by Product\n\n"
        "Reply with a number or type your question."
    )

# ============================================================
# PERIOD PARSING
# ============================================================

def parse_period(msg: str) -> str:
    q = msg.lower()
    if "last month" in q:
        return "30d"
    if "last quarter" in q:
        return "90d"
    if "ytd" in q or "year to date" in q:
        return "ytd"
    return DEFAULT_PERIOD

def period_clause(period: str) -> str:
    if period == "ytd":
        return "Invoice Date >= date('now','start of year')"
    days = int(period.replace("d", ""))
    return f"Invoice Date >= date('now','-{days} day')"

# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(msg: str) -> str:
    q = msg.lower().strip()

    if q in ["1", "dealer snapshot"]:
        return "SNAPSHOT"
    if q in ["2", "compare dealers"]:
        return "COMPARE"
    if q in ["3", "top dealers"]:
        return "TOP_DEALERS"
    if q in ["4", "sales by state"]:
        return "STATE"
    if q in ["5", "sales by product"]:
        return "PRODUCT"
    if "compare" in q:
        return "COMPARE"
    if "dealer" in q:
        return "SNAPSHOT"

    return "ANALYTICS"

# ============================================================
# DEALER HELPERS
# ============================================================

def extract_dealers(msg: str) -> list:
    parts = re.split(r"and|vs", msg, flags=re.IGNORECASE)
    dealers = []
    for p in parts:
        if "dealer" in p.lower():
            dealers.append(p.lower().split("dealer", 1)[1].strip().title())
    return dealers

def fuzzy_dealers(name: str) -> list:
    df = pd.read_sql(
        "SELECT DISTINCT Dealer FROM sales WHERE Dealer LIKE ? LIMIT 5",
        conn,
        params=[f"%{name}%"]
    )
    return df["Dealer"].tolist()

# ============================================================
# DEALER SNAPSHOT + COMPARISON
# ============================================================

def dealer_snapshot(dealer: str, period: str) -> dict | None:
    query = f"""
    SELECT
        Dealer,
        State,
        SUM("Basic Amount") AS TotalSales,
        COUNT(*) AS Orders,
        MAX("Invoice Date") AS LastTxn,
        SUM("Outstanding Amount") AS Outstanding,
        AVG("Credit Delay Days") AS AvgDelay
    FROM sales
    WHERE Dealer = ?
      AND {period_clause(period)}
    """
    df = pd.read_sql(query, conn, params=[dealer])
    if df.empty or pd.isna(df.iloc[0]["Dealer"]):
        return None
    return df.iloc[0].to_dict()

def snapshot_previous_period(dealer: str, days: int) -> float:
    query = f"""
    SELECT SUM("Basic Amount") AS Sales
    FROM sales
    WHERE Dealer = ?
      AND Invoice Date < date('now','-{days} day')
      AND Invoice Date >= date('now','-{days*2} day')
    """
    df = pd.read_sql(query, conn, params=[dealer])
    return df.iloc[0]["Sales"] or 0

# ============================================================
# HEALTH SCORE + EXPLAINABILITY
# ============================================================

def health_score(snapshot: dict) -> tuple[int, str, list]:
    today = pd.Timestamp.today()
    days_since = (today - pd.to_datetime(snapshot["LastTxn"])).days

    rec = 100 if days_since <= 30 else 60 if days_since <= 60 else 20
    freq = 100 if snapshot["Orders"] >= 20 else 60 if snapshot["Orders"] >= 5 else 20
    mon = 100 if snapshot["TotalSales"] >= 1_000_000 else 60 if snapshot["TotalSales"] >= 500_000 else 20
    cred = 100 if snapshot["AvgDelay"] <= 7 else 60 if snapshot["AvgDelay"] <= 15 else 20

    score = round(rec*0.3 + freq*0.25 + mon*0.3 + cred*0.15)
    status = "🟢 Healthy" if score >= 80 else "🟡 Watch" if score >= 60 else "🔴 At Risk"

    reasons = []
    if rec < 60:
        reasons.append("Low recent activity")
    if freq < 60:
        reasons.append("Low order frequency")
    if mon < 60:
        reasons.append("Low sales contribution")
    if cred < 60:
        reasons.append("Delayed payments")

    return score, status, reasons

# ============================================================
# RANKING + TERRITORY INSIGHTS
# ============================================================

def dealer_rank(dealer: str, state: str) -> int:
    query = """
    SELECT Dealer
    FROM sales
    WHERE State = ?
    GROUP BY Dealer
    ORDER BY SUM("Basic Amount") DESC
    """
    df = pd.read_sql(query, conn, params=[state])
    ranked = df["Dealer"].tolist()
    return ranked.index(dealer) + 1 if dealer in ranked else -1

# ============================================================
# SNAPSHOT CARD (ENHANCED)
# ============================================================

def snapshot_card(snapshot: dict, score: int, status: str, reasons: list) -> str:
    prev_sales = snapshot_previous_period(snapshot["Dealer"], 30)
    delta = snapshot["TotalSales"] - prev_sales
    delta_text = f"⬆️ ₹{int(delta):,}" if delta >= 0 else f"⬇️ ₹{int(abs(delta)):,}"

    rank = dealer_rank(snapshot["Dealer"], snapshot["State"])

    followups = []
    if snapshot["AvgDelay"] > 15:
        followups.append("⚠️ Follow up on overdue payments")
    if score < 60:
        followups.append("📍 Plan a dealer visit")
    if delta < 0:
        followups.append("📉 Sales dropped vs last period")

    return f"""
📍 *Dealer Snapshot*

🏪 Dealer: *{snapshot['Dealer']}*
📍 Territory: {snapshot['State']}
🏆 Rank in State: #{rank}

💰 Sales: ₹{int(snapshot['TotalSales']):,} ({delta_text})
📦 Orders: {int(snapshot['Orders'])}
🕒 Last Order: {snapshot['LastTxn']}

💳 Outstanding: ₹{int(snapshot['Outstanding']):,}
⏱ Avg Delay: {int(snapshot['AvgDelay'])} days

{status} *Health Score*: {score}

🔍 *Why?*
• {"; ".join(reasons) if reasons else "Strong performance across KPIs"}

👉 *Suggested Actions*
• {"; ".join(followups) if followups else "Maintain engagement"}

You can ask:
• Compare with another dealer
• Dealers at risk
• Top dealers in this state
""".strip()

# ============================================================
# TABLE + PAGINATION
# ============================================================

def format_table(df: pd.DataFrame, start: int = 0) -> str:
    chunk = df.iloc[start:start+MAX_ROWS]
    header = " | ".join(chunk.columns)
    lines = [header, "-" * len(header)]

    for _, r in chunk.iterrows():
        lines.append(" | ".join(str(v)[:MAX_TEXT_LEN] for v in r))

    text = "```\n" + "\n".join(lines) + "\n```"
    if start + MAX_ROWS < len(df):
        text += "\n➡️ Type *show next* to see more"
    return text

# ============================================================
# SETUP
# ============================================================

def setup_bot():
    global conn, llm, COLUMNS
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 1", conn)
    COLUMNS = df.columns.tolist()
    llm = ChatOpenAI(model=MODEL, temperature=0)
    print("✅ Bot initialized")

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def search_answer(msg: str) -> str:

    if is_greeting(msg):
        return menu_message()

    if msg.lower().strip() == "show next" and conversation_context["last_dataframe"] is not None:
        conversation_context["pagination_index"] += MAX_ROWS
        return format_table(
            conversation_context["last_dataframe"],
            conversation_context["pagination_index"]
        )

    intent = detect_intent(msg)
    conversation_context["last_intent"] = intent
    conversation_context["last_period"] = parse_period(msg)

    # ---------------- DEALER SNAPSHOT ----------------

    if intent == "SNAPSHOT":
        dealers = extract_dealers(msg)
        dealer = dealers[0] if dealers else conversation_context["last_dealer"]

        if not dealer:
            return "Which dealer would you like to check?"

        matches = fuzzy_dealers(dealer)
        if not matches:
            return "Dealer not found."
        if len(matches) > 1:
            return "Did you mean:\n" + "\n".join(f"- {m}" for m in matches)

        dealer = matches[0]
        conversation_context["last_dealer"] = dealer

        snap = dealer_snapshot(dealer, conversation_context["last_period"])
        if not snap:
            return "No data available for this dealer."

        score, status, reasons = health_score(snap)
        return snapshot_card(snap, score, status, reasons)

    # ---------------- FREE TEXT ANALYTICS ----------------

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a data analyst. Table: {TABLE_NAME}. Columns: {COLUMNS}. "
         "Generate ONE SQL SELECT query only."),
        ("human", msg)
    ])

    try:
        sql = llm.invoke(prompt.format_messages()).content.replace("`", '"')
        df = pd.read_sql(sql, conn)

        if df.empty:
            return "No data found."

        conversation_context["last_dataframe"] = df
        conversation_context["pagination_index"] = 0
        return format_table(df)

    except Exception:
        return "I couldn’t understand that. Try using the menu or rephrasing."
