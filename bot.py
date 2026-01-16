"""
bot.py
------
Core chatbot logic for the WhatsApp sales analytics bot.
This module processes user queries, interacts with the LLM,
executes SQL queries on the SQLite database, and formats responses.
"""

import sqlite3
import pandas as pd
import re
import yaml
import json
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

LOGS_FILE = "data/bot_logs.jsonl"

# ============================================================
# CONVERSATION MEMORY
# ============================================================

DEFAULT_CONTEXT = {
    "last_intent": None,
    "last_dealer": None,
    "last_period": DEFAULT_PERIOD,
    "last_dataframe": None,
    "pagination_index": 0,
    "last_snapshot": None,
    "disambiguation_options": [],
    "awaiting_selection": False
}

# Global single-user context (backward compatibility)
conversation_context = DEFAULT_CONTEXT.copy()

# Multi-user context storage
conversation_contexts = {}

def get_context(user_id):
    """Get conversation context for a specific user"""
    if user_id not in conversation_contexts:
        conversation_contexts[user_id] = DEFAULT_CONTEXT.copy()
    return conversation_contexts[user_id]

# ============================================================
# GREETING + MENU (TEXT VERSION)
# ============================================================

def is_greeting(msg: str) -> bool:
    return msg.lower().strip() in ["hi", "hello", "hey", "start", "help", "menu"]

def menu_message() -> str:
    return (
        "👋 *Welcome!*\n\n"
        "I can help you quickly analyze dealer performance and sales data.\n\n"
        "What would you like to do?\n\n"
        "1️⃣ Dealer Snapshot\n"
        "2️⃣ Compare Dealers\n"
        "3️⃣ Top Dealers by Sales\n"
        "4️⃣ Sales by State\n"
        "5️⃣ Sales by Product\n\n"
        "You can also ask questions like:\n"
        "• Dealers with overdue payments\n"
        "• Sales trend last quarter\n"
        "• Average credit delay by state"
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
# EXTERNAL INTENT PATTERNS (YAML-BASED)
# ============================================================

INTENT_PATTERNS = {}

def load_intents(path: str = "intents.yaml"):
    """Load intent patterns from external YAML file"""
    global INTENT_PATTERNS
    try:
        with open(path, "r") as f:
            INTENT_PATTERNS = yaml.safe_load(f)
        print(f"✅ Loaded intents from {path}")
        print(f"   Intents available: {list(INTENT_PATTERNS.keys())}")
        return True
    except FileNotFoundError:
        print(f"⚠️ Intent file not found: {path}")
        print("   Using default inline patterns")
        load_default_intents()
        return False
    except Exception as e:
        print(f"❌ Error loading intents: {e}")
        load_default_intents()
        return False

def load_default_intents():
    """Fallback to hardcoded intent patterns"""
    global INTENT_PATTERNS
    INTENT_PATTERNS = {
        "SNAPSHOT": {
            "exact": ["1", "dealer snapshot"],
            "keywords": ["snapshot", "dealer", "check dealer", "dealer performance"],
            "phrases": ["tell me about", "dealer named", "dealer info"],
            "regex": [r"dealer\s+(\w+)", r"check\s+(\w+)"]
        },
        "COMPARE": {
            "exact": ["2", "compare dealers"],
            "keywords": ["compare", "vs", "versus", "vs."],
            "phrases": ["compare with", "compare dealers", "dealer vs"],
            "regex": [r"(\w+)\s+vs\.?\s+(\w+)", r"compare\s+(.+?)\s+(?:and|with)\s+(.+?)"]
        },
        "TOP_DEALERS": {
            "exact": ["3", "top dealers"],
            "keywords": ["top", "best", "highest", "leading", "best performing"],
            "phrases": ["top dealers", "top 5", "top performers", "best dealers"],
            "regex": [r"top\s+\d+\s+dealers", r"best\s+\d+\s+dealers"]
        },
        "STATE": {
            "exact": ["4", "sales by state"],
            "keywords": ["state", "territory", "region", "by state"],
            "phrases": ["sales by state", "state wise", "state-wise", "state breakdown"],
            "regex": [r"sales\s+by\s+state", r"state\s+wise"]
        },
        "PRODUCT": {
            "exact": ["5", "sales by product"],
            "keywords": ["product", "for coils", "for tmt", "for bars", "for tg", "product-wise"],
            "phrases": ["top dealers for", "sales by product", "dealers for", "product sales"],
            "regex": [r"(?:for|of)\s+([a-z\s]+?)(?:\s+dealers|\s+sales?|\s+by|\?|$)", 
                     r"top\s+\d+\s+dealers\s+for"]
        },
        "RISK_ANALYSIS": {
            "exact": [],
            "keywords": ["at risk", "risk", "overdue", "underperforming", "struggling"],
            "phrases": ["dealers at risk", "dealers with overdue", "at risk dealers"],
            "regex": [r"dealers\s+(?:at\s+)?risk", r"overdue\s+(?:\d+\s+)?days?"]
        },
        "DASHBOARD": {
            "exact": [],
            "keywords": ["dashboard", "summary", "overview", "summary statistics"],
            "phrases": ["sales dashboard", "quick summary", "overall performance"],
            "regex": [r"(?:show|display)\s+dashboard", r"(?:quick\s+)?summary"]
        },
        "ANALYTICS": {
            "exact": [],
            "keywords": ["trend", "analysis", "analytics", "insights", "average", "total"],
            "phrases": ["sales trend", "last quarter", "ytd", "year to date"],
            "regex": [r"average\s+\w+\s+by", r"total\s+\w+\s+by"]
        }
    }
    print("✅ Using default intent patterns")

# ============================================================
# INTENT DETECTION
# ============================================================

def calculate_intent_score(msg: str, intent_patterns: dict = None) -> dict:
    """Score message against all intent patterns"""
    if intent_patterns is None:
        intent_patterns = INTENT_PATTERNS
    
    q = msg.lower().strip()
    scores = {}
    
    for intent, patterns in intent_patterns.items():
        score = 0
        
        # Exact match (highest priority)
        if q in patterns.get("exact", []):
            score += 100
        
        # Keyword matches
        for keyword in patterns.get("keywords", []):
            if keyword in q:
                score += 10
        
        # Phrase matches (substring)
        for phrase in patterns.get("phrases", []):
            if phrase in q:
                score += 5
        
        # Regex matches
        for regex_pattern in patterns.get("regex", []):
            try:
                if re.search(regex_pattern, q, re.IGNORECASE):
                    score += 7
            except re.error as e:
                print(f"⚠️ Invalid regex pattern: {regex_pattern} - {e}")
        
        scores[intent] = score
    
    return scores

def detect_intent(msg: str) -> str:
    """Enhanced intent detection with scoring using external patterns"""
    
    # SKIP intent detection for simple numbers if awaiting selection
    if conversation_context["awaiting_selection"] and msg.strip().isdigit():
        return "SKIP"  # Will be handled by selection logic above
    
    scores = calculate_intent_score(msg)
    
    # Filter out zero scores
    valid_scores = {k: v for k, v in scores.items() if v > 0}
    
    if not valid_scores:
        return "ANALYTICS"  # Default fallback
    
    # Return highest scoring intent
    best_intent = max(valid_scores, key=valid_scores.get)
    best_score = valid_scores[best_intent]
    
    # Debug: Log top matches
    top_3 = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"🎯 Intent: {best_intent} (score: {best_score}), top 3: {top_3}")
    
    return best_intent

def handle_intent_with_context(intent: str, msg: str) -> str:
    """Route intent to appropriate handler"""
    
    # Store intent in context for follow-ups
    conversation_context["last_intent"] = intent
    conversation_context["last_period"] = parse_period(msg)
    
    if intent == "SNAPSHOT":
        dealers = extract_dealers(msg)
        dealer = dealers[0] if dealers else conversation_context["last_dealer"]

        if not dealer:
            return "❓ Which dealer would you like to check?"

        matches = fuzzy_dealers(dealer)
        
        if not matches:
            return f"❌ Dealer '{dealer}' not found in database."
        
        if len(matches) == 1:
            # Single match - get snapshot directly
            dealer = matches[0]
            conversation_context["last_dealer"] = dealer
            snap = dealer_snapshot(dealer, conversation_context["last_period"])
            
            if not snap:
                return f"❌ No data available for {dealer}. Try another dealer."
            
            score, status, reasons = health_score(snap)
            response = snapshot_card(snap, score, status, reasons)
        else:
            # Multiple matches - show numbered list
            conversation_context["disambiguation_options"] = matches
            conversation_context["awaiting_selection"] = True
            conversation_context["last_intent"] = "SNAPSHOT"
            
            options_text = "Did you mean:\n"
            for i, d in enumerate(matches, 1):
                options_text += f"{i}️⃣ {d}\n"
            options_text += "\n📌 Reply with the number (e.g., 1, 2, 3)"
            response = options_text

    elif intent == "COMPARE":
        dealers = extract_dealers(msg)
        
        # Only proceed if we found dealers in the message OR we have a previous dealer AND user said compare
        if not dealers and not (conversation_context["last_dealer"] and "compare" in msg.lower()):
            return "❓ Please tell me which dealers to compare (e.g., *Dealer A vs Dealer B*)"
        
        if conversation_context["last_dealer"] and len(dealers) == 1:
            dealers.insert(0, conversation_context["last_dealer"])

        if len(dealers) < 2:
            return "❓ Please specify at least 2 dealers to compare (e.g., *A vs B*)"

        resolved = []
        for d in dealers[:3]:
            matches = fuzzy_dealers(d)
            if matches:
                resolved.append(matches[0])

        if len(resolved) < 2:
            return f"❌ Could not find enough matching dealers."
        
        response = compare_dealers(resolved, conversation_context["last_period"])

    elif intent == "TOP_DEALERS":
        response = top_dealers_overall(conversation_context["last_period"])

    elif intent == "STATE":
        response = sales_by_state(conversation_context["last_period"])

    elif intent == "PRODUCT":
        products = re.findall(r"(?:for|of)\s+([^\.,\?!]+?)(?:\s+dealers|\s+sales?|\s+by|\?|$)", msg, re.IGNORECASE)
        
        if not products:
            match = re.search(r"(?:for|of)\s+([^\.,\?!]+?)(?:\s|$)", msg, re.IGNORECASE)
            if match:
                products = [match.group(1)]
        
        if products:
            product_name = products[0].strip()
            matches = fuzzy_products(product_name)
            
            if not matches:
                available = get_unique_products(5)
                return f"❌ Product '{product_name}' not found.\n\n📦 Available products:\n" + "\n".join(f"  • {p}" for p in available)
            
            # ✅ FIX: Always return top dealers for the matched product(s)
            # Use the first (best) match directly instead of showing disambiguation list
            response = top_dealers_by_product(matches[0], conversation_context["last_period"])
        else:
            return "❓ Please specify a product (e.g., 'Top 5 dealers for coils')"

    elif intent == "RISK_ANALYSIS":
        response = dealers_at_risk(conversation_context["last_period"])

    elif intent == "DASHBOARD":
        response = summary_dashboard(conversation_context["last_period"])

    else:  # ANALYTICS
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"You are a helpful sales analytics assistant. "
                f"Greet politely when needed. "
                f"Help the user with analytics questions. "
                f"Table: {TABLE_NAME}. Columns: {COLUMNS}. "
                f"Generate ONE SQL SELECT query only. "
                f"Never modify data."
            ),
            ("human", msg)
        ])

        try:
            sql = llm.invoke(prompt.format_messages()).content.replace("`", '"')

            if not is_safe_sql(sql):
                return (
                    "I can help with analytics questions such as:\n"
                    "• Dealer performance\n"
                    "• Sales trends\n"
                    "• Comparisons and summaries"
                )

            df = pd.read_sql(sql, conn)

            if df.empty:
                return "No data found."

            conversation_context["last_dataframe"] = df
            conversation_context["pagination_index"] = 0
            response = format_table(df)

        except Exception as e:
            print(f"Analytics error: {e}")
            response = "I couldn't understand that. Try using the menu or rephrasing."

    return response

def search_answer(msg: str) -> str:
    if is_greeting(msg):
        return menu_message()

    if msg.lower().strip() == "show next" and conversation_context["last_dataframe"] is not None:
        conversation_context["pagination_index"] += MAX_ROWS
        return format_table(
            conversation_context["last_dataframe"],
            conversation_context["pagination_index"]
        )

    # PRIORITY 1: Check if user is selecting from a disambiguation list (ONLY for numbers when awaiting)
    if conversation_context["awaiting_selection"] and msg.strip().isdigit():
        choice_num = int(msg.strip()) - 1
        
        if 0 <= choice_num < len(conversation_context["disambiguation_options"]):
            selected = conversation_context["disambiguation_options"][choice_num]
            conversation_context["awaiting_selection"] = False
            last_intent_before_selection = conversation_context["last_intent"]
            conversation_context["disambiguation_options"] = []
            
            # Handle different selection types
            if last_intent_before_selection == "PRODUCT_SELECT":
                response = top_dealers_by_product(selected, conversation_context["last_period"])
                log_conversation(msg, response, "PRODUCT_SELECTION")
                return response
            else:  # SNAPSHOT_SELECT - show dealer snapshot
                conversation_context["last_dealer"] = selected
                snap = dealer_snapshot(selected, conversation_context["last_period"])
                if not snap:
                    return f"❌ No data available for {selected}."
                
                score, status, reasons = health_score(snap)
                response = snapshot_card(snap, score, status, reasons)
                log_conversation(msg, response, "SNAPSHOT_SELECTION")
                return response
        else:
            return f"❌ Please select a valid option (1-{len(conversation_context['disambiguation_options'])})"

    # PRIORITY 2: Use enhanced intent detection
    intent = detect_intent(msg)
    conversation_context["last_intent"] = intent
    response = handle_intent_with_context(intent, msg)

    # Log conversation
    log_conversation(msg, response, intent)
    return response

def setup_bot():
    global conn, llm, COLUMNS
    
    # Load intent patterns from external YAML file
    load_intents("intents.yaml")
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 1", conn)
    COLUMNS = df.columns.tolist()
    llm = ChatOpenAI(model=MODEL, temperature=0)
    print("✅ Bot initialized")

# ============================================================
# MESSAGE HANDLER (FOR FLASK APP)
# ============================================================

def handle_message(msg: str) -> tuple[str, None]:
    """
    Wrapper function for the Flask app.
    Takes a message string and returns (response_text, extra_data).
    """
    response = search_answer(msg)
    return response, None

def log_conversation(msg: str, response: str, intent: str):
    """Log user queries and bot responses"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": msg,
            "intent": intent,
            "response": response[:100],  # First 100 chars
        }
        with open(LOGS_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")

# ============================================================
# DEALER HELPERS
# ============================================================

def extract_dealers(msg: str) -> list:
    """Extract dealer names from message using regex"""
    parts = re.split(r"and|vs", msg, flags=re.IGNORECASE)
    dealers = []
    for p in parts:
        if "dealer" in p.lower():
            dealers.append(p.lower().split("dealer", 1)[1].strip().title())
    return dealers

def fuzzy_dealers(name: str) -> list:
    """Find dealers matching a search term"""
    df = pd.read_sql(
        "SELECT DISTINCT Dealer FROM sales WHERE Dealer LIKE ? LIMIT 5",
        conn,
        params=[f"%{name}%"]
    )
    return df["Dealer"].tolist()

def format_dealer_options(dealers: list) -> tuple[str, bool]:
    """Format dealer options as numbered list and store in context"""
    if not dealers:
        return "Dealer not found.", False
    
    if len(dealers) == 1:
        return dealers[0], False  # Single match, return dealer name directly
    
    # Multiple matches - format as numbered list
    conversation_context["disambiguation_options"] = dealers
    conversation_context["awaiting_selection"] = True
    
    options_text = "Did you mean:\n"
    for i, dealer in enumerate(dealers, 1):
        options_text += f"{i}️⃣ {dealer}\n"
    
    options_text += "\nReply with the number of your choice (e.g., 1, 2, 3)"
    
    return options_text, True

# ============================================================
# PRODUCT INSIGHTS
# ============================================================

def get_unique_products(limit: int = 10) -> list:
    """Fetch distinct products from the database"""
    df = pd.read_sql(
        "SELECT DISTINCT Product FROM sales LIMIT ?",
        conn,
        params=[limit]
    )
    return df["Product"].tolist()

def fuzzy_products(name: str) -> list:
    """Find products matching a search term"""
    df = pd.read_sql(
        "SELECT DISTINCT Product FROM sales WHERE Product LIKE ? LIMIT 5",
        conn,
        params=[f"%{name}%"]
    )
    return df["Product"].tolist()

def top_dealers_by_product(product: str, period: str, limit: int = 5) -> str:
    """Get top dealers for a specific product"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        Dealer,
        SUM("Basic Amount") as TotalSales,
        COUNT(*) as Orders
    FROM {TABLE_NAME}
    WHERE Product LIKE ? AND {period_filter}
    GROUP BY Dealer
    ORDER BY TotalSales DESC
    LIMIT ?
    """
    
    df = pd.read_sql(query, conn, params=[f"%{product}%", limit])
    
    if df.empty:
        return f"No dealers found for product '{product}'."
    
    return format_table(df)

def top_dealers_by_product_aggregate(products: list, period: str, limit: int = 5) -> str:
    """Get top dealers across multiple product variations"""
    period_filter = period_clause(period)
    
    # Build WHERE clause for multiple products
    product_conditions = " OR ".join([f"Product LIKE '{p}%'" for p in products])
    
    query = f"""
    SELECT 
        Dealer,
        SUM("Basic Amount") as TotalSales,
        COUNT(*) as Orders
    FROM {TABLE_NAME}
    WHERE ({product_conditions}) AND {period_filter}
    GROUP BY Dealer
    ORDER BY TotalSales DESC
    LIMIT ?
    """
    
    try:
        df = pd.read_sql(query, conn, params=[limit])
        if df.empty:
            return f"No dealers found for products: {', '.join(products)}."
        return format_table(df)
    except Exception as e:
        print(f"Error in aggregate product query: {e}")
        return f"Could not retrieve dealers for {products[0]}."

# ============================================================
# DEALER SNAPSHOT (DATA RETRIEVAL)
# ============================================================

def dealer_snapshot(dealer: str, period: str) -> dict:
    """Get sales snapshot for a specific dealer"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        '{dealer}' as Dealer,
        (SELECT DISTINCT State FROM {TABLE_NAME} WHERE Dealer = ? LIMIT 1) as State,
        SUM("Basic Amount") as TotalSales,
        COUNT(*) as Orders,
        SUM(CASE WHEN "Outstanding Amount" > 0 THEN "Outstanding Amount" ELSE 0 END) as Outstanding,
        ROUND(AVG(CASE WHEN "Outstanding Amount" > 0 THEN (julianday('now') - julianday("Invoice Date")) ELSE 0 END), 0) as AvgDelay,
        MAX("Invoice Date") as LastTxn
    FROM {TABLE_NAME}
    WHERE Dealer = ? AND {period_filter}
    """
    
    try:
        df = pd.read_sql(query, conn, params=[dealer, dealer])
        if df.empty:
            return None
        
        result = df.iloc[0].to_dict()
        # Ensure numeric fields are valid
        result["TotalSales"] = result["TotalSales"] or 0
        result["Orders"] = result["Orders"] or 0
        result["Outstanding"] = result["Outstanding"] or 0
        result["AvgDelay"] = result["AvgDelay"] or 0
        return result
    except Exception as e:
        print(f"Error fetching dealer snapshot: {e}")
        return None

def snapshot_previous_period(dealer: str, days: int) -> float:
    """Get total sales for previous period"""
    query = f"""
    SELECT SUM("Basic Amount") as TotalSales
    FROM {TABLE_NAME}
    WHERE Dealer = ? 
    AND Invoice Date >= date('now', '-{days*2} day')
    AND Invoice Date < date('now', '-{days} day')
    """
    
    try:
        df = pd.read_sql(query, conn, params=[dealer])
        return df.iloc[0]["TotalSales"] or 0 if not df.empty else 0
    except Exception:
        return 0

# ============================================================
# HEALTH SCORE + EXPLAINABILITY
# ============================================================

def health_score(snapshot: dict) -> tuple[int, str, list]:
    """Calculate health score for a dealer"""
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
    """Get dealer's rank within their state"""
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
    """Format detailed dealer snapshot response"""
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
# DEALER COMPARISON (FULL INTENT)
# ============================================================

def compare_dealers(dealers: list, period: str) -> str:
    """Compare two or more dealers"""
    rows = []

    for dealer in dealers:
        snap = dealer_snapshot(dealer, period)
        if snap:
            score, _, _ = health_score(snap)
            rows.append({
                "Dealer": dealer,
                "Sales": int(snap["TotalSales"]),
                "Orders": int(snap["Orders"]),
                "Outstanding": int(snap["Outstanding"]),
                "AvgDelay": int(snap["AvgDelay"]),
                "HealthScore": score
            })

    if len(rows) < 2:
        return "Please specify at least *two valid dealers* to compare."

    df = pd.DataFrame(rows)

    best_sales = df.loc[df["Sales"].idxmax()]["Dealer"]
    best_health = df.loc[df["HealthScore"].idxmax()]["Dealer"]

    return (
        format_table(df) +
        f"\n🏆 *Highest Sales*: {best_sales}\n"
        f"💚 *Best Health Score*: {best_health}"
    )

# ============================================================
# TABLE + PAGINATION
# ============================================================

def format_table(df: pd.DataFrame, start: int = 0) -> str:
    """Format DataFrame as WhatsApp-compatible table"""
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
# SQL SAFETY
# ============================================================

def is_safe_sql(sql: str) -> bool:
    """Validate SQL query for safety"""
    s = sql.strip().lower()
    return (
        s.startswith("select")
        and not any(k in s for k in ["insert", "update", "delete", "drop", "alter"])
        and ";" not in s
    )

# ============================================================
# TOP DEALERS HANDLER
# ============================================================

def top_dealers_overall(period: str, limit: int = 5) -> str:
    """Get top dealers by sales overall"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        Dealer,
        SUM("Basic Amount") as TotalSales,
        COUNT(*) as Orders,
        SUM(CASE WHEN "Outstanding Amount" > 0 THEN "Outstanding Amount" ELSE 0 END) as Outstanding
    FROM {TABLE_NAME}
    WHERE {period_filter}
    GROUP BY Dealer
    ORDER BY TotalSales DESC
    LIMIT ?
    """
    
    try:
        df = pd.read_sql(query, conn, params=[limit])
        if df.empty:
            return "No dealer data available."
        return format_table(df)
    except Exception as e:
        print(f"Error fetching top dealers: {e}")
        return "Could not retrieve top dealers."

# ============================================================
# STATE SALES HANDLER
# ============================================================

def sales_by_state(period: str) -> str:
    """Get sales aggregated by state"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        State,
        COUNT(DISTINCT Dealer) as DealerCount,
        SUM("Basic Amount") as TotalSales,
        COUNT(*) as Orders
    FROM {TABLE_NAME}
    WHERE {period_filter}
    GROUP BY State
    ORDER BY TotalSales DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        if df.empty:
            return "No state data available."
        return format_table(df)
    except Exception as e:
        print(f"Error fetching state data: {e}")
        return "Could not retrieve state data."

# ============================================================
# DEALERS AT RISK HANDLER
# ============================================================

def dealers_at_risk(period: str, threshold_score: int = 60) -> str:
    """Find dealers with poor health scores"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        Dealer,
        SUM("Basic Amount") as TotalSales,
        AVG(CASE WHEN "Outstanding Amount" > 0 THEN (julianday('now') - julianday("Invoice Date")) ELSE 0 END) as AvgDelay,
        COUNT(*) as Orders
    FROM {TABLE_NAME}
    WHERE {period_filter}
    GROUP BY Dealer
    HAVING AvgDelay > 15 OR Orders < 5 OR TotalSales < 100000
    ORDER BY AvgDelay DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        if df.empty:
            return "🟢 All dealers look healthy!"
        return "🔴 **Dealers At Risk:**\n" + format_table(df)
    except Exception as e:
        return f"Error: {e}"

# ============================================================
# SUMMARY DASHBOARD
# ============================================================

def summary_dashboard(period: str) -> str:
    """Show overall sales health snapshot"""
    period_filter = period_clause(period)
    query = f"""
    SELECT 
        COUNT(DISTINCT Dealer) as TotalDealers,
        COUNT(DISTINCT State) as TotalStates,
        SUM("Basic Amount") as TotalSales,
        AVG("Basic Amount") as AvgOrderValue,
        SUM(CASE WHEN "Outstanding Amount" > 0 THEN "Outstanding Amount" ELSE 0 END) as TotalOutstanding
    FROM {TABLE_NAME}
    WHERE {period_filter}
    """
    
    try:
        df = pd.read_sql(query, conn)
        stats = df.iloc[0].to_dict()
        
        return f"""
📊 *Sales Dashboard* ({period})

🏪 Total Dealers: {stats['TotalDealers']}
📍 States Covered: {stats['TotalStates']}
💰 Total Sales: ₹{int(stats['TotalSales']):,}
📦 Avg Order Value: ₹{int(stats['AvgOrderValue']):,}
⚠️ Outstanding: ₹{int(stats['TotalOutstanding']):,}
""".strip()
    except Exception:
        return "Could not retrieve dashboard."
