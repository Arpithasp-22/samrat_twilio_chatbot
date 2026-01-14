import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ======================
# CONFIG
# ======================

DB_FILE = "data/sales.db"
TABLE_NAME = "sales"
MODEL = "gpt-3.5-turbo"

conn = None
llm = None
COLUMNS = None


# ======================
# SETUP (RUNS ONCE)
# ======================

def setup_bot():
    global conn, llm, COLUMNS

    print("🔹 Connecting to SQLite database...")
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)

    # Read table schema
    sample_df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 1", conn)
    COLUMNS = list(sample_df.columns)

    llm = ChatOpenAI(model=MODEL, temperature=0)

    print("✅ SQL analytics engine ready.")
    print("Columns:", COLUMNS)


# ======================
# MAIN QUERY HANDLER
# ======================

def search_answer(user_question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior data analyst.\n"
                "You are working with a SQLite database.\n\n"
                f"Table name: {TABLE_NAME}\n"
                f"Columns: {COLUMNS}\n\n"
                "STRICT RULES:\n"
                "- Generate ONLY a single SQL SELECT query\n"
                "- Do NOT use INSERT, UPDATE, DELETE, DROP\n"
                "- Do NOT modify data\n"
                "- If user asks for TOTAL → return ONE value\n"
                "- If user asks for 'by / per / wise' → use GROUP BY\n"
                "- Use SUM(), COUNT(), AVG() when needed\n"
                "- Do NOT explain anything\n\n"
                "Examples:\n"
                "Total sales → SELECT SUM(amount) FROM sales;\n"
                "Sales by State → SELECT State, SUM(amount) FROM sales GROUP BY State;\n"
                "Total sales TG → SELECT SUM(amount) FROM sales WHERE State = 'TG';\n"
            )
        ),
        ("human", "{question}")
    ])

    try:
        response = llm.invoke(
            prompt.format_messages(question=user_question)
        )

        sql_query = response.content.strip()
        print("🧠 Generated SQL:", sql_query)

        df_result = pd.read_sql(sql_query, conn)

        # -------------------------
        # WhatsApp-friendly output
        # -------------------------

        if df_result.empty:
            return "No data found for this query."

        if df_result.shape == (1, 1):
            value = df_result.iloc[0, 0]
            return f"Total: {int(value):,}"

        return df_result.to_string(index=False)

    except Exception as e:
        print("❌ SQL Error:", e)
        return "I couldn't compute that from the database."
