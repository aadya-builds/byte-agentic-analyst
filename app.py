import os
import functools
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import tools as t

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM_PROMPT = (
    "You are a data analyst. Always use the tools to look at the real dataset before answering. "
    "Never guess numbers. Only state facts that are supported by tool output. "
    "Call only the tools needed for the question; do not run extra analyses that were not asked for. "
    "You cannot see charts: after plotting, say only that the chart was saved and never describe its shape. "
    "To count how often values occur, use top_values. "
    "If a tool cannot answer the question, say so honestly instead of answering from memory."
    "When a column holds several values in one cell (for example countries or genres), count each value separately. "
)

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊")
st.title("📊 Data Analyst Agent")


@st.cache_resource
def get_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=90000),
    )


def make_chat():
    """Creates a new Gemini chat whose tools log every call into session_state."""
    def logged(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            st.session_state.calls.append(f"{func.__name__}({kwargs})")
            return func(*args, **kwargs)
        return wrapper

    tool_list = [logged(f) for f in (
        t.get_schema, t.data_quality_audit, t.compare_groups,
        t.what_drives, t.find_outliers, t.plot_chart, t.top_values, t.describe_column,
    )]
    return get_client().chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(tools=tool_list, system_instruction=SYSTEM_PROMPT),
    )


# ---------- Sidebar: upload the dataset ----------
uploaded = st.sidebar.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Upload a CSV file in the sidebar to start.")
    st.stop()

# New file uploaded -> load it and reset the conversation
if st.session_state.get("file_name") != uploaded.name:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.file_name = uploaded.name
    st.session_state.calls = []
    st.session_state.messages = []
    st.session_state.chat = make_chat()

# Give the tools the current dataset on every run
t.set_dataframe(st.session_state.df)

with st.sidebar.expander("Preview of your data", expanded=True):
    st.write(f"{len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
    st.dataframe(st.session_state.df.head())

# ---------- Show earlier messages ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            with st.expander("Tools the agent used"):
                for c in m["tools"]:
                    st.code(c)
        if m.get("chart"):
            st.image(m["chart"])

# ---------- New question ----------
question = st.chat_input("Ask a question about your data")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    st.session_state.calls = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat.send_message(question)
                answer = response.text or "(no text answer returned)"
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

        used = list(st.session_state.calls)
        if used:
            with st.expander("Tools the agent used"):
                for c in used:
                    st.code(c)

        chart_bytes = None
        if any(c.startswith("plot_chart") for c in used) and os.path.exists("chart.png"):
            with open("chart.png", "rb") as f:
                chart_bytes = f.read()
            st.image(chart_bytes)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": used, "chart": chart_bytes}
    )