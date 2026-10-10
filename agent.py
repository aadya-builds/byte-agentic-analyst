import os
import time
import functools
from dotenv import load_dotenv
from google import genai
from google.genai import types
import tools as t
t.set_dataframe(pd.read_csv("train.csv"))
import pandas as pd
t.set_dataframe(pd.read_csv("train.csv"))

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=90000),  # 90 seconds, then show an error
)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def logged(func):
    """Wraps a tool so we can see in the terminal when the agent calls it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  [tool call] {func.__name__} {kwargs}")
        result = func(*args, **kwargs)
        print(f"  [tool done] {func.__name__}")
        return result
    return wrapper


tool_list = [logged(f) for f in (
    t.get_schema, t.data_quality_audit, t.compare_groups,
    t.what_drives, t.find_outliers, t.plot_chart,
)]

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        tools=tool_list,
        system_instruction="You are a data analyst. Always use the tools to look at the real dataset before answering. Never guess numbers.",
    ),
)

print("Data analyst agent ready! Type 'exit' to stop.")
while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break
    print("Thinking... (can take up to 90 seconds)")
    start = time.time()
    try:
        response = chat.send_message(question)
        print(f"Agent ({time.time() - start:.0f}s):", response.text or "(no text answer returned)")
    except Exception as e:
        print("Error:", e)