# Data Analyst Agent

Upload any CSV file, ask questions in plain English, and an AI agent answers by running real analysis on the given data. It does not guess numbers: it picks from 8 analysis tools, runs them with pandas, and answers from the results. The app shows which tools it used for every answer.


## Setup

1. Install Python 3.10 or newer.
2. Clone the repo and install the dependencies:
```
   git clone https://github.com/aadya-builds/byte_agentic_analyst.git
   cd byte_agentic_analyst
   pip install -r requirements.txt
```
3. Get a free Gemini API key from https://aistudio.google.com
4. Copy `.env.example` to a new file called `.env` and fill it in:
```
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=your_model_name
```
   Use a model from AI Studio that supports function calling and still has free quota (see Limitations).
5. Run the app:
```
   streamlit run app.py
```
6. Upload any CSV in the sidebar and start asking questions.

Two sample datasets are included: `train.csv` (Titanic) and `netflix_titles.csv` (Netflix).

## The 8 tools

| Tool | What it does | Example question |
|---|---|---|
| `get_schema` | Lists every column with its type and number of missing values, plus the row count | "How many rows are there?" |
| `data_quality_audit` | Finds duplicate rows, columns with missing values, single-value columns and negative numbers | "Is the dataset clean?" |
| `compare_groups` | Compares a numeric column across the groups of a category and runs a statistical test (Welch t-test for 2 groups, ANOVA for 3 or more) to say whether the difference is real or could be chance | "Did men and women pay different fares?" |
| `what_drives` | Ranks which columns are most related to a numeric target column (correlation for numbers, spread across categories for text columns) | "What affected survival the most?" |
| `find_outliers` | Finds unusually high or low values in a numeric column using the IQR method | "Are there outliers in Fare?" |
| `plot_chart` | Draws a histogram, bar chart or scatter plot and shows it in the app | "Show a bar chart of survival by Sex" |
| `top_values` | Counts how often each value appears in a column. If a cell holds several values like "United States, India", it splits them and counts each one | "Which country has the most titles?" |
| `describe_column` | Gives overall count, mean, median, standard deviation, min, max and quartiles of one numeric column | "What is the median Fare?" |

No tool is hard-coded to one dataset. Column names are always passed in by the agent after it reads the schema, so the same tools work on any CSV.

## How it works

1. The user uploads a CSV. The app loads it with pandas and gives it to the tools through `set_dataframe()` in `tools.py`.
2. The user asks a question. It is sent to Gemini together with the descriptions of the 8 tools.
3. Gemini decides which tool to call and with which arguments. My Python function runs on the real data and returns the result.
4. The result goes back to Gemini, which writes the final answer. It can call several tools in a row.
5. The app shows the answer, a "Tools the agent used" panel with the exact calls, and any chart.

The system prompt tells the agent to always use tools before answering, never guess numbers, only run the tools it needs, count multi-value cells separately, and never describe a chart (it cannot see the image).

Each tool returns a clear error message (for example, "Column 'X' not found. Available columns: ...") instead of crashing, so the agent can correct itself and retry.

## Testing

I computed the correct answers myself with pandas (`eval_truth.py`, `netflix_truth.py`) and compared them to the agent's answers. Full tables are in [evaluation.md](evaluation.md).

1. **Titanic, 10 questions:** 9/10 correct at first, 10/10 after I added `describe_column`.
2. **Netflix, a different dataset:** 4/4 correct after fixes.

## What failed and how I fixed it

1. **Median Fare had no answer.** The agent ran 8 tool calls and gave up, because no tool gave a summary of a whole column (only group-wise numbers). I added `describe_column`, and it now answers in one call.
2. **"Which country has the most titles?" had no answer on Netflix.** There was no tool for counting. I added `top_values`.
3. **The count was wrong.** After adding `top_values`, the agent said the United States had 2,818 titles, but the real number is 3,689. The cells contain values like "United States, India", and the tool treated each whole cell as one value. I added separator support and automatic splitting, plus a hint in the system prompt. It now gives 3,689.
4. **The agent described chart shapes it could not see.** It said a histogram "peaked in the 2010s" from its own general knowledge. I told it in the prompt to only say that the chart was saved.
5. **Too many tool calls per question.** The agent sometimes ran extra analyses nobody asked for, which wasted free quota. I told it in the prompt to call only the tools needed.

The main lesson: a confident answer is not always a correct answer. The country answer had the right name and the wrong number, and I only caught it because I had checked against pandas.

## Limitations

1. **Free quota is small.** The first model I used allowed only 20 requests per day, and I ran out while testing. A question can need several requests because every tool call is a round trip to the model, so I switched to a model with a bigger free limit.
2. **Outlier ranges can look odd.** The IQR method on skewed columns such as Fare gives a negative lower limit, which makes no sense for prices. Only the upper outliers are meaningful there.
3. **The agent can pass wrong arguments to a tool** and still answer confidently. The country bug above was an example. The auto-split in `top_values` is a safety net for this case, but other cases may exist.
4. **It sometimes adds background knowledge** that did not come from a tool. The prompt reduces this, but does not remove it completely.
5. **No time-series tool.** Date columns are treated as plain text, so trend questions will not work well.
6. **The agent cannot see charts.** It only knows that a chart was saved.
7. **Persistent memory is not implemented.**

## What I wrote vs what I used

1. **Used:** pandas, scipy, matplotlib, Streamlit, and the Gemini API through `google-genai`.
2. **AI assistance:** I am a first-year student and this was my first time using pandas and building an agent. I used an AI assistant (Claude) to help me learn, write and debug the code. I took assistance but did not just copy paste it but also tried to learn whatever i have mentioned in this project. [Change this to say exactly what you did yourself.]
3. **My own work:** I set up and ran everything, tested on two different datasets, worked out the correct answers with pandas, found the failures listed above, and made the fixes. I was confused over the tools to sleect but then did some hands on and selected these.

## What I would do next

1. Add persistent memory, for example remembering that "active customers" means status is not "cancelled" and applying it in later answers, with options to view and delete saved facts.
2. Add a time-series tool for date columns (monthly trends).
3. Let the user paste their own API key in the app.