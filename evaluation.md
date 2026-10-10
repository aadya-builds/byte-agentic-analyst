# Evaluation

I tested the agent in two ways: a 10-question accuracy check on the Titanic dataset, and a generalisation test on the Netflix dataset. For every question I computed the correct answer myself with pandas (`eval_truth.py`, `netflix_truth.py`) and compared it with the agent's answer.

## Table 1: Titanic accuracy (10 questions)

| # | Question | Real answer (pandas) | Agent's answer | Tool used | Correct? |
|---|---|---|---|---|---|
| 1 | How many rows? | 891 | 891 | get_schema | Yes |
| 2 | Cabin missing %? | 77.1 | 77.10% (687/891) | get_schema | Yes |
| 3 | Average Age? | 29.70 | 29.6991 | describe_column | Yes (before fix: via compare_groups) |
| 4 | Avg Fare, men vs women? | F 44.48 / M 25.52 | F 44.48 / M 25.52 | compare_groups | Yes |
| 5 | Survival rate, men vs women? | F 0.742 / M 0.189 | 74.20% / 18.89% | compare_groups | Yes |
| 6 | Survival rate per class? | 0.630 / 0.473 / 0.242 | 62.96% / 47.28% / 24.24% | compare_groups | Yes |
| 7 | Fare outliers? | 116 | 116 | find_outliers | Yes |
| 8 | Most correlated with Survived? | Pclass (-0.34) | Pclass (-0.34) | what_drives | Yes |
| 9 | Passengers per port? | S 644 / C 168 / Q 77 | S 644 / C 168 / Q 77 (2 missing) | top_values | Yes |
| 10 | Median Fare? | 14.4542 | 14.4542 | describe_column | Yes (before fix: No) |

**Score: 9/10 at first, 10/10 after adding `describe_column`.**

## Table 2: Netflix (a different dataset)

| Question | Real answer (pandas) | Before fix | After fix | Correct? |
|---|---|---|---|---|
| Which country has the most titles? | United States, 3689 | Failed: 8 tool calls, no answer | United States, 3,689 (`top_values`, `separator=', '`) | Yes |
| Which rating is most common? | TV-MA, 3207 | TV-MA 3,207, but via `compare_groups` (wrong tool) | TV-MA 3,207 (`top_values`) | Yes |
| Avg release_year, movies vs TV shows? | Movie 2013.12 / TV Show 2016.61 | - | 2013.12 vs 2016.61 (`compare_groups`) | Yes |
| Outliers in release_year? | 719 | - | 719 of 8,807 (`find_outliers`) | Yes |

**Score: 4/4 after fixes. Two of these failed or used the wrong tool before I added `top_values`.**

## Table 3: Gaps I found and how I fixed them

| # | Dataset | Problem | Root cause | Fix | Result |
|---|---|---|---|---|---|
| 1 | Titanic | No answer for median Fare, 8 tool calls | No tool gave a whole-column summary | Added `describe_column` | Correct, 1 call |
| 2 | Netflix | No answer for "most titles by country", 8 tool calls | No counting tool | Added `top_values` | United States found |
| 3 | Netflix | United States count was 2,818 (real: 3689) | Cells like "United States, India" were treated as one value | Added separator support and auto-split in `top_values`, plus a hint in the system prompt | 3,689, correct |
| 4 | Both | Agent described chart shapes it had never seen | The model cannot see the saved image | System prompt: never describe charts | No longer happens |
| 5 | Both | Too many tool calls per question, free quota ran out quickly | Agent ran extra, unrequested analyses | System prompt: call only the tools needed | Fewer calls |

## Tools

`get_schema`, `data_quality_audit`, `compare_groups`, `what_drives`, `find_outliers`, `plot_chart`, `top_values`, `describe_column`