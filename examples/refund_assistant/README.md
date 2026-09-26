# Refund assistant example

This example shows the application boundary that LoopEval expects:

1. `app.py` represents an application, RAG pipeline, or agent.
2. `build_scenarios.py` calls that application and captures its output, context,
   and trace as LoopEval samples.
3. LoopEval evaluates those captured outcomes.

From this directory:

```bash
python build_scenarios.py

export TYPESAFE_API_KEY='your-typesafe-key'
export OPENAI_API_KEY='your-openai-key'

loopeval doctor
loopeval bootstrap --requirements requirements.md --scenarios scenarios.jsonl
loopeval candidates --status proposed
loopeval run scenarios.jsonl
```

The bootstrap command proposes checks but does not activate them. Inspect,
review, and validate each candidate before promotion.

In a real application, replace `answer()` with the function, API request, agent
run, or test fixture that produces the behavior you want to evaluate. Keep
stable scenario ids so reports and cached decisions remain comparable.
