# Coding agent integration prompt

Copy the prompt below into a coding agent that has access to your application
repository. Replace the bracketed values if you already know them. If you leave
them blank, the agent should infer a sensible starting point from the codebase
and explain its choices.

```text
Integrate LoopEval into this repository and leave me with a working initial
evaluation suite.

LoopEval repository: https://github.com/tbskin/loopeval
Application area to evaluate: [OPTIONAL: feature, pipeline, or agent]
Preferred Jev path: [OPTIONAL: TypeSafe direct or OpenRouter]
Preferred LLM path: [OPTIONAL: OpenAI direct, OpenRouter, compatible endpoint, or none]
Existing evaluation dataset: [OPTIONAL: path or description]

Please do the following:

1. Read LoopEval's README, docs/PROVIDERS.md, and docs/LEARNING_LOOP.md before
   making changes.
2. Inspect this application and identify the smallest useful evaluation target.
   Prefer an existing test dataset or captured examples. Do not change production
   behavior merely to make the evaluation pass.
3. Install LoopEval from the repository or from an existing local checkout.
4. Create an evaluation directory in this application containing:
   - loopeval.yaml
   - a checks directory
   - a small sanitized JSONL dataset
   - a short README explaining how to run the suite
5. Map application records into LoopEval's input, output, context, expected,
   trace, metadata, and data fields as appropriate. Preserve stable sample ids.
6. Create a concise requirements.md describing the behavior these scenarios
   should satisfy. Run `loopeval bootstrap` to propose a small initial check
   library. Do not activate proposals automatically.
7. Review the proposed checks with me. Prefer deterministic checks for facts
   that code can calculate exactly. Keep semantic checks narrow and limited to
   questions that require understanding language.
8. Run the complete workflow offline first with mock providers:
   - loopeval doctor
   - loopeval run <dataset>
   Confirm that artifacts are written locally and that no credential is needed.
9. Prepare the requested real-provider configuration, but reference API keys by
   environment-variable name only. Never write, print, request, or commit an
   actual key. If the provider choice is unspecified, keep the offline setup and
   document the OpenRouter option without enabling it.
10. Add a project-native test or CI command that runs the evaluation suite only
   if doing so is deterministic and does not require a secret in pull requests.
11. Run the application's existing tests plus the new evaluation smoke test.
12. Summarize:
    - files created or changed
    - how samples map into LoopEval
    - checks added and why
    - exact commands to run offline and with the selected provider
    - where local reports are stored
    - any labeled data still needed before thresholds can be trusted

Keep the first integration small, readable, and easy to review. Do not invent
quality claims from unlabeled data, do not activate generated candidate checks,
and do not place model judgments in an authorization path.
```
