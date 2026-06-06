# Model-Specific Examples

Four small runnable examples live here:

- [`gemma4_12b/`](./gemma4_12b) for Gemma 4-12B
- [`nemotron_3_nano/`](./nemotron_3_nano) for Nemotron 3 Nano
- [`qwen3_6/`](./qwen3_6) for Qwen 3.6
- [`lidocaine_gemma4_12b/`](./lidocaine_gemma4_12b) for a real lidocaine DES run with Gemma 4-12B

Each folder includes:

- a runnable `run.sh`
- a short `README.md`
- a captured `input.txt`
- a captured `output.txt`
- a model-specific `llm.*.yaml`

These examples all call the same demo entrypoint. In LLM-enabled runs, candidates are reviewed one by one so large candidate sets stay manageable:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --llm-config <folder>/llm.<name>.yaml
```


See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for the full walkthrough and output guide.
