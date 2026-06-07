# Model-Specific Examples

Eleven small runnable examples live here:

- [`des_viscosity/`](./des_viscosity) for the offline DES viscosity workflow
- [`viscosity_template/`](./viscosity_template) for a template-style DES viscosity workflow you can adapt
- [`metal_binding/`](./metal_binding) for the offline metal-binding workflow
- [`ligand_binding_template/`](./ligand_binding_template) for a template-style ligand-binding workflow you can adapt
- [`gemma4_12b/`](./gemma4_12b) for Gemma 4-12B
- [`nemotron_3_nano/`](./nemotron_3_nano) for Nemotron 3 Nano
- [`qwen3_6/`](./qwen3_6) for Qwen 3.6
- [`lidocaine_gemma4_12b/`](./lidocaine_gemma4_12b) for a real lidocaine DES run with Gemma 4-12B
- [`plain_language_gemma4_12b/`](./plain_language_gemma4_12b) for a plain-language DES run routed through Gemma 4-12B
- [`plain_language_metal_binding_gemma4_12b/`](./plain_language_metal_binding_gemma4_12b) for a plain-language metal-binding run routed through Gemma 4-12B
- [`task_router/`](./task_router) for translating a plain-language request into a JSON job

Each folder includes and can be used as a template for your own work:

- a runnable `run.sh`
- a short `README.md`
- a captured `input.txt`
- a captured `output.txt`

The LLM-backed examples also include a model-specific `llm.*.yaml` file.

The DES examples call the shared demo entrypoint. In LLM-enabled runs, candidates are reviewed one by one so large candidate sets stay manageable:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --llm-config <folder>/llm.<name>.yaml
```

The DES viscosity examples use the same demo entrypoint with `--viscosity-model-path artifacts/designsolvents/viscosity/model.json`.
The metal-binding examples use `python -m des_multi_agent.cli --workflow metal-binding ...` and the bundled stability-constant artifact.
The task-router example uses `python -m des_multi_agent.cli task-router "..."` and prints JSON only.

See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for the full walkthrough and output guide.
