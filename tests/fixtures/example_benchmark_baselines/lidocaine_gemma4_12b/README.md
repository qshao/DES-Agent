# Lidocaine + Gemma 4-12B Example

This folder records a real DES screening run for lidocaine using the shared multi-agent demo, the shipped `ml_des_mp` checkpoint, and Ollama Gemma 4-12B. It is also a good place to see how the chemistry-advisor layer adds rationale and warnings around the final ranked candidates.

## Input

- Component A: `lidocaine` free base
- SMILES: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

The wrapper saves stdout to `output.txt` and suppresses stderr so the captured artifact starts with the report table.

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the real run, including:

- ranked DES results
- uncertainty annotations
- Gemma brainstorm candidates (two-stage: chemical family selection first, then SMILES distribution)
- proposal-diversity controls for keeping the brainstorm broad or focused
- explanation notes
- critique notes
- contradiction analysis (`agree` / `conflict` / `uncertain` per candidate, when available)
- chemistry-advisor notes that summarize why the best candidates look plausible and what caveats remain

This is the best place to try `--proposal-diversity-mode explore` if the lidocaine brainstorm gets too narrow, or `balanced` if you want to keep the original family spread.

If you want to keep this run for later reuse, add `--save-run-memory runs/run_001/run.memory.json` to the underlying DES command; you can then label the saved memory in place with `python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good"` and later reuse that file with `--reuse-run`. If you keep several labeled runs under `runs/`, you can also point `--reuse-run` at the parent `runs/` directory to reuse the whole labeled history.
