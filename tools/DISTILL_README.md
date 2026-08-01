# Distilling the Bayes filter into a small onboard LLM

## The idea

This repo has two fault diagnosers behind one interface (`sim/advisor/base.py`):

- **`BayesAdvisor`** (`sim/advisor/bayes.py`) — a discrete Bayes filter. Given its
  belief and the shared cost matrix it is **decision-theoretically optimal**, and
  it is free to query. It is the accuracy anchor.
- **`GemmaAdvisor`** (`sim/advisor/gemma.py`) — asks a language model the same
  question, phrased as a natural-language *evidence prompt* built by
  `GemmaAdvisor._prompt(obs)`.

Because both arms see the **same observations through the same interface**, every
adverse link the simulator produces gives us a free training pair:

```
(the exact evidence prompt the LLM would be asked)  ->  (the cause the optimal Bayes filter concluded)
```

Collect thousands of these across many simulated scenarios and you have a
supervised fine-tuning (SFT) dataset that teaches a **small student LLM to imitate
the optimal filter** — a "distillation." The student is deployable onboard: it
runs the diagnosis directly, in natural language, without needing a Bayes filter
and gossip bookkeeping wired into the flight software, and it can explain itself
where the filter cannot.

Pipeline:

```
tools/distill_dataset.py   (local, CPU)   -> out/distill_{dataset,train,val}.jsonl
tools/distill_train.py     (GPU / Colab)  -> a LoRA adapter
serve the adapter behind an OpenAI-compatible endpoint
point sim/advisor/gemma.py's SPUR-BASE-URL / SPUR-MODEL at it
```

---

## Step (a) — generate the dataset locally

Runs in the existing repo venv (numpy + skyfield only; **no torch, no GPU**).

```bash
source .venv/bin/activate
python -m tools.distill_dataset --seeds 30 --faults 12
```

Outputs (in `out/`):

- `distill_dataset.jsonl` — every unique row
- `distill_train.jsonl` / `distill_val.jsonl` — a deterministic 90/10 split

Each line is a chat-format row:

```json
{"messages": [
  {"role": "user", "content": "<the gemma evidence prompt>"},
  {"role": "assistant", "content": "{\"cause\": \"C\", \"confidence\": 0.87, \"rationale\": \"\"}"}
]}
```

The `user` prompt is byte-for-byte what `GemmaAdvisor._prompt` sends at inference
(same cause menu A–F, same evidence lines). The `assistant` label is the Bayes
filter's argmax cause (mapped to its A–F letter) plus its top posterior
probability. `rationale` is intentionally empty — the Bayes filter has no natural
language, and we do not want to teach the student to invent one.

The script prints the total row count, the label (cause) distribution, and one
example row.

**Knobs:** `--seeds` (number of distinct scenarios), `--faults` (faults per
scenario), `--start-seed`, `--val-frac`, `--shuffle-seed`.

### How many rows is enough?

For this task the student is learning a fairly deterministic
evidence-bucket → cause mapping, so it needs far less data than open-ended
instruction tuning:

- **~500 rows**: minimum to see the mapping learned at all.
- **1,000–3,000 rows**: a good target — enough for LoRA to nail the common causes.
  `--seeds 30` lands here.
- **5,000+ rows**: only helps if you also broaden coverage of *rare* feature
  combinations; bump `--seeds` to 50+.

Note the evidence prompt is a **coarse, bucketed** summary (e.g. silence is
short/medium/long, co-failure is yes/no/unknown), so there is a finite ceiling of
*distinct* prompts. The generator dedups exact (prompt, label) pairs — many raw
diagnoses collapse to the same unique row, which is expected. More seeds mainly
surface rarer combinations rather than multiplying near-duplicates.

**On label balance:** the distribution is the teacher's *honest* output, and it is
skewed (node_down and pointing dominate; buffer/weather are rare because the
simulator's traffic and geometry make the Bayes filter rarely conclude them from
local evidence). That is a faithful reflection of what the optimal filter
actually decides. If you want a more balanced student, either raise `--faults`,
or downsample the majority classes / upweight minority ones before training —
but do not expect the student to beat the teacher on classes the teacher itself
rarely picks.

---

## Step (b) — train on Colab or a GPU box

`tools/distill_train.py` is a LoRA SFT script (transformers + peft + trl's
`SFTTrainer`). It imports torch/peft/trl **inside `__main__`**, so it stays
importable on the local box; it is meant to run on a GPU.

### What to upload

- `tools/distill_train.py`
- `out/distill_train.jsonl` and `out/distill_val.jsonl`

### Environment (Colab cell or GPU box shell)

```bash
pip install -U "transformers>=4.45" "trl>=0.12" "peft>=0.13" \
               "datasets>=3.0" accelerate bitsandbytes
```

### Run

```bash
python tools/distill_train.py \
    --train out/distill_train.jsonl \
    --val   out/distill_val.jsonl \
    --base  Qwen/Qwen2.5-1.5B-Instruct \
    --out   out/distill_adapter \
    --epochs 3 --load-4bit
```

Notes:

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` is tiny, permissive, and ungated —
  a good default. To match the "gemma" framing use `google/gemma-2-2b-it`
  (requires accepting Google's license on Hugging Face and `huggingface-cli
  login`). `meta-llama/Llama-3.2-1B-Instruct` also works.
- `--load-4bit` (QLoRA) lets a 1–2B base train inside a **free Colab T4 (16 GB)**.
  Drop it if you have a bigger GPU.
- **Time:** on a T4, ~1–2k rows × 3 epochs is roughly **10–25 minutes**. On an
  A100 it is a few minutes.

### What you get

- `out/distill_adapter/` — the **LoRA adapter** (a few MB) plus the tokenizer.
  This is the deployable artifact.
- Add `--merge` to also write `out/distill_adapter_merged/`, a standalone
  full-weights model that servers can load without knowing about PEFT.

Sanity-check before serving (the output must be JSON gemma.py can parse):

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("out/distill_adapter")
model = AutoPeftModelForCausalLM.from_pretrained("out/distill_adapter").cuda()
prompt = open("out/distill_val.jsonl").readline()  # take a row's user content
# apply chat template, generate, confirm you get {"cause": "<A-F>", ...}
```

---

## Step (c) — serve it behind an OpenAI-compatible endpoint

`sim/advisor/gemma.py` talks to any OpenAI-compatible `/v1/chat/completions`
endpoint. Stand the student up the same way and point the advisor at it.

### Option 1 — vLLM (recommended; native LoRA support)

```bash
pip install vllm
# merge first (Step b with --merge), then serve the merged model:
python -m vllm.entrypoints.openai.api_server \
    --model out/distill_adapter_merged \
    --served-model-name spur-student \
    --port 8000
# vLLM can also serve the adapter directly via --enable-lora / --lora-modules.
```

### Option 2 — any OpenAI-compatible server

TGI, llama.cpp's `server`, Ollama (`ollama create` from the merged model), etc.
all expose the same `/v1/chat/completions` shape.

### Point the simulator at it

`sim/advisor/gemma.py` reads three values from the repo-root `.env`
(`SPUR-GEMMA4-API-KEY`, `SPUR-BASE-URL`, `SPUR-MODEL`). Set:

```
SPUR-GEMMA4-API-KEY=dummy-key-if-your-server-ignores-auth
SPUR-BASE-URL=http://localhost:8000/v1
SPUR-MODEL=spur-student
```

Then run the gemma arm exactly as before:

```bash
python -m tools.gemma_smoke        # confirm the endpoint answers + parses
python -m sim.trace gemma          # full run against the distilled student
```

The advisor already extracts `{"cause", "confidence", "rationale"}` from the
reply (`GemmaAdvisor._parse`) and reuses the shared optimal decision rule
`bayes_action`, so the distilled student drops straight into the existing
comparison against the Bayes filter — now as a self-contained onboard model
instead of a hosted frontier LLM.
```
