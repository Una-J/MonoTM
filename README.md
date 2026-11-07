# MonoTM

Topic modeling with interpretable semantic features, accompanying **Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features**, by Una Joh and Bei Yu (AACL-IJCNLP 2026).

MonoTM estimates document-topic mixtures from **all positive SAE feature activations**, then learns topic descriptions from a separate vocabulary of **validated semantic features**, with the mixtures held fixed. Each topic is described by ranked feature labels rather than only top words.

MonoTM accepts a CSV corpus and returns document-topic mixtures and semantic topic descriptions. The number of topics is configurable, and no gold labels are required.

Use the **command line** below, or open [examples/start_here.ipynb](examples/start_here.ipynb) to choose settings, review your environment, and run from a notebook. For post-run label inspection and repairs, use [examples/inspect_and_retry.ipynb](examples/inspect_and_retry.ipynb).

## Install

Python 3.9 or newer is supported. A CUDA GPU is recommended for the default 8B embedding model and SAE training. CPU execution is available but the default embedding model is large.

```bash
git clone https://github.com/Una-J/MonoTM.git
cd MonoTM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for your CUDA environment first, if needed.
python -m pip install -e '.[embeddings]'
```

The embedding extra pins Transformers 4.51.0 to match the [Nemotron model card](https://huggingface.co/nvidia/llama-embed-nemotron-8b). The loader uses eager attention, without a FlashAttention installation. Model downloads use the Hugging Face cache. See the model card for its model-specific terms.

For long documents, eager attention can require substantial GPU memory. After installing a compatible FlashAttention build as described by the model card, set `"embedding_attention": "flash_attention_2"` in your JSON config. Alternatively, reduce `embedding_max_length` (default 32,768); the truncated-document count is recorded. Nemotron's custom model supports eager or FlashAttention, not SDPA.

For precomputed embeddings, install the core package with `python -m pip install -e .` and supply `--embeddings`.

## Run on your data

Create a UTF-8 CSV with a `text` column and one nonempty document per row:

```csv
text
"A document about a scientific discovery."
"Another document about public transportation."
```

This snippet illustrates the CSV format. Default feature validation requires at least 30 activating documents per feature and enough inactive documents for labeling and validation.

Export your own [DeepInfra API token](https://docs.deepinfra.com/chat/overview), then run:

```bash
export DEEPINFRA_API_KEY='your-own-key'
monotm check --input my_documents.csv --topics 20 --output runs/my_corpus
monotm run --input my_documents.csv --topics 20 --output runs/my_corpus
```

The LLM stages send document excerpts to the configured chat provider and incur that provider's charges. Interpreter and Predictor use Gemma 3 27B and 12B by default. The endpoint and both models are configurable. `MONOTM_API_KEY` overrides `DEEPINFRA_API_KEY`; keys are never written to run metadata or response caches.

For different column names and stable IDs:

```bash
monotm run --input my_documents.csv --text-column body --id-column document_id \
  --topics 12 --output runs/my_corpus
```

Without `--id-column`, IDs are generated from CSV row order. Empty text, empty supplied IDs, and duplicate IDs are rejected rather than silently dropped.

## What happens

1. Generate L2-normalized, mean-pooled document embeddings using `nvidia/llama-embed-nemotron-8b`.
2. Standardize the embeddings and train a mixture SAE with dictionary width `0.5 × embedding dimension` and Top-K activity 256.
3. Fit LDA to its nonnegative activations to estimate document-topic mixtures **Θ**.
4. Train an interpretation SAE with width `2 × embedding dimension` and activity 16.
5. Label supported features with the Interpreter and validate the labels with a separate Predictor. Its activation-prediction F1 is the interpretability score **IS**.
6. Retain nonempty labels with IS ≥ 0.8 and support ≥ 30 documents. Weight each retained activation by IS, then estimate the topic-feature distribution **B** using smoothed EM with **Θ fixed**.

Both SAEs and their feature vocabularies are learned from the supplied corpus. Dictionary width, Top-K activity, training limits, and validation settings are configurable.

## Configure a run

```bash
monotm config --output settings.json
# Edit settings.json, then:
monotm run --input my_documents.csv --topics 20 --config settings.json \
  --output runs/my_corpus
```

JSON files may contain only the settings you want to override. For example:

```json
{
  "mixture": {"width": 0.5, "k": 128},
  "interpretation": {"width": 1, "k": 8},
  "is_threshold": 0.8,
  "max_features": 100,
  "llm_workers": 2
}
```

`max_features` limits the number of interpretation features sent for validation, ranked by document support. It does not filter the mixture SAE's input to LDA. Set it to `null` for all eligible features. Validation can require one Interpreter call and up to 60 Predictor calls per feature; this setting is useful for a first run.

`--topics` and explicitly supplied CLI options override the JSON configuration. `monotm run --help` lists CLI options; `monotm config` lists the full configuration. To use another OpenAI-compatible endpoint:

```bash
export MONOTM_API_KEY='your-provider-key'
monotm run --input my_documents.csv --topics 20 --output runs/custom \
  --base-url https://your-provider.example/v1 \
  --interpreter-model your-interpreter-model --predictor-model your-predictor-model
```

## Scikit-learn LDA parameters

The `lda_params` dictionary uses the constructor names of scikit-learn's [`LatentDirichletAllocation`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html). `monotm config` writes the complete dictionary, and the first settings cell in [examples/start_here.ipynb](examples/start_here.ipynb) lists every parameter with an explanation. Edit the values directly:

| Parameter | MonoTM default | Purpose |
|---|---|---|
| `n_components` | `20` | Topic count throughout MonoTM, including Θ and descriptor outputs |
| `doc_topic_prior` | `null` | Document-topic prior; `null` uses `1 / n_components` |
| `topic_word_prior` | `null` | Topic-feature prior; `null` uses `1 / n_components` |
| `learning_method` | `"batch"` | `"batch"` or `"online"` variational Bayes |
| `learning_decay` | `0.7` | Learning-rate decay for online updates |
| `learning_offset` | `10.0` | Offset for online updates |
| `max_iter` | `200` | Maximum fitting iterations |
| `batch_size` | `128` | Mini-batch size for online updates |
| `evaluate_every` | `-1` | Perplexity evaluation frequency; nonpositive values disable it |
| `total_samples` | `1000000.0` | Used by scikit-learn's `partial_fit`; inactive in this fit workflow |
| `perp_tol` | `0.1` | Perplexity convergence tolerance when evaluation is enabled |
| `mean_change_tol` | `0.001` | Document-topic inference convergence tolerance |
| `max_doc_update_iter` | `100` | Maximum document-topic inference iterations |
| `n_jobs` | `1` | CPU parallelism; `-1` uses all available CPUs, `null` uses the joblib default |
| `verbose` | `0` | Logging verbosity |
| `random_state` | Pipeline `seed` (`42`) | LDA-specific integer seed; `null` uses scikit-learn's random generator |

For JSON, use numbers, strings, and `null`; in Python use `None` for `null`. A `RandomState` object is not accepted because run settings are saved as JSON. `random_state` can be set independently of the pipeline seed used for SAE training and example sampling.

`lda_params` takes precedence over the fallback aliases `n_topics`, `lda_max_iter`, and `lda_jobs`. If `n_components` is supplied, MonoTM uses it as the topic count for all stages. An explicit CLI `--topics` overrides both `n_topics` and `lda_params.n_components`. The CLI flag is optional when the topic count is provided in the configuration. Defaults apply when it is omitted from both.

`monotm check` displays all resolved constructor parameters and identifies inactive options. Batch learning does not use `learning_decay`, `learning_offset`, or `batch_size`. `perp_tol` is inactive when `evaluate_every <= 0`. This pipeline calls `fit_transform`, so `total_samples`, which controls `partial_fit`, is always inactive here. Invalid values and constructor keywords are rejected before SAE training or model downloads. Accepted parameters are forwarded to scikit-learn and recorded in `run.json`.

Changing LDA settings changes the fitted topic mixtures and requires a new run directory. `retry` repairs LLM validation and descriptors while retaining the saved LDA mixtures.

## Review setup before running

`monotm check` displays the actual settings after merging defaults, JSON, and explicit CLI overrides. It does not download models, train, contact the chat endpoint, or create/modify a run directory.

```bash
monotm check --input my_documents.csv --topics 20 --config settings.json \
  --id-column document_id --output runs/my_corpus
```

The report shows:

| Item | What you can check |
|---|---|
| Models | Embedding, Interpreter, and Predictor model IDs |
| Hardware | PyTorch/CUDA build, visible GPU names, total/free VRAM, Apple MPS availability |
| Selected device | Requested `auto`/`cpu`/`cuda:0` and the device actually selected |
| Execution | Which stages train, fit on CPU, use the chat endpoint, or reuse saved results |
| LDA | Complete scikit-learn constructor parameters, LDA/pipeline seeds, and inactive options |
| Data/output | CSV text/ID columns, document count, topic count, embedding dimensions when available, output identity/lock |
| LLM settings | Endpoint, key registration/source with its value hidden, support/IS cutoff, feature cap, requests per feature |

`auto` selects a CUDA device if PyTorch detects one, otherwise CPU. Explicit `cpu` remains CPU even if a GPU is present. Explicit unavailable CUDA/MPS devices or an invalid GPU index are reported before training. The embedding model and SAEs use this device; LDA/descriptor EM use CPU, and the LLM endpoint manages its own device independently. The CPU smoke profile explicitly selects CPU, including its separate example LLM server.

For configuration/environment review before selecting data, use `monotm check --config settings.json`. Provide `--input` and `--output` for a full corpus/resume check. For machine-readable settings, add `--json`. A header/ID check of precomputed embeddings uses memory mapping; array finiteness is checked during the run. Matching an existing run also hashes the supplied embedding file to verify its identity.

The same preview is available through the run command:

```bash
monotm run --input my_documents.csv --topics 20 --config settings.json \
  --output runs/my_corpus --dry-run
```

A normal `run` prints the same summary, stops on local blocking issues, and otherwise starts execution. An identical completed run can be reviewed and resumed without an API key because its validated labels are reused.

These are **local checks**. They do not verify API authentication, endpoint connectivity, model access, or model memory requirements. No inference requests are made by `check` or `--dry-run`.

For a notebook interface, [examples/start_here.ipynb](examples/start_here.ipynb) has separate cells for editable settings, a read-only setup report, hidden key input if needed, and execution. The first settings cell includes a **Scikit-learn LDA parameters** dictionary with all constructor options. The equivalent Python check returns the complete report as a dictionary:

```python
from monotm import check_run
from monotm.config import RunConfig
from monotm.preflight import format_report

plan = check_run("my_documents.csv", "runs/my_corpus", config=RunConfig(n_topics=20))
print(format_report(plan))
# plan["config"] contains the effective settings; plan["errors"] lists issues.
```

## Use precomputed embeddings

```bash
monotm run --input my_documents.csv --topics 20 --output runs/precomputed \
  --embeddings document_embeddings.npy
```

The `.npy` file must be a finite numeric matrix with one row per document in **the same order as the CSV**. To align by ID instead, supply `--id-column document_id --embedding-ids embedding_ids.json`. The JSON file must be an array of string IDs in embedding-row order, matching the CSV IDs exactly.

Dictionary width is a multiplier of the supplied embedding dimension. With lower-dimensional embeddings, reduce K or increase width so that `width × dimension ≥ K` for each SAE. The program checks this before training.

## Outputs

| File | Contents |
|---|---|
| `topics.json` | Ranked semantic descriptors and their probabilities for each topic |
| `topic_features.csv` | The same top-feature report in tabular form |
| `document_topics.csv` | Full topic mixture for every document, with its ID |
| `documents.csv` | Top topic, its probability, and whether the document had positive mixture features |
| `theta.npy` | Document-topic matrix, aligned with `document_ids.json` |
| `beta.npy` | Topic-feature matrix, aligned with `vocabulary.csv` |
| `feature_scores.csv` | Candidate feature labels, support, validation status, and IS |
| `models/` | Both SAE checkpoints, LDA components, descriptor counts, and training/EM metadata |
| `cache/` | Embeddings, activations, example IDs, prompt snapshots, accepted LLM responses, and every request attempt |
| `run.json` | Configuration, input fingerprint, package versions, and completion status |

Feature IDs are specific to the interpretation SAE. Probabilities in `topic_features.csv` are taken from B before top-feature truncation, so the displayed subset need not sum to one. Documents with no positive mixture activations retain their row and receive LDA's prior mixture.

## Resume and validation behavior

Rerun the same command to reuse completed stages and cached successful LLM calls. An interrupted SAE training stage restarts; completed SAE stages are reused. Changing input text, ID order, training settings, prompts, or recorded package versions through `run` requires a **new output directory**. Use `retry` below to change LLM models or an individual label within an existing run. An existing directory belonging to other work is rejected. A process lock prevents simultaneous writes; after an unclean process termination, remove `run.lock` only once the old process is no longer running.

Support and activation targets use **strictly positive values**. Predictor documents are disjoint from all documents shown to the Interpreter. Validation uses balanced positive/negative sets of up to 30 documents each. The actual count is recorded, and features with too few held-out examples are skipped.

Malformed responses and transient API failures are retried. Rejected request parameters and replies with `finish_reason: "length"` stop immediately; MonoTM does not silently change parameters or increase token budgets. If API calls fail, the run stops with successful calls cached and feature progress saved incrementally. If no labels pass validation, mixtures remain saved in `theta.npy` and scores in `feature_scores.csv`. Only validated labels meeting the configured thresholds are used for topic descriptions. Individual labels can be revised and revalidated with `retry`.

Response caches and attempt logs contain document excerpts. Run directories under `runs/` are excluded from Git.

## LLM generation settings

Interpreter and Predictor have independent `interpreter_generation` and `predictor_generation` objects in the JSON configuration:

| Setting | Default | Behavior |
|---|---|---|
| `temperature` | `0.0` | Send this value; JSON `null` omits the field entirely |
| `token_limit` | `1024` | Positive token budget for this role |
| `token_limit_parameter` | `"max_tokens"` | Choose `"max_tokens"` or `"max_completion_tokens"`; only one is sent |
| `reasoning_effort` | `null` | Omit by default, or send a model-supported effort value |

`reasoning_effort: "none"` explicitly requests no reasoning; `null` leaves the choice to the provider. Not every model supports every effort value. Temperature omission uses the provider's default and does not mean temperature zero.

For OpenAI reasoning models served through Chat Completions, select `max_completion_tokens` and omit unsupported temperature settings. For example, these entries can be added to a run configuration:

```json
{
  "base_url": "https://api.openai.com/v1",
  "interpreter_model": "o3",
  "predictor_model": "o4-mini",
  "interpreter_generation": {
    "temperature": null,
    "token_limit_parameter": "max_completion_tokens",
    "token_limit": 4096,
    "reasoning_effort": "low"
  },
  "predictor_generation": {
    "temperature": null,
    "token_limit_parameter": "max_completion_tokens",
    "token_limit": 4096,
    "reasoning_effort": "low"
  }
}
```

The example budget is illustrative. `max_completion_tokens` includes reasoning and visible output, so even a short label may need more than the default budget. Review the provider's [Chat Completions parameter reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and model limits. These options configure Chat Completions; models requiring Responses or native Anthropic Messages need a custom backend.

`monotm check` and `monotm run --dry-run` display the exact parameters that will be sent for each role. Known incompatible combinations on official OpenAI and Anthropic endpoints are blocked before training or requests. These checks do not cover every model, proxy or alias; the provider validates other combinations during execution. Explicit values are never automatically removed or substituted. If the provider rejects a parameter, inspect the raw response and adjust the indicated generation setting.

The same settings are available as CLI overrides on `check`, `run`, and `retry`: `--interpreter-temperature`, `--interpreter-token-limit`, `--interpreter-token-limit-parameter`, and `--interpreter-reasoning-effort`, plus the corresponding `--predictor-*` options. Use `--interpreter-temperature none` to omit temperature. For reasoning effort, `--interpreter-reasoning-effort omit` omits the field while `--interpreter-reasoning-effort none` sends the string `"none"`.

In Python, use `GenerationConfig` from `monotm.config` for either role, for example `RunConfig(interpreter_generation=GenerationConfig(temperature=None, token_limit_parameter="max_completion_tokens", token_limit=4096))`.

## Inspect and retry features without retraining

Use `inspect` to review feature labels, validation scores, and request logs. Use `retry` to regenerate labels or predictions from saved interpretation activations and **Θ**. After validation, **B** is reestimated using the updated labels and scores while Θ remains fixed.

First inspect the candidate IDs and scores, then inspect a particular feature (replace `17` with an ID from your run):

```bash
monotm inspect --run runs/my_corpus
monotm inspect --run runs/my_corpus --feature 17
# Export feature diagnostics as JSON:
monotm inspect --run runs/my_corpus --feature 17 --output feature_17.json
```

The feature view includes its label, support, example document IDs, predictions, IS, model provenance, and request attempts. Each attempt contains the **submitted prompt**, actual request parameters, model and endpoint, response text, raw HTTP response body, finish reason and token usage (when supplied by the provider), and error/status. Predictor attempts also identify the document. `finish_reason: "length"` is recorded as `status: "truncated"`, with feature `error_kind: "token_limit"`. Such replies are not cached or accepted, even when they contain a parseable label. Increase that role's token budget or select a supported lower reasoning effort when retrying. Network failures may have no response body. Credentials are omitted or redacted.

Retry particular features, optionally changing either model:

```bash
monotm retry --run runs/my_corpus --input my_documents.csv --features 17 29 \
  --interpreter-model your-interpreter-model --predictor-model your-predictor-model \
  --max-tokens 2048
```

Model options default to the models recorded in the run. The default endpoint is DeepInfra (`https://api.deepinfra.com/v1/openai`), with Interpreter `google/gemma-3-27b-it` and Predictor `google/gemma-3-12b-it`. Add `--base-url` and set `MONOTM_API_KEY` to use another OpenAI-compatible provider. Overrides apply to the selected features in the current retry operation and are recorded in feature metadata and recovery history. Supply the overrides again in subsequent commands to use the same replacement models.

Generation overrides inherit unspecified fields from the original run and are also recorded per role. To repair a parameter rejection without repeating SAE/LDA training:

```bash
monotm retry --run runs/my_corpus --input my_documents.csv --failed \
  --interpreter-temperature none \
  --interpreter-token-limit-parameter max_completion_tokens \
  --interpreter-token-limit 4096
```

Apply Predictor overrides too if its model requires them. `--max-tokens 2048` remains a shorthand to change both token budgets, using each role's selected token-limit field. Do not combine it with a stage-specific token-limit override. Increasing a budget may increase API cost; it is never done automatically.

| Selection or option | Behavior |
|---|---|
| `--features 17 29` | Fresh Interpreter and Predictor requests for these IDs; bypass accepted cached replies by default |
| `--failed` | Resume every unfinished feature, including pending/cancelled ones; reuse accepted calls by default |
| `--stage predictor` | Keep the existing labels and rerun their Predictor checks |
| `--refresh` | Force new requests even when accepted cache entries exist |
| `--reuse-cache` | Reuse matching accepted responses; useful when continuing a partial repair |
| `--label "your revised label"` | Supply a label for exactly one feature; always validate it with the Predictor |

```bash
# Resume an interrupted LLM stage; successful identical requests are reused.
monotm retry --run runs/my_corpus --input my_documents.csv --failed

# Keep a label but change the model used to check it.
monotm retry --run runs/my_corpus --input my_documents.csv --features 17 \
  --stage predictor --predictor-model your-predictor-model

# Replace a label yourself, then validate it.
monotm retry --run runs/my_corpus --input my_documents.csv --features 17 \
  --label "complaints about public transportation reliability"
```

`--failed` targets request failures and unfinished work, not low-scoring but successfully validated labels. Choose low-IS features explicitly with `--features`. There is no automatic model fallback: you select a model after inspecting the output. A syntactically valid but poor label can be retried because explicit selection bypasses its accepted cache by default.

Resuming with `--failed` preserves manually supplied labels whose Predictor checks were interrupted. If validation already finished and only descriptor estimation was interrupted, it rebuilds descriptors without making LLM requests or requiring an API key.

Supply the CSV used for the run; its text, IDs, and row order are verified before modifying results. Recorded text/ID column names are used automatically and can be overridden with `--text-column` and `--id-column`. Labeling and validation document sets are preserved across retries, and each new label is revalidated.

Previous feature records and descriptor outputs are archived in `cache/repairs/<operation_id>/before.json` and `before/`. Stale descriptor files are removed while repairing; if requests fail or other candidates remain pending, no current B/topics report is published. Run `retry --failed` to finish. If no labels reach the original IS cutoff, the new scores remain available for another repair. A successful repair updates `beta.npy`, `vocabulary.csv`, topics, and scores; the SAE/LDA/Θ files are checked for unchanged contents. A normal identical `run` command then reuses the repaired results.

For a notebook workflow, open [examples/inspect_and_retry.ipynb](examples/inspect_and_retry.ipynb), or call the same functions directly:

```python
from monotm import inspect_run, retry_features

details = inspect_run("runs/my_corpus", feature_id=17)
print(details["features"][0])
for attempt in details["attempts"]:
    print(attempt["context"], attempt["status"])
    print(attempt["request"]["messages"][0]["content"])
    print(attempt.get("raw_response"))

updated = retry_features("runs/my_corpus", "my_documents.csv", feature_ids=[17],
                         interpreter_model="your-interpreter-model",
                         predictor_model="your-predictor-model")
```

Custom `chat_client` backends used through the Python API must implement `complete(prompt, model, *, validator=None, context=None, refresh=False)`. The built-in `ChatClient` applies the generation settings and provides disk caching and raw attempt history; custom clients manage their own generation parameters and response history.

## Small synthetic example

```bash
python examples/generate_sample.py --output sample.csv --documents 180
monotm run --input sample.csv --id-column id --topics 3 \
  --config examples/small.json --output runs/sample
```

The generator creates synthetic documents. This example uses reduced training and validation settings with the configured embedding model and chat provider.

### Local CPU example

This profile runs [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings and [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) locally on CPU. It downloads approximately 1.1 GB of weights on the first run. The example chat server processes one request at a time and binds to `127.0.0.1`.

In terminal 1, activate the environment where you installed MonoTM, then:

```bash
python -m pip install -e '.[embeddings]'
python examples/local_chat_server.py
# Wait for "Ready at http://127.0.0.1:8765/v1".
```

In terminal 2, activate the same environment, then:

```bash
python examples/generate_sample.py --output cpu_sample.csv --documents 60
MONOTM_API_KEY=local-smoke OMP_NUM_THREADS=4 monotm run \
  --input cpu_sample.csv --id-column id --topics 3 \
  --config examples/cpu_smoke.json --output runs/cpu_smoke
monotm inspect --run runs/cpu_smoke
```

`local-smoke` is a dummy credential for this loopback server; no cloud key is needed. On the first run, downloads and CPU generation take time. The server caps generated outputs at 128 tokens by default; increase `--max-new-tokens` if a reply is truncated.

For Interpreter requests, the server includes a system instruction requesting a short `FINAL:` label. The instruction is recorded in the raw response metadata. Predictor requests use the bundled Predictor prompt.

The profile trains both SAEs for at most 300 steps, fits 3 topics, and validates at most 2 features with up to 3 active and 3 inactive documents each. Its **IS cutoff is 0.5**, compared with the default 0.8. These reduced settings are intended for checking execution; use larger corpora and suitable model settings to assess topic quality. Both LLM roles use the same model in this example.

The regular `inspect`/`retry` workflow works against this server:

```bash
# Replace 17 with an actual candidate ID printed by inspect.
MONOTM_API_KEY=local-smoke monotm retry --run runs/cpu_smoke \
  --input cpu_sample.csv --features 17 --stage predictor
```

Keep terminal 1 running while testing; stop it with Ctrl+C when finished. A repeated identical `run` command reuses embeddings, training, and accepted validation. Choose a different output directory to test a fresh complete run.

## Python interface

```python
from monotm import run
from monotm.config import RunConfig

result = run("my_documents.csv", "runs/python", config=RunConfig(n_topics=20))
print(result.theta.shape, result.beta.shape)
print(result.topics[0])
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers SAE training, LDA, fixed-mixture EM, ID alignment, validation sampling, response parsing, caching, retries, setup checks, and device selection. Tests use synthetic embeddings, mocked HTTP responses, and mocked GPU probes without model downloads or external inference requests.

## Authors

Una Joh (`sjoh01@syr.edu`) and Bei Yu (`byu@syr.edu`), School of Information Studies, Syracuse University.
