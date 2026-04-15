# Phase D follow-up: Moondream2 reranker, coexistence, no CPU spill

## Why

Phase D landed the reranker architecture (persistent subprocess, listwise
prompt, integration into the search path) but we could not actually
measure its effect on the reference 8 GB card. With ColQwen3 resident at
~5 GiB, Qwen2.5-VL-3B at bnb 4-bit does not fit alongside it; the
subprocess falls back to CPU and a single rerank call takes >5 minutes,
which is not a benchmark, it is a hang.

The fix we want: keep Qwen2.5-VL-3B-Instruct where it already works
well (the topic chunker) and use a much smaller VLM for the reranker
that can genuinely coexist with ColQwen3 on the same GPU. Target:
**`vikhyatk/moondream2`**, quantized to int4, ~0.8 GiB weights + a few
hundred MiB of activation for a 10-candidate rerank. Hard rule: the
reranker subprocess must run on GPU or refuse to run at all — no CPU
spill, no silent fallback that makes the benchmark unusable.

## Non-negotiables

- **Chunker stays on Qwen2.5-VL-3B-Instruct**. Phase B's quality work
  is tied to that prompt. Do not touch `vlm_topic_client.py`.
- **Reranker model is configurable and distinct from the chunker
  model**. `settings.reranker_model_id` with default
  `"vikhyatk/moondream2"`, separate from `vlm_model_id`.
- **No CPU spill for the reranker**. If the reranker cannot load on
  GPU, the worker exits with a fatal status line and the search path
  logs once and falls back to MaxSim-only ranking for the session.
- **Both models coexist under the 8 GB ceiling with a busy desktop**.
  Rough budget: 1.5 GiB desktop + 2.5 GiB ColQwen3 resident + 1.5 GiB
  Moondream2 int4 resident + ~1 GiB peak activation = 6.5 GiB. Comfortable.

## Scope

1. **New reranker model hook in `vlm_loader.py`**
   - Add `load_moondream_reranker()` that builds a `VLMHandle` using
     the moondream2 HF model id. Moondream2 uses a custom
     `trust_remote_code=True` class; we already have the fallback
     chain `AutoModelForImageTextToText → AutoModelForVision2Seq →
     AutoModel`, so the existing `load_vlm()` should work with only
     the model id swapped.
   - Hard failure if the handle lands on anything other than
     `"cuda"`. Raise a clear `RuntimeError` and let the worker emit
     `{"status": "fatal", "error": ...}` on stdout.

2. **Settings change**
   - Add `reranker_model_id: str = "vikhyatk/moondream2"`.
   - Keep `reranker_model: str = "none"` as the on/off switch.
   - Add `reranker_require_gpu: bool = True` so the CPU-spill
     fallback is opt-in for power users with huge RAM.

3. **Rerank worker**
   - `ragrag/retrieval/vlm_rerank_worker.py` already takes
     `--model-id` from argv. Wire `settings.reranker_model_id` into
     the parent-side spawn in `reranker.py`.
   - Update the worker's prompt builder: Moondream2's instruction
     tuning is looser than Qwen2.5-VL's, so the prompt should be
     **shorter, more imperative, and explicitly demand the JSON
     array** (the existing prompt already does the latter; we only
     need to trim the preamble).
   - Honor the `reranker_require_gpu` flag: if `handle.device !=
     "cuda"`, emit `{"status": "fatal", ...}` and exit non-zero. The
     parent already treats fatal as "worker unusable, disable
     reranker for the rest of the session".

4. **Placement probe**
   - Reuse `plan_vlm_placement()` from `vlm_loader.py` but with a
     tighter budget sized for Moondream2 (requires ~2 GiB free, not
     ~3 GiB). Call it from the worker *after* loading its own
     weights but before accepting the first rerank command — if
     free VRAM after our own load is below a safety margin
     (`moondream_activation_headroom_mib`, default 512), log a
     warning and continue, but do not fall back.

5. **Tests**
   - `tests/test_reranker_worker_moondream_dispatch.py`: monkeypatch
     `load_vlm` to assert it is called with the moondream model id
     when `reranker_model_id` is set.
   - `tests/test_reranker_gpu_required.py`: simulate a non-cuda
     handle and assert the worker emits a fatal status line.
   - Existing `tests/test_reranker.py` stays as-is — it exercises
     the parent protocol with a fake worker and does not care which
     model the real worker would have loaded.

6. **Benchmark**
   - Re-run `scripts/benchmark_stm32h743.py --warm-only` against the
     existing `/tmp/stm32h743_bench` index with
     `reranker_model = "vlm"` in `ragrag.json`, capture
     `validation/benchmarks/phase-D.json`, diff against phase-B.
   - Gate: if phase-D does not beat phase-B on Sem@5 by at least
     +0.05, we keep the code but leave `reranker_model` defaulted
     to `"none"` and document the result in the commit message —
     same stop-loss the roadmap already called out.

## Out of scope

- Downloading moondream2 weights. The user runs
  `scripts/fetch_validation_data.py` style commands on their own
  once we add the model id; we do not bundle HF downloads inside
  the test suite.
- GPU-coexistence for the chunker. The chunker is a one-shot during
  indexing; the embedder is already unloaded by then via the
  subprocess isolation story in Phase B. No change needed there.
- Any BGE-reranker / text-only cross-encoder path. We explicitly
  picked a visual reranker; falling back to a text-only model is a
  separate decision the user has not signed off on.

## Risks

- **Moondream2 quality on dense datasheets is unknown**. We chose it
  for size, not for table reading. If it ranks worse than the
  MaxSim baseline we keep it behind the opt-in flag and document.
- **Non-PyTorch CUDA state** from moondream2 + ColQwen3 may still
  fragment the allocator in ways that break the embedder's next
  forward pass. Mitigation: subprocess isolation already protects
  the parent from the reranker's CUDA context, so the only risk is
  the reranker's own subsequent calls. If those get flaky, the
  next step is to teach the subprocess to `torch.cuda.empty_cache`
  between calls (cheap) and bail to fatal if that is not enough.
