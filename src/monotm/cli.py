"""Command-line entry point; --help does not load ML libraries or models."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from . import __version__
from .config import RunConfig


def _run_options(run, *, required):
    run.add_argument("--input", required=required, type=Path, help="UTF-8 CSV with one nonempty document per row")
    run.add_argument("--output", required=required, type=Path, help="Empty directory or matching existing run")
    run.add_argument("--topics", required=required, type=int, help="Number of topics; overrides JSON n_topics")
    run.add_argument("--text-column", default="text")
    run.add_argument("--id-column", help="Optional unique string ID column; otherwise generated from row order")
    run.add_argument("--config", type=Path, help="JSON config; --topics and explicitly supplied options override it")
    run.add_argument("--embeddings", type=Path, help="Optional precomputed .npy document matrix")
    run.add_argument("--embedding-ids", type=Path, help="Optional JSON array of embedding IDs for explicit row alignment")
    run.add_argument("--embedding-model")
    run.add_argument("--device", help="auto, cpu, cuda, cuda:0, or another PyTorch device")
    run.add_argument("--base-url", help="OpenAI-compatible chat base URL")
    run.add_argument("--interpreter-model")
    run.add_argument("--predictor-model")
    run.add_argument("--workers", type=int, help="Parallel feature validations")
    run.add_argument("--max-features", type=int, help="Validate at most this many features, ranked by support")


def _resolve_config(args):
    config = RunConfig.from_dict(json.loads(args.config.expanduser().read_text(encoding="utf-8"))) if args.config else RunConfig()
    for arg, target in (("topics", "n_topics"), ("embedding_model", "embedding_model"), ("device", "device"),
                        ("base_url", "base_url"), ("interpreter_model", "interpreter_model"),
                        ("predictor_model", "predictor_model"), ("workers", "llm_workers"), ("max_features", "max_features")):
        value = getattr(args, arg)
        if value is not None:
            setattr(config, target, value)
    config.validate()
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="monotm", description="Topic modeling with full-feature mixtures and validated semantic descriptors")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    defaults = commands.add_parser("config", help="Print the default JSON configuration")
    defaults.add_argument("--output", type=Path, help="Write JSON to a new file instead of stdout")
    check = commands.add_parser("check", help="Review effective settings, GPUs, credentials and inputs without executing")
    _run_options(check, required=False)
    check.add_argument("--json", action="store_true", help="Print the complete report as JSON; credentials are omitted")
    run = commands.add_parser("run", help="Show setup and fit MonoTM to your CSV; matching runs resume automatically")
    _run_options(run, required=True)
    run.add_argument("--dry-run", action="store_true", help="Only check and display settings; no writes, downloads or API calls")
    run.add_argument("--json", action="store_true", help="JSON setup report; requires --dry-run")
    inspect = commands.add_parser("inspect", help="Inspect feature status, prompts and raw LLM replies without API calls")
    inspect.add_argument("--run", required=True, type=Path)
    inspect.add_argument("--feature", type=int, help="Show the full record and all LLM attempts for this feature ID")
    inspect.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    inspect.add_argument("--output", type=Path, help="Export diagnostics JSON to a new file")
    retry = commands.add_parser("retry", help="Repair selected LLM stages and rebuild descriptors, reusing SAE/LDA")
    retry.add_argument("--run", required=True, type=Path)
    retry.add_argument("--input", required=True, type=Path, help="CSV used for the run; verified against its saved fingerprint")
    selection = retry.add_mutually_exclusive_group(required=True)
    selection.add_argument("--features", nargs="+", type=int, help="Feature IDs to retry; accepted caches are bypassed by default")
    selection.add_argument("--failed", action="store_true", help="Resume all failed/pending features, reusing accepted cached calls")
    retry.add_argument("--stage", choices=("both", "predictor"), default="both",
                       help="Regenerate and validate labels (both), or revalidate existing labels (predictor)")
    retry.add_argument("--interpreter-model")
    retry.add_argument("--predictor-model")
    retry.add_argument("--base-url", help="OpenAI-compatible chat base URL; use MONOTM_API_KEY for its credential")
    retry.add_argument("--max-tokens", type=int, default=1024, help="Output-token limit per request (default: 1024)")
    retry.add_argument("--label", help="Manual label for exactly one --features ID; always runs Predictor validation")
    retry.add_argument("--text-column", help="Defaults to the column recorded in run.json")
    retry.add_argument("--id-column", help="ID column override; defaults to the column recorded in run.json")
    caching = retry.add_mutually_exclusive_group()
    caching.add_argument("--refresh", action="store_true", dest="refresh", default=None, help="Bypass accepted cache entries")
    caching.add_argument("--reuse-cache", action="store_false", dest="refresh", help="Reuse accepted calls when resuming")
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            text = json.dumps(RunConfig().to_dict(), indent=2) + "\n"
            if args.output:
                with args.output.open("x", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                print(text, end="")
            return 0
        if args.command == "inspect":
            from .recovery import inspect_run
            result = inspect_run(args.run, feature_id=args.feature)
            text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            if args.output:
                with args.output.open("x", encoding="utf-8") as handle:
                    handle.write(text)
                print(f"Saved diagnostics to {args.output}")
            elif args.json or args.feature is not None:
                print(text, end="")
            else:
                print(f"Run: {result['status']}; feature counts: {result['status_counts']}")
                print(f"IS cutoff: {result['is_threshold']}; low-scoring features: {result['low_score_feature_ids']}")
                print("feature  status                  IS     label")
                for record in result["features"]:
                    print(f"{record['feature_id']:7}  {record['status']:22}  {record.get('is_score', 0):.3f}  {record.get('label', '')}")
            return 0
        if args.command == "retry":
            from .recovery import retry_features
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            result = retry_features(args.run, args.input, feature_ids=args.features, failed=args.failed,
                                    stage=args.stage, interpreter_model=args.interpreter_model,
                                    predictor_model=args.predictor_model, base_url=args.base_url,
                                    refresh=args.refresh, max_tokens=args.max_tokens, label=args.label,
                                    text_column=args.text_column, id_column=args.id_column)
            print(f"Rebuilt topic descriptors in {result['output_dir']}; SAE/LDA and theta.npy reused")
            return 0
        config = _resolve_config(args)
        if args.command == "run" and args.json and not args.dry_run:
            raise ValueError("--json on run requires --dry-run; use check --json to review settings")
        from .preflight import check_run, format_report
        report = check_run(args.input, args.output, config=config, text_column=args.text_column,
                           id_column=args.id_column, embeddings_path=args.embeddings, embedding_ids_path=args.embedding_ids)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report), flush=True)
        if not report["ready"]:
            return 1
        if args.command == "check" or args.dry_run:
            return 0
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        from .pipeline import run as run_pipeline
        result = run_pipeline(args.input, args.output, config=config, text_column=args.text_column,
                              id_column=args.id_column, embeddings_path=args.embeddings, embedding_ids_path=args.embedding_ids)
        print(f"Saved {len(result.topics)} topics to {result.output_dir / 'topics.json'}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to reuse finished stages and cached LLM responses.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"monotm: {exc}", file=sys.stderr)
        return 1
