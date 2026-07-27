#!/usr/bin/env python3
"""FPT 2026 Track A — Agent-based HLS code generation + iterative repair with budget control."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bench4hls.settings import BenchConfig, load_config
from bench4hls.agent.agent_runner import AgentRunner, AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench4hls.agent")

SCRIPT_DIR = Path(__file__).parent.resolve()
BENCHMARK_DIR = SCRIPT_DIR / "benchmark"
TB_DIR = BENCHMARK_DIR / "testbenches"
TCL_DIR = SCRIPT_DIR / "tcl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bench4HLS Agent — Iterative HLS code generation with budget-controlled repair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["huggingface", "ollama"], required=True,
                   help="LLM inference backend")
    p.add_argument("--config", default="bench4hls_config.json",
                   help="Path to bench4hls_config.json (default: bench4hls_config.json)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: from config)")
    p.add_argument("--csim-budget", type=int, default=500,
                   help="Total C-simulation call budget (default: 500)")
    p.add_argument("--synth-budget", type=int, default=500,
                   help="Total synthesis call budget (default: 500)")
    p.add_argument("--token-budget", type=int, default=2_000_000,
                   help="Total token budget (default: 2,000,000)")
    p.add_argument("--max-attempts", type=int, default=5,
                   help="Max attempts per task (default: 5)")
    p.add_argument("--max-retries-per-type", type=int, default=2,
                   help="Max retries per error type (default: 2)")
    p.add_argument("--budget-preset", choices=["default", "aggressive", "conservative"],
                   default="default",
                   help="Per-task budget preset (default: default)")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task list (e.g. Prob001,Prob002). Default: all tasks.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip hardware evaluation (dry-run mode)")
    p.add_argument("--ppa-threshold", type=float, default=0.05,
                   help="PPA improvement threshold for optimization (default: 0.05)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()

    if ns.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load Bench4HLS config (for model, Xilinx paths, etc.)
    try:
        cfg_path = Path(ns.config)
        if not cfg_path.is_absolute():
            cfg_path = SCRIPT_DIR / cfg_path
        cfg = load_config(cfg_path.parent, raw=json.loads(cfg_path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    # Initialize backend
    if ns.backend == "huggingface":
        from bench4hls.backends import HuggingFaceBackend
        backend = HuggingFaceBackend(
            model_name=cfg.model,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            use_4bit=cfg.hf_use_4bit,
            max_seq_length=cfg.hf_max_seq_length,
            hf_token=cfg.hf_token,
            hf_endpoint=cfg.hf_endpoint,
        )
    else:
        from bench4hls.backends import OllamaBackend
        backend = OllamaBackend(
            model_name=cfg.model,
            host=cfg.ollama_host,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            timeout=cfg.ollama_timeout,
        )

    # Workdir
    workdir = Path(ns.output_dir) if ns.output_dir else cfg.output_dir
    if workdir is None:
        slug = cfg.model.replace("/", "_").replace(":", "_").replace(".", "_")
        workdir = SCRIPT_DIR / f"eval_{slug}"
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", workdir)

    # Agent config
    agent_config = AgentConfig(
        csim_budget=0 if ns.skip_eval else ns.csim_budget,
        synth_budget=0 if ns.skip_eval else ns.synth_budget,
        token_budget=ns.token_budget,
        max_attempts_per_task=ns.max_attempts,
        max_retries_per_error_type=ns.max_retries_per_type,
        ppa_threshold=ns.ppa_threshold,
        budget_preset=ns.budget_preset,
    )

    # Create AgentRunner
    runner = AgentRunner(
        backend=backend,
        config=agent_config,
        workdir=workdir,
        tb_dir=TB_DIR,
        tcl_dir=TCL_DIR,
        input_prompts=cfg.input_prompts,
        vitis_bin=None if ns.skip_eval else cfg.vitis_bin,
        vivado_bin=None if ns.skip_eval else cfg.vivado_bin,
        xilinx_version=cfg.xilinx_version,
        design_timeout=cfg.design_timeout_seconds,
        parallel_workers=cfg.parallel_workers,
    )

    # Optional: filter to specific tasks
    if ns.tasks:
        requested = set(ns.tasks.split(","))
        runner.tasks = {k: v for k, v in runner.tasks.items() if k in requested}
        if not runner.tasks:
            logger.error("No matching tasks found. Available: %s",
                         ", ".join(sorted(runner.tasks.keys()))[:200])
            sys.exit(1)
        logger.info("Filtered to %d tasks: %s", len(runner.tasks),
                     ", ".join(sorted(runner.tasks.keys())))

    # Run
    try:
        report = runner.run()
    finally:
        backend.close()

    # Print summary to stdout
    print(f"\n{'='*50}")
    print(f"  Agent Run Complete")
    print(f"{'='*50}")
    print(f"  Accepted:  {report['accepted']}/{report['total_tasks']}")
    print(f"  Skipped:   {report['skipped']}")
    print(f"  Exhausted: {report['exhausted']}")
    print(f"  Errors:    {report['errors']}")
    print(f"  Attempts:  {report['total_attempts']}")
    print(f"  CSIM:      {report['csim_used']}")
    print(f"  Synth:     {report['synth_used']}")
    print(f"  Tokens:    {report['tokens_used']}")
    print(f"  Time:      {report['elapsed_seconds']:.1f}s")
    print(f"{'='*50}")
    print(f"  Report:    {workdir / 'agent_report.json'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()