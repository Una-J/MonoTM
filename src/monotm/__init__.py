"""MonoTM: full-feature mixtures and validated semantic descriptions."""

__version__ = "0.1.0"


def check_run(*args, **kwargs):
    """Inspect configuration and local readiness without executing the pipeline."""
    from .preflight import check_run as check_pipeline
    return check_pipeline(*args, **kwargs)


def run(*args, **kwargs):
    """Run the pipeline; see monotm.pipeline.run for the Python interface."""
    from .pipeline import run as run_pipeline
    return run_pipeline(*args, **kwargs)


def inspect_run(*args, **kwargs):
    """Return feature records and raw attempts; see monotm.recovery.inspect_run."""
    from .recovery import inspect_run as inspect_pipeline
    return inspect_pipeline(*args, **kwargs)


def retry_features(*args, **kwargs):
    """Repair LLM stages without retraining; see monotm.recovery.retry_features."""
    from .recovery import retry_features as retry_pipeline
    return retry_pipeline(*args, **kwargs)
