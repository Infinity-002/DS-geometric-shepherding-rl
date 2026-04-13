"""Research pipeline helpers for shepherding experiments."""

from shepherding.research.benchmark import run_benchmark
from shepherding.research.callbacks import (
    AdaptiveCurriculumCallback,
    LinearCurriculumCallback,
    ResearchMetricsCallback,
    build_curriculum_callback,
)
from shepherding.research.evaluation import (
    EpisodeSummary,
    aggregate_results,
    collect_episode,
    create_significance_table,
    evaluate_scenarios,
)
from shepherding.research.io import (
    load_yaml_config,
    resolve_project_path,
    save_rows,
    save_summaries,
    write_json,
)
from shepherding.research.models import (
    build_feedforward_model,
    build_recurrent_model,
    load_model,
    make_research_env,
)

__all__ = [
    "AdaptiveCurriculumCallback",
    "LinearCurriculumCallback",
    "ResearchMetricsCallback",
    "EpisodeSummary",
    "aggregate_results",
    "build_curriculum_callback",
    "build_feedforward_model",
    "build_recurrent_model",
    "collect_episode",
    "create_significance_table",
    "evaluate_scenarios",
    "load_model",
    "load_yaml_config",
    "make_research_env",
    "resolve_project_path",
    "run_benchmark",
    "save_rows",
    "save_summaries",
    "write_json",
]
