"""Research pipeline helpers for shepherding experiments."""

from shepherding.research.benchmark import run_benchmark
from shepherding.research.callbacks import (
    AdaptiveCurriculumCallback,
    GeneralizationEvalCallback,
    LinearCurriculumCallback,
    ResearchMetricsCallback,
    StageTracker,
    build_curriculum_callback,
    collect_stage_summary,
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
from shepherding.research.reporting import (
    GRADED_AGGREGATE_COLUMNS,
    GRADED_EPISODE_COLUMNS,
    available_aggs,
    generalization_gap_table,
    holdout_splits,
    preferred_holdout_split,
    warn_if_success_is_degenerate,
)
from shepherding.research.models import (
    build_feedforward_model,
    build_recurrent_model,
    load_model,
    load_obs_normalizer,
    load_vecnormalize,
    make_research_env,
    make_research_vec_env,
    save_vecnormalize,
)
from shepherding.research.training import train_v3

__all__ = [
    "AdaptiveCurriculumCallback",
    "GRADED_AGGREGATE_COLUMNS",
    "GRADED_EPISODE_COLUMNS",
    "available_aggs",
    "generalization_gap_table",
    "holdout_splits",
    "preferred_holdout_split",
    "warn_if_success_is_degenerate",
    "GeneralizationEvalCallback",
    "StageTracker",
    "collect_stage_summary",
    "load_obs_normalizer",
    "load_vecnormalize",
    "make_research_vec_env",
    "save_vecnormalize",
    "train_v3",
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
