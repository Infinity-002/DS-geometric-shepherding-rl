from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "images"


@dataclass(frozen=True)
class Asset:
    title: str
    filename: str
    caption: str


FEATURED_ASSETS = [
    Asset(
        title="Main Comparison Dashboard",
        filename="main_dashboard.png",
        caption=(
            "High-level comparison of the main methods across success rate, "
            "goal proximity, and episode return."
        ),
    ),
    Asset(
        title="Scenario Heatmaps",
        filename="scenario_heatmaps.png",
        caption=(
            "Scenario-by-scenario performance view showing where each method "
            "generalizes and where it struggles."
        ),
    ),
    Asset(
        title="Success Rate by Scenario",
        filename="success_rate_by_scenario.png",
        caption=(
            "Per-scenario success breakdown for presentation of seen versus unseen "
            "environment performance."
        ),
    ),
    Asset(
        title="Behavioral Cloning Offline Metrics",
        filename="bc_offline_metrics.png",
        caption=(
            "Offline validation metrics for the behavioral cloning baseline, "
            "useful for the data-science comparison slide."
        ),
    ),
]

DEMO_ASSETS = [
    Asset(
        title="PPO Herding v2 Demo",
        filename="ppo_herding_v2_demo.gif",
        caption="Animated rollout from the earlier PPO-based shepherding setup.",
    ),
    Asset(
        title="Structured RL 3D Demo",
        filename="v3_structured_3d.gif",
        caption="3D render of the structured v3 recurrent RL policy in action.",
    ),
    Asset(
        title="Behavioral Cloning 3D Demo",
        filename="bc_structured_3d.gif",
        caption="3D render of the behavioral cloning agent for direct visual comparison.",
    ),
]


def asset_path(filename: str) -> Path | None:
    path = IMAGE_ROOT / filename
    if path.exists():
        return path
    return None


@st.cache_data(show_spinner=False)
def discover_gallery_assets() -> list[Path]:
    if not IMAGE_ROOT.exists():
        return []
    valid_suffixes = {".png", ".gif", ".jpg", ".jpeg", ".webp"}
    assets = []
    for path in IMAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_suffixes:
            continue
        if "Zone.Identifier" in path.name:
            continue
        assets.append(path)
    return sorted(assets, key=lambda path: path.name.lower())


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(186, 95, 63, 0.18), transparent 26%),
                    radial-gradient(circle at top right, rgba(36, 104, 103, 0.16), transparent 24%),
                    linear-gradient(180deg, #f5efe4 0%, #efe6d7 100%);
                color: #201a16;
                font-family: "Avenir Next", "Trebuchet MS", sans-serif;
            }
            .hero-shell {
                padding: 1.6rem 1.8rem;
                border-radius: 24px;
                background: rgba(255, 252, 246, 0.78);
                border: 1px solid rgba(77, 53, 36, 0.12);
                box-shadow: 0 18px 44px rgba(74, 54, 40, 0.10);
                margin-bottom: 1rem;
            }
            .hero-kicker {
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9a4e2f;
                font-size: 0.78rem;
                margin-bottom: 0.4rem;
                font-weight: 700;
            }
            .hero-title {
                font-size: 2.5rem;
                line-height: 1.0;
                margin: 0;
                color: #1f1915;
            }
            .hero-copy {
                margin-top: 0.8rem;
                max-width: 48rem;
                color: #4b3b31;
                font-size: 1.05rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 252, 246, 0.82);
                border-radius: 20px;
                border: 1px solid rgba(77, 53, 36, 0.10);
                padding: 0.75rem 1rem;
                box-shadow: 0 12px 28px rgba(74, 54, 40, 0.08);
            }
            .asset-note {
                padding: 0.95rem 1rem;
                border-left: 4px solid #b55d37;
                background: rgba(255, 252, 246, 0.78);
                border-radius: 14px;
                color: #483930;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_asset(asset: Asset, *, use_container_width: bool = True) -> None:
    path = asset_path(asset.filename)
    if path is None:
        st.warning(f"Missing asset: images/{asset.filename}")
        return
    st.subheader(asset.title)
    st.image(str(path), use_container_width=use_container_width)
    st.caption(asset.caption)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Presentation Dashboard</div>
            <h1 class="hero-title">Geometric Shepherding RL</h1>
            <p class="hero-copy">
                Geometric-informed reinforcement learning for shepherding under
                partial observability, shown through exported results and rollout demos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Geometric Shepherding Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()

    gallery_assets = discover_gallery_assets()
    asset_registry = {asset.filename: asset for asset in FEATURED_ASSETS + DEMO_ASSETS}

    render_hero()

    if not IMAGE_ROOT.exists():
        st.error("The `images/` directory was not found.")
        return
    if not gallery_assets:
        st.warning("No supported presentation assets were found in `images/`.")
        return

    overview_tab, graphs_tab, demos_tab, gallery_tab = st.tabs(
        ["Overview", "Graphs", "GIF Demos", "Gallery"]
    )

    with overview_tab:
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            render_asset(FEATURED_ASSETS[0])
        with right:
            # st.markdown(
            #     """
            #     <div class="asset-note">
            #         A single-screen view of the project: benchmark comparison,
            #         scenario-level generalization, and rollout behavior.
            #     </div>
            #     """,
            #     unsafe_allow_html=True,
            # )
            if asset_path("scenario_heatmaps.png") is not None:
                render_asset(FEATURED_ASSETS[1])

    with graphs_tab:
        available_featured = [asset for asset in FEATURED_ASSETS if asset_path(asset.filename)]
        if not available_featured:
            st.info("No featured graph assets were found in `images/`.")
        else:
            selected_title = st.selectbox(
                "Figure",
                options=[asset.title for asset in available_featured],
            )
            spotlight = next(
                asset for asset in available_featured if asset.title == selected_title
            )
            render_asset(spotlight)

            st.divider()
            st.markdown("### More Results")
            remaining = [asset for asset in available_featured if asset.title != selected_title]
            if remaining:
                columns = st.columns(2, gap="large")
                for idx, asset in enumerate(remaining):
                    with columns[idx % 2]:
                        render_asset(asset)
            else:
                st.info("Only one featured graph is currently available.")

    with demos_tab:
        available_demos = [asset for asset in DEMO_ASSETS if asset_path(asset.filename)]
        if not available_demos:
            st.info("No GIF demos were found in `images/`.")
        else:
            demo_columns = st.columns(min(3, len(available_demos)), gap="large")
            for idx, asset in enumerate(available_demos):
                with demo_columns[idx % len(demo_columns)]:
                    render_asset(asset)

    with gallery_tab:
        st.markdown("### Asset Gallery")
        selected_file = st.selectbox(
            "Asset",
            options=[path.name for path in gallery_assets],
        )
        selected_path = next(path for path in gallery_assets if path.name == selected_file)
        st.image(str(selected_path), use_container_width=True)
        selected_asset = asset_registry.get(selected_path.name)
        if selected_asset is not None:
            st.caption(selected_asset.caption)


if __name__ == "__main__":
    main()
