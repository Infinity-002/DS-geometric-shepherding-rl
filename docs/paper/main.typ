#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Geometric-Informed Recurrent Reinforcement Learning for Partially Observable Shepherding],
  abstract: [
    The shepherding problem asks a single herding agent to drive a flock of interacting, non-cooperative targets into a goal region.
    Classical collect-and-drive heuristics reproduce empirical sheep--dog behavior, but they degrade when sensing is local, the workspace contains obstacles, and the flock can split.
    This paper studies a geometric-informed reinforcement-learning (RL) alternative.
    We simulate Strömbom-style sheep on a continuous two-dimensional field, expose the dog to a range-limited observation in which occluded sheep are masked, and train a recurrent proximal policy optimization (PPO) controller with an explicitly geometric reward: centroid-to-goal progress, convex-hull compactness, an incursion penalty, a drive-position bonus, visibility maintenance, stray and collision costs, and a terminal success bonus.
    Training uses an adaptive curriculum over domain-randomized goals, obstacle layouts, visibility radii, and flock dynamics, with a structured-obstacle variant that spawns the flock opposite the goal.
    The learned policy is compared with a cluster-aware collect-and-drive heuristic and with random-forest behavioral cloning of that heuristic.
    On the training distribution, structured recurrent PPO reaches a 75% success rate versus 25% for both baselines and a higher episode return (584.8 versus about 531).
    The same policy fails on held-out split-field and open-field geometries (0% success), whereas the heuristic solves the open field completely (100%) and the cloning policy interpolates between the two (50% open-field success).
    Offline, the forest reconstructs expert actions with validation $R^2$ of 0.59--0.69 and a 29° mean heading error, but this does not yield reliable closed-loop herding.
    Feature importances concentrate on cluster geometry rather than raw obstacle coordinates.
    The results show that geometric reward shaping plus recurrence can solve in-distribution, partially observed herding, while out-of-distribution robustness remains dominated by the classical heuristic.
  ],
  authors: (
    (
      name: "Author One",
      department: [Department of Computer Science],
      organization: [University Name],
      location: [City, Country],
      email: "author.one@university.edu",
    ),
    (
      name: "Author Two",
      department: [Department of Computer Science],
      organization: [University Name],
      location: [City, Country],
      email: "author.two@university.edu",
    ),
    (
      name: "Author Three",
      department: [Department of Computer Science],
      organization: [University Name],
      location: [City, Country],
      email: "author.three@university.edu",
    ),
  ),
  index-terms: (
    "Shepherding",
    "Reinforcement learning",
    "Proximal policy optimization",
    "Partial observability",
    "Behavioral cloning",
    "Curriculum learning",
    "Geometric reward shaping",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Shepherding is the problem of using one or a few active agents to move a larger group of passive, interacting targets into a designated region @lien2004 @strombom2014.
The same structure appears in livestock robotics @king2023, crowd guidance, environmental clean-up, and the indirect control of robot swarms @vaughan2000 @pierson2018.
A single sheepdog solving the task in the field is a canonical example: the dog never transports a sheep, it only shapes the flock's own dynamics.

Strömbom _et al._ showed that a surprisingly small program---collect the flock when it is too dispersed, then drive it from behind toward the goal---reproduces GPS traces of working dogs @strombom2014.
That heuristic, and its later robotic descendants @strombom2018, assumes that the herder can see the whole flock and that the workspace is comparatively open.
Those assumptions fail as soon as visibility is limited, obstacles fracture line of sight, or the flock splits into spatially separated clusters.
Hand-tuning a collect-and-drive rule for every geometry is brittle; learning a policy from interaction is attractive, but the resulting control problem is a partially observable Markov decision process (POMDP) with sparse success, long horizons, and a many-body plant.

This paper treats that setting as a geometric control problem and asks how far a shaped reward plus a recurrent policy can go relative to the classical heuristic and to a supervised clone of that heuristic.
The contributions are:

- A Gymnasium environment @towers2024 in which sheep obey a Strömbom-style force model with obstacle avoidance, the dog has a finite visibility radius, and unseen sheep are replaced by a sentinel in a fixed-size observation vector.
- A composite geometric reward that scores centroid progress, convex-hull compactness, blocking (incursion), drive-position geometry, visibility, strays, and collisions, rather than a sparse goal indicator alone.
- A recurrent PPO agent @schulman2017 @raffin2021 trained with an adaptive performance curriculum and optional structured obstacle layouts.
- A data-science comparison against a cluster-aware collect-and-drive baseline and random-forest behavioral cloning @pomerleau1989 @breiman2001, including offline regression metrics and online herding success on seen and unseen scenarios.

Empirically, geometric recurrent PPO is the strongest _in-distribution_ controller in our suite (75% training success versus 25% for the heuristic and the clone) and the weakest _out-of-distribution_ one (0% on held-out split and open fields).
The heuristic, conversely, is weak on the cluttered training layout and essentially perfect in open space.
Behavioral cloning sits in between: it imitates actions with moderate fidelity, inherits some of the expert's open-field competence, and does not match either parent on the full task.
That train--test reversal is the central experimental finding.

= Related Work

== Heuristic shepherding

Reynolds' boids model established local attraction, alignment, and repulsion as a sufficient description of flocking @reynolds1987.
Shepherding inverts the usual cooperative-control question: a small number of herders must _exploit_ those local rules.
Lien _et al._ catalogued geometric herding primitives (collect, drive, cover) for motion planning @lien2004.
Vaughan _et al._ demonstrated automatic flock control with a mobile robot @vaughan2000.
The algorithm of Strömbom _et al._ remains the standard single-herder baseline: switch from collecting the furthest agent to driving the centroid once the flock is compact @strombom2014.
Pierson and Schwager give a control-theoretic treatment of non-cooperative herds @pierson2018, and King _et al._ survey biologically inspired robot herding @king2023.
Our heuristic baseline is a visibility- and obstacle-aware descendant of the Strömbom collect--drive switch, with an additional cluster-selection rule when the flock clearly splits.

== Learning-based shepherding

Deep RL has been applied to shepherding both as a replacement for the waypoint heuristic and as a hierarchical controller.
Hussein _et al._ use curriculum PPO to learn collect versus drive without an external waypoint generator @hussein2022aamas.
Hasan _et al._ map multi-shepherd cooperation onto a payload-protection RL formulation @hasan2022.
Recent work studies decentralized multi-herder policies for non-cohesive targets @napolitano2024.
Those papers typically assume broader sensing than we do, or they factor the problem into target assignment plus single-target driving.
We instead learn a single continuous steering command for one dog that must remember unseen sheep, thread axis-aligned obstacles, and keep a cohesive flock of ten.

== Reward shaping, recurrence, and imitation

Potential-based shaping can accelerate RL without changing the optimal policy in the fully observed case @ng2000; in practice, task-specific geometric terms are widely used to densify long-horizon control @sutton2018.
Partial observability is classically handled by recurrent critics and actors @hausknecht2015.
Domain randomization @tobin2017 and curriculum learning @narvekar2020 are the standard tools for making those policies less brittle.
On the supervised side, behavioral cloning @pomerleau1989 @hussein2017il is the simplest imitation baseline; we use a random forest @breiman2001 because the expert is itself a geometric program, so axis-aligned splits over engineered features are a natural hypothesis class.
The comparison in this paper is therefore three-way: a geometric program, a geometric regressor trained on that program, and a geometric-reward recurrent policy trained in the closed loop.

= Problem Formulation

== Environment

The workspace is a square $[0, G]^2$ with $G = 20$.
A dog $d_t in [0,G]^2$ and $N = 10$ sheep ${s^i_t}_{i=1}^N$ occupy the plane together with up to $M = 8$ axis-aligned rectangular obstacles.
The dog's action $a_t in [-1,1]^2$ is $ell_2$-normalized and scaled by a speed $v_d = 1$, then clipped to the largest collision-free prefix of the attempted step so that agents cannot tunnel through thin obstacles.
An episode succeeds when every sheep lies inside a ball of radius $rho = 2$ about a goal $g in [0,G]^2$, and is truncated after $T_max$ steps ($T_max = 650$ in the structured training regime).

#figure(
  image("figures/ppo_v2_still.png", width: 90%),
  caption: [Snapshot of the continuous herding environment.
  The dog (red square) has a finite visibility radius (dashed arc).
  Sheep inside that radius are observed as relative coordinates; sheep outside it are hidden.
  Brown rectangles are obstacles.
  The yellow star marks the goal region.
  The shaded polygon is the flock convex hull used both in the reward and as a compactness diagnostic.],
) <fig:env>

@fig:env shows the geometry.
The dog observes
$
  o_t = (d_t,; g - d_t,; {tilde(s)^i_t}_{i=1}^N,; {b_j}_{j=1}^M) in bb(R)^(56),
$
where $tilde(s)^i_t = s^i_t - d_t$ if $norm(s^i_t - d_t) <= R_"vis"$ and a sentinel value $999$ otherwise, and each obstacle $b_j$ is a normalized tuple $(x,y,w,h)/G$ (or $-1$ if unused).
The visibility radius is $R_"vis" = 8.5$ in the structured configuration and is randomized in $[5.5, 9.0]$ under domain randomization.
Because the observation is a fixed-length vector with masked entries, a memoryless policy cannot distinguish "sheep $i$ is behind the dog" from "sheep $i$ has left the field of view."
That is the motivation for a recurrent actor--critic.

== Flock dynamics

Each sheep is a kinematic agent whose velocity is a sum of local forces in the spirit of @strombom2014 @reynolds1987.
Let $c_t = (1/N) sum_i s^i_t$ be the flock centroid and let $ell_t$ be the sheep closest to $c_t$ (a de facto leader).
The force on sheep $i$ comprises:

- _Flee:_ a unit vector away from the dog if $norm(s^i_t - d_t) < R_"flee" = 5.5$.
- _Cohesion:_ $alpha (c_t - s^i_t)$ with $alpha = 0.07$.
- _Repulsion:_ a pairwise push from neighbors closer than a repulsion length (base value $1.0$).
- _Leader:_ $beta (ell_t - s^i_t)$ with $beta = 0.04$, zero for the leader itself.
- _Obstacle avoidance:_ a linearly decaying repulsion from nearby rectangle boundaries.
- _Noise:_ isotropic Gaussian jitter of standard deviation $0.02$.
- _Pressured goal-seek:_ a small term $gamma (g - s^i_t)$ with $gamma = 0.03$, applied only while the sheep is inside the flee radius.

The resulting vector is normalized to the sheep speed $v_s = 0.32$ (randomized under domain randomization) and then clipped to free space.
Sheep therefore never become fully passive: they cluster, avoid walls, and, when pressured, leak slightly toward the goal.
The dog's job is to apply that pressure from positions that keep the flock together.

== Objective

An episode is a trajectory $tau = (o_t, a_t, r_t)_{t=0}^{T}$.
The learning objective is the expected discounted return $bb(E)[sum_t gamma^t r_t]$ with $gamma = 0.99$, subject to the POMDP observation model above.
Success rate---the fraction of episodes in which all sheep enter the goal ball---is the primary deployment metric; return, terminal mean distance to the goal, hull area, stray count, and collision count are secondary.

= Geometric Reward and Learning Architecture

== Geometric reward

Sparse success is too rare to train from.
We therefore score every state with a sum of geometric terms, extending the four-term base reward used in the first version of the environment.

Let $Delta = G sqrt(2)$ be the workspace diagonal and let $H_t$ be the area of the flock convex hull (or the axis-aligned bounding box if the hull is degenerate).
The _base_ reward is
$
  r^"base"_t
  &= w_c (1 - norm(c_t - g)/Delta)
    + w_p (-H_t / G^2) \
  &+ w_i (-bb(1)[norm(d_t - g) < norm(c_t - g)])
    + w_n (1 - norm(d_t - c_t)/Delta),
$
with default weights $(w_c, w_p, w_i, w_n) = (1.0, 0.1, 0.5, 0.3)$.
The four terms have direct shepherding readings: move the centroid toward the goal, keep the flock compact, do not stand between the flock and the goal (the classical "incursion" error of a dog that overruns the group), and stay close enough to apply pressure.

The research environment adds denser, locally informed terms.
Write $delta_t = (1/N) sum_i norm(s^i_t - g)$ and $m_t = max_i norm(s^i_t - g)$, and let $nu_t$ be the fraction of sheep currently visible.
A _drive point_ $p_t$ is placed behind the centroid along the goal axis at distance $min(0.7 R_"vis", 0.9 R_"flee")$, which is the geometric locus of a textbook driving position @strombom2014.
The full per-step reward is
$
  r_t
  &= r^"base"_t
    + 3.0 (delta_(t-1) - delta_t)
    + w_w (m_(t-1) - m_t) \
  &+ 0.8 (nu_t - nu_(t-1))
    - 0.05 (1 - nu_t)
    - 0.25 bb(1)[nu_t = 0] \
  &- 0.05 n^"stray"_t
    + 0.35 (1 - norm(d_t - p_t)/Delta)
    + r^"coll"_t,
$
where $w_w = 4.0$ in the structured regime, $n^"stray"_t$ counts sheep farther than $1.8 rho$ from the centroid, and $r^"coll"_t$ is $-0.2$ on a newly started obstacle collision and $-0.03$ while the dog remains in contact.
A terminal success bonus of $+125$ is granted when every sheep is inside the goal ball.
The progress and worst-sheep terms are potential-like differences @ng2000; the visibility terms penalize losing the flock; the drive term shapes _where_ the dog should stand, not only what the flock does.

== Recurrent PPO

The policy is Recurrent PPO with an LSTM actor--critic (`MlpLstmPolicy` in `sb3-contrib` @raffin2021).
The structured configuration uses $1024$ rollout steps, batch size $128$, $10$ epochs, learning rate $3 times 10^(-4)$, GAE $lambda = 0.95$, clip $0.2$, entropy coefficient $0.005$, and LSTM hidden size $256$, for $5 times 10^5$ environment steps.
A feed-forward PPO variant with a larger rollout ($4096$ steps) exists in the same code path; all quantitative results below use the recurrent policy, which is the architecture matched to the masked observation.
Actions are continuous and unit-normalized, matching the environment's steering semantics.

== Adaptive curriculum and domain randomization

Training episodes sample a difficulty stage $sigma in {0.0, 0.33, 0.66, 1.0}$.
An adaptive callback advances $sigma$ only when a rolling window of recent episodes (window $30$, warmup $8$) jointly satisfies thresholds on success rate, mean visibility, collision-event count, progress reward, and a minimum fraction of the training budget.
Early stages freeze goal, obstacle, and dynamics randomization and only jitter visibility; later stages re-enable them.
Independently, domain randomization---when active---samples the goal in the far quadrant, the visibility radius, sheep speed, cohesion, repulsion, leader gain, and obstacle layouts.

The structured training preset used for the main results further replaces purely random rectangles with four fixed-shape layouts that are only translated, spawns the flock in the corner opposite the goal, and holds visibility, goal, and dynamics fixed so that the geometric signal is consistent.
That preset is the "RL (Structured v3)" agent in the figures.

== Cluster-aware heuristic

The expert implements a visibility-restricted collect-and-drive rule.
If no sheep is visible, it searches toward the last seen centroid, or laterally about the goal direction.
Otherwise it optionally partitions visible sheep into two clusters by a farthest-pair split and focuses on the cluster whose centroid is farther from the goal whenever that split exceeds a separation of $2.0$.
If the focused sheep are spread beyond $1.6 rho$, the dog aims for a collect point behind the furthest individual; otherwise it aims for a drive point behind the cluster centroid.
An obstacle-avoidance force is added in both modes.
This is the demonstrator for behavioral cloning and the non-learning baseline in the online benchmark.

== Behavioral cloning

Demonstrations are generated by rolling out the cluster-aware expert.
Each observation is expanded into a fixed-length feature vector: raw dog, goal, sheep, and obstacle coordinates, plus fifteen geometric summaries (visible count and ratio, centroid offset, mean/max distances to dog, goal, and centroid, hull area, per-axis spread, and focus-cluster statistics).
A multi-output random forest @breiman2001 with $300$ trees, maximum depth $18$, and minimum leaf size $2$ regresses the expert's $(Delta x, Delta y)$ command.
At inference the predicted vector is $ell_2$-normalized, matching the environment.
Offline metrics include mean squared, absolute, and angular error, and per-axis $R^2$.
Online, the forest is dropped into the same benchmark loop as the heuristic and the recurrent policy.

= Experimental Setup

== Scenarios

Evaluation uses a training scenario and two held-out families from a larger library (corridor, dense clutter, narrow gate, split field, open field).
The structured comparison reported here evaluates:

- `train`: structured obstacle layouts, opposite-goal spawn, the distribution on which RL is trained.
- `unseen_split_field`: horizontal barriers that separate the workspace into channels, encouraging the flock to fracture.
- `unseen_open_field`: empty workspace, reduced visibility ($R_"vis" = 5.6$), slightly faster sheep.

These two held-out layouts are intentionally unlike the four training rectangles: one is more fragmented, one has no clutter at all.
A stronger test would include the corridor, dense, and narrow-gate families; they are implemented in the environment but are not part of the structured evaluation figures.

== Protocols

Three online agents are compared: the cluster-aware heuristic, the random-forest clone, and structured recurrent PPO.
Figures in @fig:dashboard and @fig:heatmap come from that comparison.
A fourth, cheaper recurrent PPO trained for only $1.2 times 10^5$ steps with a smaller LSTM (hidden size $128$) is reported in @fig:fast as a sample-efficiency ablation; it solves none of the three scenarios.
Offline cloning metrics in @fig:bc match the demonstration corpus used for the forest (7473 training rows, 2527 validation rows, 20% episode-level hold-out).
A larger demonstration set (7983 / 2017) yields similar errors (RMSE $0.46$, mean angle error $34°$) and is the source of the feature-importance ranking in @tbl:importance.

Primary metrics are success rate and terminal mean distance of sheep to the goal.
Episode return is reported on the training scenario.
Because published evaluation budgets in the accompanying configs are small (on the order of six episodes per scenario), we treat the numbers as a directional benchmark rather than a multi-seed statistical claim.
All plotted agents use deterministic actions at test time.

= Results

== In-distribution performance

@fig:dashboard and @tbl:success summarize the structured comparison.
On the training layout, recurrent PPO succeeds in 75% of episodes, against 25% for both the heuristic and the clone.
Training return follows the same order: $584.8$ for RL versus $531.4$ (heuristic) and $531.8$ (clone).
The extra return is consistent with more frequent collection of the $+125$ success bonus and with the progress terms that fire when the flock actually moves.

#figure(
  image("figures/main_dashboard.png", width: 100%),
  caption: [Structured comparison dashboard.
  Recurrent PPO dominates the training distribution in success rate and episode return, but its unseen-scenario success collapses to zero.
  The heuristic is the most reliable agent once the evaluation leaves the training geometry.
  Mean training distances in this figure disagree in the assignment of $2.79$ versus $4.07$ with the per-scenario heatmap of @fig:heatmap; @tbl:success follows the labeled heatmap cells.],
) <fig:dashboard>

The training mean distance in @fig:heatmap is $4.07$ for RL, $2.79$ for the clone, and $5.43$ for the heuristic.
That ordering does not contradict the success rates.
Success requires _every_ sheep inside a ball of radius $2$.
A policy that finishes 75% of episodes near the goal and fails the rest with a large residual (a mixture of order $0.75 times 1.5 + 0.25 times 12 approx 4$) can post a worse average distance than a policy that rarely finishes but keeps the flock nearby.
The clone exhibits exactly that pattern: it is "almost there" more often than it succeeds.

== Out-of-distribution reversal

#figure(
  image("figures/scenario_heatmaps.png", width: 100%),
  caption: [Scenario-by-scenario success rate (left) and terminal mean distance to the goal (right).
  Darker blue is higher success; darker red is closer to the goal.
  Recurrent PPO is strongest on `train` and weakest on both held-out layouts, including a $10.68$ open-field residual that indicates a flock left far from the goal.],
) <fig:heatmap>

@tbl:success makes the generalization gap explicit.
On `unseen_open_field` the heuristic succeeds in every evaluated episode (mean distance $1.14$), the clone in half of them (distance $2.02$), and RL in none (distance $10.68$).
On `unseen_split_field` the heuristic retains 25% success, matching its training number, while both learned agents drop to 0%.
Averaging the two held-out scenarios recovers the "train versus unseen reliability" panel of @fig:dashboard: heuristic $0.25 slash 0.62$, clone $0.25 slash 0.25$, RL $0.75 slash 0.00$.

#figure(
  caption: [Success rate and terminal mean distance to the goal by method and scenario, read from the labeled cells of @fig:heatmap.],
  table(
    columns: (1.35fr, 0.7fr, 0.7fr, 0.7fr, 0.85fr, 0.85fr, 0.85fr),
    align: (left, center, center, center, center, center, center),
    table.header(
      [Method],
      [Train SR],
      [Split SR],
      [Open SR],
      [Train dist.],
      [Split dist.],
      [Open dist.],
    ),
    [Heuristic], [0.25], [0.25], [1.00], [5.43], [3.84], [1.14],
    [Behavioral cloning], [0.25], [0.00], [0.50], [2.79], [3.08], [2.02],
    [Recurrent PPO], [0.75], [0.00], [0.00], [4.07], [6.76], [10.68],
  ),
) <tbl:success>

Two qualitative observations follow.
First, the heuristic's collect--drive geometry is almost independent of clutter: when obstacles disappear, the original Strömbom strategy is recovered and is sufficient.
Second, the structured RL policy appears to overfit the training rectangles and the opposite-goal spawn.
In open space there is no clutter to "lean on," and the LSTM has no layout cue that matches its training distribution; the flock is then left far from the goal rather than being driven in a wide arc.
The split-field layout forces a cluster choice that the heuristic encodes explicitly and that the policy never had to make on the four training rectangles.

== Sample efficiency

#figure(
  image("figures/success_rate_by_scenario.png", width: 95%),
  caption: [Success rates for the same heuristic and cloning agents against a _short-budget_ recurrent PPO ($1.2 times 10^5$ steps, LSTM size $128$).
  Under-trained RL records 0% success on every scenario, while the heuristic still solves the open field.
  This figure is an ablation of sample efficiency, not a restatement of @fig:heatmap.],
) <fig:fast>

@fig:fast shows the $1.2 times 10^5$-step recurrent agent.
It contributes no successes.
Geometric shaping and recurrence are therefore not a substitute for training budget: the 75% in-distribution result of @fig:dashboard appears only after the structured $5 times 10^5$-step regime (and the associated curriculum and LSTM size).
Even that longer budget does not buy transfer.

== Offline cloning and feature importance

#figure(
  image("figures/bc_offline_metrics.png", width: 95%),
  caption: [Offline validation of the random-forest clone.
  Error metrics (left) are dominated by a $29°$ mean heading error; squared and absolute errors are small because actions are unit-length.
  Fit metrics (right) give $R^2 approx 0.59$ on $Delta x$ and $R^2 approx 0.69$ on $Delta y$.],
) <fig:bc>

@fig:bc reports validation MSE $0.178$, RMSE $0.422$, MAE $0.323$, and a mean angle error of $29.0°$, with $R^2$ of $0.591$ ($Delta x$) and $0.687$ ($Delta y$).
Those numbers describe a competent action imitator, not a competent herder.
Compounded over hundreds of steps, a $29°$ heading bias, combined with the absence of closed-loop recovery, is enough to miss the all-sheep success ball, which is exactly the "close but not finished" pattern in @tbl:success.

#figure(
  caption: [Top geometric features by random-forest impurity importance on the larger demonstration corpus.
  Cluster and centroid statistics dominate; individual sheep coordinates and obstacle box parameters rank far lower.],
  table(
    columns: (0.45fr, 2.2fr, 0.7fr),
    align: (center, left, right),
    table.header([Rank], [Feature], [Importance]),
    [1], [`focus_cluster_fraction`], [0.147],
    [2], [`centroid_dy`], [0.099],
    [3], [`visible_std_x`], [0.073],
    [4], [`visible_std_y`], [0.069],
    [5], [`centroid_dx`], [0.064],
    [6], [`goal_dy`], [0.047],
    [7], [`dog_x`], [0.042],
    [8], [`dog_y`], [0.041],
    [9], [`goal_dx`], [0.039],
    [10], [`max_sheep_goal_dist`], [0.035],
  ),
) <tbl:importance>

@tbl:importance is consistent with the expert's internal structure.
The single most important feature is the fraction of visible sheep assigned to the focus cluster---precisely the quantity the heuristic uses to decide whether the flock has split.
Centroid offsets and visible spread follow; raw obstacle coordinates are near zero (several of the eight slots are unused in a typical layout and receive exactly zero importance).
The forest is therefore not discovering a new representation so much as recovering the geometric program that generated the labels.
That is useful as analysis and a warning as a method: cloning cannot outperform an expert on layouts the expert itself cannot solve, and it will not invent a driving strategy for geometries absent from the demonstrations.

== Qualitative rollouts

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("figures/v3_structured_still.png", width: 100%),
    image("figures/bc_structured_still.png", width: 100%),
  ),
  caption: [Mid-episode 3-D renders on a structured layout.
  Left: recurrent PPO (step 85 of 254) has already taken a relatively direct path around the rectangles and holds the flock in view near the goal.
  Right: the cloning policy (step 100 of 299) traces a longer, looping trajectory before approaching the same cluster.
  Pink traces are dog paths; blue points are sheep; the yellow star is the goal.],
) <fig:qual>

@fig:qual contrasts closed-loop traces.
The RL dog's path is shorter and stays on the goal side of the obstacles, matching the drive-position term.
The cloned dog wanders, including a wide loop past the last rectangle---plausible as a collect maneuver, expensive as a drive.
Neither figure should be read as a success; they illustrate typical _in-distribution_ geometry, which is the only regime in which RL is the more decisive of the two.

= Discussion

The experimental picture is not "RL versus heuristics" in the abstract; it is a statement about _which inductive bias transfers_.
The heuristic encodes a geometry---collect the outlier, stand on the goal axis, switch---that is correct in open space and merely inconvenienced by a handful of rectangles.
The recurrent policy encodes whatever regularities the structured curriculum made cheap to exploit: opposite-corner spawns, four rectangle layouts, a visibility radius that rarely hides the whole flock.
Those regularities yield a 3$times$ in-distribution success gain and a complete out-of-distribution collapse.
Behavioral cloning interpolates because it copies the heuristic's local steering without copying its global mode structure perfectly, and because a 29° heading error accumulates.

Several design choices likely contribute to the gap.
The observation represents obstacles as up to eight independent boxes; nothing in the feature vector is a topological descriptor (passage width, homology of free space, number of clusters).
The LSTM can in principle track unseen sheep, but nothing forces it to track a _split that has not yet occurred_ on the training layouts.
The adaptive curriculum advances on training success, visibility, and collisions---all in-distribution statistics---so it never rewards open-field competence.
Domain randomization is deliberately turned _off_ for goal, visibility, and dynamics in the structured preset that produced the 75% result; that preset is the right tool for making RL solvable, and the wrong tool for making it general.

These observations suggest a concrete next step rather than a generic call for more data: train with a mixture of the structured layouts _and_ the held-out families, or with a topological curriculum that introduces splits and open space as explicit stages, and add cluster-count and passage-width features to both the policy input and the cloning vector.
A hybrid controller---heuristic collect--drive with an RL residual, or RL gated by the same focus-cluster test the forest already ranks first---is also a natural compromise given @tbl:importance.

Limitations of the present evidence should be stated clearly.
The reported success rates come from a small evaluation budget and, for the structured RL agent, a single training seed.
The environment is a 2-D kinematic simulation with rectangular obstacles, not a physical robot and not a 3-D animal-behavior study.
Only one shepherd is controlled; multi-herder cooperation @hasan2022 @napolitano2024 is out of scope.
GIF rollouts in the accompanying repository are qualitative and were reduced to still frames for print.

= Conclusion

We formalized single-dog shepherding under limited visibility as a geometric POMDP, shaped a dense reward from centroid, hull, incursion, drive-position, and visibility signals, and trained a recurrent PPO policy against a cluster-aware collect-and-drive expert and a random-forest clone of that expert.
Structured recurrent PPO is the best agent on the training distribution and the worst on the held-out split-field and open-field tests; the classical heuristic shows the opposite pattern; cloning is a middling action matcher that only partially closes the loop.
Geometric feature importances recover the expert's cluster logic and show that obstacle box coordinates contribute little.
The practical implication is that geometric reward shaping is effective _for learning to herd here_, and that transferring _there_ still depends on putting the missing geometries---splits, open space, and the collect--drive switch itself---into the training distribution rather than hoping recurrence will invent them.

Author names, affiliations, and a target IEEE venue are left as placeholders and will be filled in a later revision.