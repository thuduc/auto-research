# Autonomous Research Program

You are operating one governed autoresearch experiment for one champion-eligible model class.

Rules:

- Only modify `train.py`.
- Never modify `prepare.py` or `config/variables.yaml`.
- Work inside the assigned experiment folder under `experiments/<experiment_id>/`.
- Use the experiment-local Python 3.12 virtual environment in `experiments/<experiment_id>/venv/`.
- One trial means one concrete specification.
- Search autonomously over variable subsets, lag structure, hyperparameters, and architecture choices appropriate to the assigned model family.
- Keep all trial artifacts.
- Prefer simpler, more governable specifications when GOF is similar.
- Do not use the holdout during search.
- If 5 consecutive trials fail, stop the experiment and move on.
- Stop early when the experiment has plateaued and meaningful GOF improvement is no longer occurring.
- Treat econometric family coverage as incomplete unless both single-equation and multivariate cointegration approaches are represented where applicable.
- Distinguish clearly between exogenous-regressor selection, endogenous-system selection, and ML feature-set construction when designing trials.

Current reference outcome:

- The current best governed champion candidate is the finalized `Gradient Boosting` / `XGBoost` ensemble at `0.6422114864199716`.
- The current fixed production blend is `0.60 * Gradient Boosting + 0.40 * XGBoost`.
- The strongest single-model benchmark remains `Gradient Boosting` with `consumer_confidence`.
- The strongest improved deep-learning benchmark is refined `N-BEATS` with `housing_inventory`.

Search priorities:

1. Start from a simple baseline.
2. Spend most trials on variable selection or system-definition selection.
3. Use hyperparameter refinement only after finding plausible variable sets.
4. Penalize specifications that violate expected economic sign behavior when that concept applies.
5. Produce the best experiment-level GOF without human intervention.
6. For econometric classes, prefer standard benchmark families such as `ARDL`, `UECM / ECM`, `VECM`, `ETS`, and regime-switching alternatives unless they are explicitly out of scope for the current experiment.
