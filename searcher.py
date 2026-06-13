"""Hyperparameter optimization: random, grid, Optuna."""
import itertools, random, numpy as np
from typing import Dict, Any, Callable, List, Tuple, Optional


def _sample_random(grid: Dict[str, list], seed: int, n: int) -> List[Dict]:
    random.seed(seed); np.random.seed(seed)
    combos = list(itertools.product(*grid.values()))
    if len(combos) > n:
        combos = random.sample(combos, n)
    return [dict(zip(grid.keys(), c)) for c in combos]


def _all_combos(grid: Dict[str, list]) -> List[Dict]:
    return [dict(zip(grid.keys(), c)) for c in itertools.product(*grid.values())]


def search(
    model_class,
    grid: Dict[str, list],
    train_df,
    test_df,
    metric_fn: Callable,
    method: str = "random",
    n_trials: int = 20,
    seed: int = 42,
    progress_fn: Optional[Callable] = None,
) -> Tuple[Dict[str, Any], float, List[Dict]]:
    """
    Run hyperparameter search. Returns (best_params, best_score, all_results).
    """
    if method == "optuna":
        return _optuna_search(model_class, grid, train_df, test_df, metric_fn, n_trials, seed, progress_fn)

    if method == "grid":
        combos = _all_combos(grid)
    else:
        combos = _sample_random(grid, seed, n_trials)

    best_params, best_score = {}, float("inf")
    results = []
    for i, params in enumerate(combos):
        try:
            m = model_class(seed=seed, **params)
            m.fit(train_df)
            preds = m.predict(len(test_df))
            score = metric_fn(test_df["y"].values, preds["yhat"].values[:len(test_df)])
            results.append({"params": params, "score": score})
            if score < best_score:
                best_score = score; best_params = params.copy()
        except Exception as e:
            results.append({"params": params, "score": float("inf"), "error": str(e)})
        if progress_fn:
            progress_fn(i + 1, len(combos))

    return best_params, best_score, results


def _optuna_search(model_class, grid, train_df, test_df, metric_fn, n_trials, seed, progress_fn):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return search(model_class, grid, train_df, test_df, metric_fn, "random", n_trials, seed, progress_fn)

    results = []
    trial_count = [0]

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in grid.items()}
        try:
            m = model_class(seed=seed, **params)
            m.fit(train_df)
            preds = m.predict(len(test_df))
            s = metric_fn(test_df["y"].values, preds["yhat"].values[:len(test_df)])
        except Exception:
            s = float("inf")
        results.append({"params": params, "score": s})
        trial_count[0] += 1
        if progress_fn:
            progress_fn(trial_count[0], n_trials)
        return s

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    best = study.best_params if study.best_trial else (results[0]["params"] if results else {})
    best_score = study.best_value if study.best_trial else float("inf")
    return best, best_score, results
