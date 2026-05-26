"""Hydra CLI entrypoint: build -> fit -> diagnose -> write parquets -> STATUS."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cdsbi.reproducibility.env import capture_env
from cdsbi.reproducibility.run_dir import RunDir, RunStatus
from cdsbi.reproducibility.seeding import seed_everything

log = logging.getLogger(__name__)


def _instantiate(target_path: str, **kwargs) -> Any:
    """Light-weight Hydra-style instantiation by import path."""
    module_path, cls_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(**kwargs)


def _build_flow(cfg: DictConfig) -> Any:
    flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
    target = flow_dict.pop("_target_")
    flow_dict.pop("name", None)
    return _instantiate(target, **flow_dict)


def _build_simulator(cfg: DictConfig) -> Any:
    sim_dict = OmegaConf.to_container(cfg.target, resolve=True)
    target = sim_dict.pop("_target_")
    sim_dict.pop("name", None)
    return _instantiate(target, **sim_dict)


def _build_method(cfg: DictConfig, simulator) -> Any:
    m = cfg.method
    runner_class = m.runner_class
    if m.name == "cd_sbi":
        from cdsbi.conditioners.identity import Identity
        from cdsbi.losses.nfmle import NFMLELoss
        flow = _build_flow(cfg)
        allow_ablation = bool(OmegaConf.select(cfg, "method.allow_ablation", default=False))
        return _instantiate(
            runner_class,
            flow=flow,
            conditioner=Identity(),
            loss=NFMLELoss(),
            allow_ablation=allow_ablation,
            device=cfg.device,
        )
    if m.name in ("npe", "nle"):
        flow = _build_flow(cfg)
        return _instantiate(runner_class, flow=flow, device=cfg.device)
    if m.name == "nre":
        return _instantiate(
            runner_class,
            classifier_hidden=m.classifier_hidden,
            classifier_depth=m.classifier_depth,
            device=cfg.device,
        )
    if m.name == "lf2i":
        flow = _build_flow(cfg)
        return _instantiate(
            runner_class,
            stat_flow=flow,
            quantile_hidden=m.quantile_hidden,
            quantile_depth=m.quantile_depth,
            theta_ref=m.theta_ref,
            device=cfg.device,
        )
    raise ValueError(f"Unknown method: {m.name}")


def _fit_config(cfg: DictConfig, method_name: str) -> dict:
    t = cfg.training
    if method_name == "cd_sbi":
        return {
            "lr": float(t.lr),
            "batch_size": int(t.batch_size),
            "n_steps": int(t.n_steps),
            "n_train": int(t.n_train),
            "fresh_batch": bool(OmegaConf.select(t, "fresh_batch", default=True)),
        }
    if method_name in ("npe", "nle", "nre"):
        return {"n_train": int(t.n_train), "n_epochs": int(t.n_epochs)}
    if method_name == "lf2i":
        return {
            "n_train_stat": int(t.n_train),
            "n_train_quantile": int(t.n_train) // 2,
            "n_epochs_stat": int(t.n_epochs),
            "n_epochs_quantile": 100,
            "alpha_grid": list(cfg.experiment.alpha_grid),
        }
    raise ValueError(method_name)


def _run_diagnostics(cfg: DictConfig, trained, simulator, eval_data, rd: RunDir):
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
    from cdsbi.diagnostics.coverage import Coverage
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.diagnostics.pivot_rmse import PivotRMSE

    n_bins = max(2, len(list(cfg.experiment.eval_thetas_interior)))
    diagnostics = [
        ("pivot_rmse", PivotRMSE()),
        ("marginal_pit", MarginalPIT()),
        ("conditional_pit", ConditionalPIT(n_bins=n_bins)),
        ("coverage", Coverage(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            alpha_grid=list(cfg.experiment.alpha_grid),
            n_per_theta=int(cfg.experiment.n_eval_per_theta),
        )),
    ]
    diag_results = {}
    diag_dir = rd.path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    for name, diag in diagnostics:
        result = diag(trained, simulator, eval_data=eval_data)
        if isinstance(result.value, pd.DataFrame):
            df = result.value
        else:
            df = pd.DataFrame([{
                "value": result.value,
                "passed": result.passed,
                "noise_floor": result.noise_floor,
                "n_samples": result.n_samples,
            }])
        df.to_parquet(diag_dir / f"{name}.parquet")
        diag_results[name] = result
    return diag_results


def _write_index_row(cfg: DictConfig, rd: RunDir, trained, diag_results, config_hash, env, n_params,
                     budget_status: str, budget_rel_err: float):
    cov_df = pd.read_parquet(rd.path / "diagnostics" / "coverage.parquet")
    coverage_error_max = float((cov_df["empirical"] - cov_df["nominal"]).abs().max())
    marg = diag_results.get("marginal_pit")
    pivot = diag_results.get("pivot_rmse")
    row = {
        "config_hash": config_hash,
        "experiment": cfg.experiment.name,
        "method": cfg.method.name,
        "flow": cfg.flow.name,
        "target": cfg.target.name,
        "budget_name": cfg.budget.name,
        "target_params": int(cfg.budget.target_params),
        "actual_params_total": int(n_params["total"]),
        "actual_params_kind": n_params["kind"],
        "budget_status": budget_status,
        "budget_rel_err": float(budget_rel_err),
        "seed": int(cfg.seed),
        "device": env["device"],
        "git_sha": env["git_sha"],
        "dirty_tree": env["dirty_tree"],
        "final_loss": float(trained.final_loss),
        "wall_clock_sec": float(trained.wall_clock_sec),
        "coverage_error_max": coverage_error_max,
        "marginal_ks": float(marg.value) if marg and isinstance(marg.value, float) else None,
        "pivot_rmse": float(pivot.value) if pivot and isinstance(pivot.value, float) else None,
    }
    pd.DataFrame([row]).to_parquet(rd.path / "index_row.parquet")


def _config_hash(cfg: DictConfig) -> str:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    canonical = yaml.safe_dump(resolved, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir_path = Path(HydraConfig.get().runtime.output_dir)
    rd = RunDir(run_dir_path)
    rd.set_status(RunStatus.RUNNING)

    try:
        rd.write_atomic(run_dir_path / "config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
        env = capture_env()
        rd.write_atomic(run_dir_path / "env.json", json.dumps(env, indent=2))
        config_hash = _config_hash(cfg)
        rngs = seed_everything(int(cfg.seed))
        rd.write_atomic(run_dir_path / "seeds.json", json.dumps({
            "seed": int(cfg.seed), "config_hash": config_hash,
        }, indent=2))

        from cdsbi.methods.budget import validate_budget

        simulator = _build_simulator(cfg)
        runner = _build_method(cfg, simulator)
        if cfg.method.name == "lf2i":
            n_params = runner.n_params(alpha_grid_len=len(list(cfg.experiment.alpha_grid)))
        elif cfg.method.name == "nre":
            n_params = runner.n_params(d_theta=simulator.d_theta, d_x=simulator.d_x)
        else:
            n_params = runner.n_params()

        status, rel_err = validate_budget(n_params["total"], int(cfg.budget.target_params))
        budget_msg = (
            f"Budget '{cfg.budget.name}' (target {cfg.budget.target_params}): "
            f"actual {n_params['total']} (rel_err {rel_err:.1%}, status {status})"
        )
        if status == "unreachable":
            log.warning(
                f"BUDGET MISMATCH (>15%): {budget_msg} "
                f"— proceeding but paper-table 'matched-budget' claim weakens"
            )
        elif status == "matched_with_warning":
            log.warning(f"BUDGET DRIFT: {budget_msg}")
        else:
            log.info(budget_msg)

        fit_cfg = _fit_config(cfg, cfg.method.name)
        trained = runner.fit(simulator=simulator, config=fit_cfg, seed=int(cfg.seed))

        n_eval = int(OmegaConf.select(cfg, "experiment.n_eval", default=1000))
        theta_eval, x_eval = simulator.sample(n_eval, rngs.eval)

        diag_results = _run_diagnostics(cfg, trained, simulator, (theta_eval, x_eval), rd)
        _write_index_row(cfg, rd, trained, diag_results, config_hash, env, n_params,
                         budget_status=status, budget_rel_err=rel_err)

        torch.save(
            {"arch_metadata": trained.arch_metadata, "final_loss": trained.final_loss},
            run_dir_path / "model.pt",
        )

        rd.set_status(RunStatus.OK)
    except Exception:
        rd.write_atomic(run_dir_path / "stdout.log", traceback.format_exc())
        rd.set_status(RunStatus.FAILED)
        raise


if __name__ == "__main__":
    main()
