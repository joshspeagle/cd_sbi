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
import numpy as np
import pandas as pd
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.reproducibility.env import capture_env
from cdsbi.reproducibility.run_dir import RunDir, RunStatus
from cdsbi.reproducibility.seeding import seed_everything

log = logging.getLogger(__name__)


def _instantiate(target_path: str, **kwargs) -> Any:
    """Light-weight Hydra-style instantiation by import path."""
    module_path, cls_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(**kwargs)


def _build_flow(cfg: DictConfig, simulator) -> Any:
    """Instantiate the flow requested by cfg.method.flow, resolved against cfg.budget.

    cfg.flow is the Hydra group default (always additive_umnn from config.yaml's
    defaults list).  Each method config carries a `flow:` string label that names
    the *intended* flow group.  When method.flow != cfg.flow.name we instantiate
    directly from cfg.budget rather than relying on the Hydra group — this avoids
    the Hydra 1.3 limitation where a secondary config cannot override a parent's
    group default via a nested defaults list.
    """
    method_flow_label = OmegaConf.select(cfg, "method.flow", default=None)
    hydra_flow_name = cfg.flow.name

    # v3 flows (doubly_monotone, joint_umnn, joint_umnn_1d): when the experiment
    # overrides /flow to one of these, respect it regardless of the method's
    # default flow label. The flow YAMLs carry their own _target_ + hidden refs
    # so the fast-path instantiation is sufficient.
    v3_flow_names = {"doubly_monotone", "joint_umnn", "joint_umnn_1d"}
    if hydra_flow_name in v3_flow_names:
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        return _instantiate(target, **flow_dict)

    if method_flow_label is None or method_flow_label == hydra_flow_name:
        # Fast path: cfg.flow already holds the right config (cd_sbi case).
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        # Multivariate triangular flow needs d injected from the simulator.
        if "triangular_additive.TriangularAdditiveFlow" in target:
            flow_dict.setdefault("d", int(simulator.d_theta))
        return _instantiate(target, **flow_dict)

    # Slow path: method requests a different flow than the Hydra group default.
    # Build from first principles using cfg.budget to resolve hidden-size params.
    if method_flow_label == "maf":
        return _instantiate(
            "cdsbi.flows.maf_adapter.MAFAdapter",
            features=int(simulator.d_theta),
            context_features=int(simulator.d_x),
            hidden=int(cfg.budget.maf_hidden),
            num_layers=2,
        )
        # TODO(v1+): for asymmetric d (d_x != d_theta), NLE wants features=d_x
        # and context_features=d_theta. v1's loc_gauss_2d_iid is symmetric so
        # this is correct; revisit when an asymmetric target lands.
    if method_flow_label == "additive_umnn":
        if int(simulator.d_theta) == 1:
            return _instantiate(
                "cdsbi.flows.additive.AdditiveFlow1D",
                hidden=int(cfg.budget.cdsbi_flow_hidden),
            )
        # d > 1: use the d-specific budget key if present (e.g.
        # cdsbi_flow_hidden_d2 for d=2). Fall back to the 1D value as a
        # rough estimate when no d-specific value is defined.
        d = int(simulator.d_theta)
        d_key = f"cdsbi_flow_hidden_d{d}"
        budget_hidden = int(OmegaConf.select(
            cfg.budget, d_key, default=cfg.budget.cdsbi_flow_hidden,
        ))
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=d,
            hidden=budget_hidden,
        )
    if method_flow_label == "triangular_additive":
        d = int(simulator.d_theta)
        d_key = f"cdsbi_flow_hidden_d{d}"
        budget_hidden = int(OmegaConf.select(
            cfg.budget, d_key, default=cfg.budget.cdsbi_flow_hidden,
        ))
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=d,
            hidden=budget_hidden,
        )
    # v3 flows all share the same constructor signature (hidden, theta_ref, depth)
    # and the same budget key (doubly_monotone_hidden). depth defaults to the
    # codebase convention (2) unless the flow YAML overrides.
    v3_targets = {
        "doubly_monotone": "cdsbi.flows.doubly_monotone.DoublyMonotoneUMNN",
        "joint_umnn": "cdsbi.flows.joint_umnn.JointUMNNFlow",
        "joint_umnn_1d": "cdsbi.flows.joint_umnn_1d.JointUMNN1DFlow",
    }
    if method_flow_label in v3_targets:
        depth = int(OmegaConf.select(cfg, "flow.depth", default=2))
        # Default theta_ref: center of the proposal range (matches the §8.4
        # reference implementation). Keeping the integral path centered on
        # the mid-range reduces |theta - theta_ref| on average, which keeps
        # the Gauss-Legendre quadrature error bounded. The lower-endpoint
        # choice (which the manuscript prose suggests) was an earlier bug
        # caught by diffing against the reference code.
        a, b = simulator.theta_range
        default_theta_ref = 0.5 * (a + b)
        theta_ref = float(OmegaConf.select(cfg, "flow.theta_ref", default=default_theta_ref))
        return _instantiate(
            v3_targets[method_flow_label],
            hidden=int(cfg.budget.doubly_monotone_hidden),
            theta_ref=theta_ref,
            depth=depth,
        )
    raise ValueError(
        f"Unknown flow label '{method_flow_label}' in method.flow. "
        "Expected one of: 'maf', 'additive_umnn', 'triangular_additive', "
        "'doubly_monotone', 'joint_umnn', 'joint_umnn_1d'."
    )


def _build_simulator(cfg: DictConfig) -> Any:
    sim_dict = OmegaConf.to_container(cfg.target, resolve=True)
    target = sim_dict.pop("_target_")
    sim_dict.pop("name", None)
    return _instantiate(target, **sim_dict)


def _build_method(cfg: DictConfig, simulator) -> Any:
    m = cfg.method
    runner_class = m.runner_class
    if m.name == "cd_sbi":
        from cdsbi.losses.nfmle import NFMLELoss
        flow = _build_flow(cfg, simulator)
        allow_ablation = bool(OmegaConf.select(cfg, "method.allow_ablation", default=False))
        # Conditioner dispatch: exp_rate uses MLPConditioner(frozen_sum) to reduce
        # X ∈ R^{n_iid} to the sufficient statistic T = Σ X_i with the
        # accompanying log|∂T/∂X|; all other targets pass X through unchanged.
        if cfg.target.name == "exp_rate":
            from cdsbi.conditioners.mlp import MLPConditioner
            conditioner = MLPConditioner(
                input_dim=int(simulator.d_x),
                output_dim=1,
                mode="frozen_sum",
            )
        else:
            from cdsbi.conditioners.identity import Identity
            conditioner = Identity()
        return _instantiate(
            runner_class,
            flow=flow,
            conditioner=conditioner,
            loss=NFMLELoss(),
            allow_ablation=allow_ablation,
            device=cfg.device,
        )
    if m.name in ("npe", "nle"):
        flow = _build_flow(cfg, simulator)
        return _instantiate(runner_class, flow=flow, device=cfg.device)
    if m.name == "nre":
        return _instantiate(
            runner_class,
            classifier_hidden=m.classifier_hidden,
            classifier_depth=m.classifier_depth,
            device=cfg.device,
        )
    if m.name == "lf2i_bff":
        return _instantiate(
            runner_class,
            classifier_hidden=m.classifier_hidden,
            classifier_depth=m.classifier_depth,
            quantile_hidden=m.quantile_hidden,
            quantile_depth=m.quantile_depth,
            marginal_grid_n=int(OmegaConf.select(m, "marginal_grid_n", default=64)),
            device=cfg.device,
        )
    raise ValueError(f"Unknown method: {m.name}")


def _recipe_dict(t: DictConfig) -> dict:
    """Common training-recipe extraction shared by all methods that use
    train_with_recipe (cd_sbi, npe, nle, nre, lf2i, lf2i_bff)."""
    return {
        "lr": float(t.lr),
        "batch_size": int(t.batch_size),
        "n_steps": int(t.n_steps),
        "n_train": int(t.n_train),
        "fresh_batch": bool(OmegaConf.select(t, "fresh_batch", default=True)),
        "optimizer": str(OmegaConf.select(t, "optimizer", default="adam")),
        "weight_decay": float(OmegaConf.select(t, "weight_decay", default=0.0)),
        "betas": list(OmegaConf.select(t, "betas", default=[0.9, 0.999])),
        "momentum": float(OmegaConf.select(t, "momentum", default=0.9)),
        "lr_schedule": str(OmegaConf.select(t, "lr_schedule", default="constant")),
        "warmup_steps": int(OmegaConf.select(t, "warmup_steps", default=0)),
        "lr_min_ratio": float(OmegaConf.select(t, "lr_min_ratio", default=0.0)),
        "lr_gamma": float(OmegaConf.select(t, "lr_gamma", default=0.999)),
        "batching": str(OmegaConf.select(t, "batching", default="random_replacement")),
        "grad_clip_norm": float(OmegaConf.select(t, "grad_clip_norm", default=5.0)),
    }


def _fit_config(cfg: DictConfig, method_name: str) -> dict:
    t = cfg.training
    if method_name == "cd_sbi":
        return _recipe_dict(t)
    if method_name in ("npe", "nle", "nre"):
        return _recipe_dict(t)
    if method_name == "lf2i_bff":
        return {
            **_recipe_dict(t),
            "n_train_stat": int(t.n_train),
            "n_train_quantile": int(t.n_train) // 2,
            "alpha_grid": list(cfg.experiment.alpha_grid),
        }
    raise ValueError(method_name)


def _run_diagnostics(cfg: DictConfig, trained, simulator, eval_data, rd: RunDir):
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
    from cdsbi.diagnostics.coverage import Coverage
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.diagnostics.pivot_rmse import PivotRMSE
    from cdsbi.diagnostics.set_size import SetSize

    n_bins = max(2, len(list(cfg.experiment.eval_thetas_interior)))
    # SetSize is intentionally cheaper (~1/5 the X_obs of Coverage) — width
    # distribution converges much faster than coverage rate. Override via
    # cfg.experiment.set_size_n_per_theta if needed.
    set_size_n = int(OmegaConf.select(
        cfg, "experiment.set_size_n_per_theta",
        default=max(100, int(cfg.experiment.n_eval_per_theta) // 5),
    ))
    diagnostics = [
        ("pivot_rmse", PivotRMSE()),
        ("marginal_pit", MarginalPIT()),
        ("conditional_pit", ConditionalPIT(n_bins=n_bins)),
        ("coverage", Coverage(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            alpha_grid=list(cfg.experiment.alpha_grid),
            n_per_theta=int(cfg.experiment.n_eval_per_theta),
        )),
        ("set_size", SetSize(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            alpha_grid=list(cfg.experiment.alpha_grid),
            n_per_theta=set_size_n,
        )),
        ("joint_mahalanobis", JointMahalanobis(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            n_per_theta=int(OmegaConf.select(
                cfg, "experiment.joint_mahalanobis_n_per_theta", default=2000,
            )),
        )),
        ("jacobian_recovery", JacobianRecovery(
            n_points=int(OmegaConf.select(
                cfg, "experiment.jacobian_recovery_n_points", default=200,
            )),
            max_residual_tol=float(OmegaConf.select(
                cfg, "experiment.jacobian_recovery_tol", default=0.05,
            )),
        )),
    ]
    # F6: precompute r = procedure.pivot(theta, x) once for pivot-based
    # procedures and share it across PivotRMSE / MarginalPIT / ConditionalPIT
    # via a 3-tuple eval_data. Saves two forward passes per run.
    theta_eval, x_eval = eval_data
    if isinstance(trained.procedure, PivotBasedProcedure):
        with torch.no_grad():
            r_precomputed = trained.procedure.pivot(theta_eval, x_eval).detach()
        eval_data_shared = (theta_eval, x_eval, r_precomputed)
    else:
        eval_data_shared = eval_data
    # F7: pre-draw X|θ_0 once per θ_0 at the max n across the three consumers
    # (Coverage / SetSize / JointMahalanobis). Each diagnostic slices the
    # shared tensor down to its own n_per_theta. This also gives cross-
    # diagnostic comparability (same X seen by all three at a given θ_0).
    jm_n = int(OmegaConf.select(
        cfg, "experiment.joint_mahalanobis_n_per_theta", default=2000,
    ))
    n_max = max(int(cfg.experiment.n_eval_per_theta), set_size_n, jm_n)
    shared_rng = np.random.default_rng(0)
    shared_x_per_theta = {}
    for theta_0 in list(cfg.experiment.eval_thetas_interior):
        theta_repr = str(list(map(
            float,
            list(theta_0) if hasattr(theta_0, "__iter__") else [theta_0],
        )))
        shared_x_per_theta[theta_repr] = simulator.sample_x_given_theta(
            theta_0, n_max, shared_rng,
        )

    diag_results = {}
    diag_dir = rd.path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    x_sharing_names = {"coverage", "set_size", "joint_mahalanobis"}
    for name, diag in diagnostics:
        if name in x_sharing_names:
            result = diag(
                trained, simulator, eval_data=eval_data_shared,
                x_per_theta=shared_x_per_theta,
            )
        else:
            result = diag(trained, simulator, eval_data=eval_data_shared)
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
        # Report the actual instantiated flow group (cfg.flow.name) — this
        # reflects experiment-level /flow overrides (e.g. 8_3's triangular_additive,
        # 8_4's doubly_monotone) rather than the method's default flow label.
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
        "marginal_ks": (
            float(marg.value) if marg and isinstance(marg.value, float)
            else float(marg.value["ks"].mean()) if marg and hasattr(marg.value, "columns") and "ks" in marg.value.columns
            else None
        ),
        "pivot_rmse": float(pivot.value) if pivot and isinstance(pivot.value, float) else None,
    }
    jm_path = rd.path / "diagnostics" / "joint_mahalanobis.parquet"
    if jm_path.exists():
        jm_df = pd.read_parquet(jm_path)
        if "ks" in jm_df.columns and len(jm_df):
            row["joint_mahal_ks"] = float(jm_df["ks"].mean())
    jac_path = rd.path / "diagnostics" / "jacobian_recovery.parquet"
    if jac_path.exists():
        jac_df = pd.read_parquet(jac_path)
        if "max_residual" in jac_df.columns and len(jac_df):
            row["jacobian_max_residual"] = float(jac_df["max_residual"].iloc[0])
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
        if cfg.method.name == "lf2i_bff":
            n_params = runner.n_params(
                d_theta=simulator.d_theta,
                d_x=simulator.d_x,
                alpha_grid_len=len(list(cfg.experiment.alpha_grid)),
            )
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
