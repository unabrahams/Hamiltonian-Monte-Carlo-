import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import models
import samplers


def make_rng(seed):
    return np.random.default_rng(seed)


def run_single_experiment(
    model_name,
    algorithm_name,
    sampler_fn,
    sampler_kwargs,
    seed
):
    rng = make_rng(seed)

    samples, info = sampler_fn(rng=rng, **sampler_kwargs)

    return {
        "model": model_name,
        "algorithm": algorithm_name,
        "seed": seed,
        "samples": samples,
        "info": info,
    }


def run_many_experiments(experiment_specs):
    results = []

    for spec in experiment_specs:
        out = run_single_experiment(
            model_name=spec["model"],
            algorithm_name=spec["algorithm"],
            sampler_fn=spec["sampler_fn"],
            sampler_kwargs=spec["sampler_kwargs"],
            seed=spec["seed"],
        )
        results.append(out)

    return results

# Global configuration


N_SAMPLES = 10_000
N_ADAPT = 2_000
SEEDS = [11]
DELTA_GRID = [ 0.65]

PLOT_DIR = "mixture_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# Helpers for evaluations


def mode_indicator(samples, mean1, mean2):
    samples = np.asarray(samples)
    d1 = np.sum((samples - mean1) ** 2, axis=1)
    d2 = np.sum((samples - mean2) ** 2, axis=1)
    return (d2 < d1).astype(int)

def mixture_metrics(samples, mean1, mean2):
    modes = mode_indicator(samples, mean1, mean2)
    freq_mode0 = np.mean(modes == 0)
    freq_mode1 = np.mean(modes == 1)
    n_switches = int(np.sum(modes[1:] != modes[:-1]))
    return {
        "mode_indicator": modes,
        "freq_mode0": float(freq_mode0),
        "freq_mode1": float(freq_mode1),
        "n_switches": n_switches,
    }

def save_mixture_plots(samples, mean1, mean2, model_name, algorithm_name, seed, delta=None):
    modes = mode_indicator(samples, mean1, mean2)

    if delta is None:
        delta_tag = "no_delta"
        title_suffix = f"{algorithm_name} | seed={seed}"
    else:
        delta_tag = f"delta_{str(delta).replace('.', 'p')}"
        title_suffix = f"{algorithm_name} | seed={seed} | delta={delta}"

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    h = axes[0].hist2d(samples[:, 0], samples[:, 1], bins=60)
    plt.colorbar(h[3], ax=axes[0], label="conteos")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"{title_suffix}: hist2d")

    axes[1].plot(modes, lw=0.8)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mode")
    axes[1].set_title(f"{title_suffix}: mode indicator (0=mode1, 1=mode2)")

    plt.tight_layout()

    fname = f"{model_name}__{algorithm_name}__seed_{seed}__{delta_tag}.png"
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fpath

# Helpers for ESS computation

def autocovariance(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_centered = x - np.mean(x)

    if max_lag is None:
        max_lag = min(n - 1, n // 2)

    gamma = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        gamma[k] = np.dot(x_centered[:n-k], x_centered[k:]) / n
    return gamma

def ess_simple(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)

    gamma = autocovariance(x, max_lag=max_lag)
    rho = gamma / gamma[0]

    # truncate when the autocorrelation becomes non-positive
    positive_rhos = []
    for r in rho[1:]:
        if r <= 0:
            break
        positive_rhos.append(r)

    tau_hat = 1 + 2 * np.sum(positive_rhos)
    ess_hat = n / tau_hat
    return ess_hat, rho

# Models


gamma_model = {
    "name": "gamma_5_1",
    "dim": 1,
    "logpdf": models.logpdf_gamma_5_1,
    "U": models.U_gamma_5_1,
    "grad_U": models.grad_U_gamma_5_1,
    "x0": models.gamma_initial_point(100.0),
    "Sigma": np.eye(1),
}

cauchy_model = {
    "name": "cauchy",
    "dim": 1,
    "logpdf": models.logpdf_cauchy,
    "U": models.U_cauchy,
    "grad_U": models.grad_U_cauchy,
    "x0": models.cauchy_initial_point(3.0),
    "Sigma": np.eye(1),
}

mvn_obj = models.make_neal_mvn_model(d=100)
neal_model = {
    "name": "neal_mvn_d100",
    "dim": 100,
    "logpdf": mvn_obj["logpdf"],
    "U": mvn_obj["U"],
    "grad_U": mvn_obj["grad_U"],
    "x0": models.neal_mvn_initial_point(d=100, value=7.0),
    "Sigma": np.eye(100),
}

mix_obj = models.make_bivariate_gaussian_mixture()
mixture_model = {
    "name": "bivariate_gaussian_mixture",
    "dim": 2,
    "logpdf": mix_obj["logpdf"],
    "U": mix_obj["U"],
    "grad_U": mix_obj["grad_U"],
    "x0": models.gaussian_mixture_initial_point((-1.5, -1.5)),
    "Sigma": np.eye(2),
    "mean1": mix_obj["mean1"],
    "mean2": mix_obj["mean2"],
}

schools_obj = models.make_eight_schools_model()
schools_model = {
    "name": "eight_schools",
    "dim": 10,
    "logpdf": schools_obj["logpdf"],
    "U": schools_obj["U"],
    "grad_U": schools_obj["grad_U"],
    "x0": models.eight_schools_initial_point(mu0=2.2, tau0=2.0, eta0=2.0),
    "Sigma": np.eye(10),
}

MODEL_LIST = [
    gamma_model,
    cauchy_model,
    neal_model,
    mixture_model,
    schools_model,
]


# Hyperparameters


BASE_CFG = {
    "gamma_5_1": {
        "rwmh": {"proposal_std": 1.0},
        "hmc": {"epsilon": 0.1, "L": 100},
        "nuts": {"epsilon": 0.1, "Delta_max": 1000.0},
        "rahmc": {"epsilon": 0.08, "gamma": 0.4, "L": 20},
        "uhmc": {"T": 1.0, "h": 0.02},
        "hmc_da": {"epsilon0": 0.03, "lam": 6.0},
        "nuts_da": {"epsilon0": 0.03, "Delta_max": 1000.0},
        "rahmc_da": {"epsilon0": 0.1, "gamma0": 0.4, "T": 10.2},
    },
    "cauchy": {
        "rwmh": {"proposal_std": 1.0},
        "hmc": {"epsilon": 0.15, "L": 25},
        "nuts": {"epsilon": 0.15, "Delta_max": 1000.0},
        "rahmc": {"epsilon": 0.12, "gamma": 0.5, "L": 20},
        "uhmc": {"T": 1.0, "h": 0.03},
        "hmc_da": {"epsilon0": 0.15, "lam": 4.0},
        "nuts_da": {"epsilon0": 0.12, "Delta_max": 1000.0},
        "rahmc_da": {"epsilon0": 0.10, "gamma0": 0.5, "T": 4.0},
    },
    "neal_mvn_d100": {
        "rwmh": {"proposal_std": 0.02},
        "hmc": {"epsilon": 0.003, "L": 300},
        "nuts": {"epsilon": 0.013, "Delta_max": 1000.0},
        "rahmc": {"epsilon": 0.003, "gamma": 0.2, "L": 300},
        "uhmc": {"T": 1.5, "h": 0.003},
        "hmc_da": {"epsilon0": 0.013, "lam": 2.0},
        "nuts_da": {"epsilon0": 0.013, "Delta_max": 1000.0},
        "rahmc_da": {"epsilon0": 0.009, "gamma0": 0.3, "T": 2.0},
    },
    "bivariate_gaussian_mixture": {
        "rwmh": {"proposal_std": 1.0},
        "hmc": {"epsilon": 0.20, "L": 25},
        "nuts": {"epsilon": 0.20, "Delta_max": 1000.0},
        "rahmc": {"epsilon": 0.20, "gamma": 0.35, "L": 24},
        "uhmc": {"T": 1.0, "h": 0.02},
        "hmc_da": {"epsilon0": 0.20, "lam": 5.0},
        "nuts_da": {"epsilon0": 0.20, "Delta_max": 1000.0},
        "rahmc_da": {"epsilon0": 0.05, "gamma0": 0.10, "T": 5.0},
    },
    "eight_schools": {
        "rwmh": {"proposal_std": 1.0},
        "hmc": {"epsilon": 0.05, "L": 60},
        "nuts": {"epsilon": 0.05, "Delta_max": 1000.0},
        "rahmc": {"epsilon": 0.05, "gamma": 0.35, "L": 60},
        "uhmc": {"T": 0.8, "h": 0.01},
        "hmc_da": {"epsilon0": 0.05, "lam": 3.0},
        "nuts_da": {"epsilon0": 0.03, "Delta_max": 1000.0},
        "rahmc_da": {"epsilon0": 0.025, "gamma0": 0.25, "T": 3.0},
    },
}


# Results


ALL_RESULTS = []

global_start = time.time()

for model in MODEL_LIST:
    model_name = model["name"]
    cfg = BASE_CFG[model_name]

    print(f"\n==================== MODELO: {model_name} ====================")

    for seed in SEEDS:
        print(f"  seed = {seed}")

        # RWMH
        res = run_single_experiment(
            model_name=model_name,
            algorithm_name="RWMH",
            sampler_fn=samplers.run_rwmh,
            sampler_kwargs={
                "logpdf": model["logpdf"],
                "x0": model["x0"],
                "n_samples": N_SAMPLES,
                **cfg["rwmh"],
            },
            seed=seed,
        )
        if model_name == "bivariate_gaussian_mixture":
            mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
            plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "RWMH", seed)
            res["mixture_metrics"] = {
                "freq_mode0": mm["freq_mode0"],
                "freq_mode1": mm["freq_mode1"],
                "n_switches": mm["n_switches"],
                "plot_path": plot_path,
            }
        ALL_RESULTS.append(res)

        # HMC
        res = run_single_experiment(
            model_name=model_name,
            algorithm_name="HMC",
            sampler_fn=samplers.run_hmc,
            sampler_kwargs={
                "U": model["U"],
                "grad_U": model["grad_U"],
                "x0": model["x0"],
                "n_samples": N_SAMPLES,
                **cfg["hmc"],
            },
            seed=seed,
        )
        if model_name == "bivariate_gaussian_mixture":
            mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
            plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "HMC", seed)
            res["mixture_metrics"] = {
                "freq_mode0": mm["freq_mode0"],
                "freq_mode1": mm["freq_mode1"],
                "n_switches": mm["n_switches"],
                "plot_path": plot_path,
            }
        ALL_RESULTS.append(res)

        # NUTS
        res = run_single_experiment(
            model_name=model_name,
            algorithm_name="NUTS",
            sampler_fn=samplers.run_nuts,
            sampler_kwargs={
                "L": model["logpdf"],
                "grad_L": lambda q, gu=model["grad_U"]: -gu(q),
                "x0": model["x0"],
                "M": N_SAMPLES,
                **cfg["nuts"],
            },
            seed=seed,
        )
        if model_name == "bivariate_gaussian_mixture":
            mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
            plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "NUTS", seed)
            res["mixture_metrics"] = {
                "freq_mode0": mm["freq_mode0"],
                "freq_mode1": mm["freq_mode1"],
                "n_switches": mm["n_switches"],
                "plot_path": plot_path,
            }
        ALL_RESULTS.append(res)

        # rAHMC
        res = run_single_experiment(
            model_name=model_name,
            algorithm_name="rAHMC",
            sampler_fn=samplers.run_rahmc,
            sampler_kwargs={
                "U": model["U"],
                "grad_U": model["grad_U"],
                "Sigma": model["Sigma"],
                "x0": model["x0"],
                "n_samples": N_SAMPLES,
                **cfg["rahmc"],
            },
            seed=seed,
        )
        if model_name == "bivariate_gaussian_mixture":
            mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
            plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "rAHMC", seed)
            res["mixture_metrics"] = {
                "freq_mode0": mm["freq_mode0"],
                "freq_mode1": mm["freq_mode1"],
                "n_switches": mm["n_switches"],
                "plot_path": plot_path,
            }
        ALL_RESULTS.append(res)

        # uHMC
        res = run_single_experiment(
            model_name=model_name,
            algorithm_name="uHMC",
            sampler_fn=samplers.run_uhmc,
            sampler_kwargs={
                "grad_U": model["grad_U"],
                "x0": model["x0"],
                "n_samples": N_SAMPLES,
                **cfg["uhmc"],
            },
            seed=seed,
        )
        if model_name == "bivariate_gaussian_mixture":
            mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
            plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "uHMC", seed)
            res["mixture_metrics"] = {
                "freq_mode0": mm["freq_mode0"],
                "freq_mode1": mm["freq_mode1"],
                "n_switches": mm["n_switches"],
                "plot_path": plot_path,
            }
        ALL_RESULTS.append(res)

        for delta in DELTA_GRID:
            print(f"    delta = {delta}")

            # HMC-DA
            res = run_single_experiment(
                model_name=model_name,
                algorithm_name="HMC-DA",
                sampler_fn=samplers.run_hmc_dual_averaging,
                sampler_kwargs={
                    "U": model["U"],
                    "grad_U": model["grad_U"],
                    "x0": model["x0"],
                    "n_samples": N_SAMPLES,
                    "n_adapt": N_ADAPT,
                    "delta": delta,
                    **cfg["hmc_da"],
                },
                seed=seed,
            )
            res["delta"] = delta
            if model_name == "bivariate_gaussian_mixture":
                mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
                plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "HMC-DA", seed, delta)
                res["mixture_metrics"] = {
                    "freq_mode0": mm["freq_mode0"],
                    "freq_mode1": mm["freq_mode1"],
                    "n_switches": mm["n_switches"],
                    "plot_path": plot_path,
                }
            ALL_RESULTS.append(res)

            # NUTS-DA
            res = run_single_experiment(
                model_name=model_name,
                algorithm_name="NUTS-DA",
                sampler_fn=samplers.run_nuts_dual_averaging,
                sampler_kwargs={
                    "L": model["logpdf"],
                    "grad_L": lambda q, gu=model["grad_U"]: -gu(q),
                    "x0": model["x0"],
                    "n_samples": N_SAMPLES,
                    "n_adapt": N_ADAPT,
                    "delta": delta,
                    **cfg["nuts_da"],
                },
                seed=seed,
            )
            res["delta"] = delta
            if model_name == "bivariate_gaussian_mixture":
                mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
                plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "NUTS-DA", seed, delta)
                res["mixture_metrics"] = {
                    "freq_mode0": mm["freq_mode0"],
                    "freq_mode1": mm["freq_mode1"],
                    "n_switches": mm["n_switches"],
                    "plot_path": plot_path,
                }
            ALL_RESULTS.append(res)

            # rAHMC-DA
            res = run_single_experiment(
                model_name=model_name,
                algorithm_name="rAHMC-DA",
                sampler_fn=samplers.run_rahmc_dual_averaging,
                sampler_kwargs={
                    "U": model["U"],
                    "grad_U": model["grad_U"],
                    "Sigma": model["Sigma"],
                    "x0": model["x0"],
                    "n_samples": N_SAMPLES,
                    "n_adapt": N_ADAPT,
                    "delta": delta,
                    **cfg["rahmc_da"],
                },
                seed=seed,
            )
            res["delta"] = delta
            if model_name == "bivariate_gaussian_mixture":
                mm = mixture_metrics(res["samples"], model["mean1"], model["mean2"])
                plot_path = save_mixture_plots(res["samples"], model["mean1"], model["mean2"], model_name, "rAHMC-DA", seed, delta)
                res["mixture_metrics"] = {
                    "freq_mode0": mm["freq_mode0"],
                    "freq_mode1": mm["freq_mode1"],
                    "n_switches": mm["n_switches"],
                    "plot_path": plot_path,
                }
            ALL_RESULTS.append(res)

global_elapsed = time.time() - global_start

print("\n============================================================")
print(f"TOTAL DE CORRIDAS: {len(ALL_RESULTS)}")
print(f"TIEMPO TOTAL (s): {global_elapsed:.2f}")
print(f"GRAFICAS DE LA MEZCLA GUARDADAS EN: {PLOT_DIR}")

rows = []

for r in ALL_RESULTS:

    info = r["info"]
    algo = r["algorithm"]
    model = r["model"]

    row = {
        "model": model,
        "algorithm": algo,
        "seed": r["seed"],
        "delta": r.get("delta", None),
        "n_samples": len(r["samples"])
    }

    # info numérica del sampler
    for k,v in info.items():
        if np.isscalar(v):
            row[k] = v

    # métricas de mezcla si existen
    if "mixture_metrics" in r:
        mm = r["mixture_metrics"]
        row["freq_mode0"] = mm["freq_mode0"]
        row["freq_mode1"] = mm["freq_mode1"]
        row["n_switches"] = mm["n_switches"]

    rows.append(row)

df = pd.DataFrame(rows)

ess_list = []
min_ess_list = []
sec_per_ess_list = []
sec_per_min_ess_list = []
n_used_draws_list = []

for r in ALL_RESULTS:
    samples = np.asarray(r["samples"])
    info = r["info"]

    burn = int(info.get("n_adapt", 0))
    used = samples[burn:]

    if used.ndim == 1:
        used = used.reshape(-1, 1)

    n_used_draws = used.shape[0]
    n_used_draws_list.append(n_used_draws)

    ess_dims = []
    for j in range(used.shape[1]):
        chain_j = used[:, j]   # shape (1, draws)  # MODIFIED LINE: Removed [None, :]
        ess_raw, rho_raw = ess_simple(chain_j)
        ess_j = float(np.asarray(ess_raw))
        ess_dims.append(ess_j)

    ess_dims = np.asarray(ess_dims, dtype=float)

    ess_mean = float(np.mean(ess_dims))
    ess_min = float(np.min(ess_dims))

    ess_list.append(ess_mean)
    min_ess_list.append(ess_min)

    runtime = float(info["time"])

    sec_per_ess = runtime / ess_mean if ess_mean > 0 else np.nan
    sec_per_min_ess = runtime / ess_min if ess_min > 0 else np.nan

    sec_per_ess_list.append(sec_per_ess)
    sec_per_min_ess_list.append(sec_per_min_ess)

df["n_used_draws"] = n_used_draws_list
df["ess"] = ess_list
df["min_ess"] = min_ess_list
df["sec_per_ess"] = sec_per_ess_list
df["sec_per_min_ess"] = sec_per_min_ess_list

# Save results
os.makedirs("results", exist_ok=True)

csv_path = os.path.join("results", "summary_results.csv")
df.to_csv(csv_path, index=False)

print(f"Results saved to {csv_path}")

# plots for Gamma example

def get_result(all_results, model_name, algorithm_name, seed, delta=None):
    for r in all_results:
        if r["model"] != model_name:
            continue
        if r["algorithm"] != algorithm_name:
            continue
        if r["seed"] != seed:
            continue
        r_delta = r.get("delta", None)
        if delta is None and r_delta is None:
            return r
        if delta is not None and r_delta == delta:
            return r
    raise ValueError("No encontré esa corrida.")

def gamma_pdf(x):
    # Gamma(shape=5, rate=1)
    return (x**4) * np.exp(-x) / 24.0

def plot_gamma_hist(all_results, algorithm_name="HMC", seed=11, delta=None, burn=2000, bins=60):
    r = get_result(all_results, "gamma_5_1", algorithm_name, seed, delta=delta)
    samples = np.asarray(r["samples"]).reshape(-1)
    used = samples[burn:]

    xs = np.linspace(0.001, max(20, np.percentile(used, 99.5)), 500)
    ys = gamma_pdf(xs)

    plt.figure(figsize=(8,5))
    plt.hist(used, bins=bins, density=True, alpha=0.6, label="samples")
    plt.plot(xs, ys, lw=2, label="Gamma(5,1) true density")
    plt.xlabel("x")
    plt.ylabel("density")
    title = f"Gamma(5,1) | {algorithm_name}"
    if delta is not None:
        title += f" | delta={delta}"
    plt.title(title)
    plt.legend()
    plt.show()

for seed in [11]:
    for algo in ["RWMH", "HMC", "NUTS", "rAHMC", "uHMC"]:
        plot_gamma_hist(ALL_RESULTS, algorithm_name=algo, seed=seed, burn=2000)