"""Generate the release summary tables and publication-quality figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gkp_consolidated_mpl")

import matplotlib.pyplot as plt
import numpy as np


TYPE_ORDER = ("(1,3)", "(1,5)", "(1,1,2)", "(1,1,3)", "(1,2,2)")
TYPE_COLORS = {
    "(1,3)": "#d62728",
    "(1,5)": "#9467bd",
    "(1,1,2)": "#1f77b4",
    "(1,1,3)": "#ff7f0e",
    "(1,2,2)": "#2ca02c",
}


def _load(name: str):
    return json.loads((PROJECT / "data" / name).read_text())


def _label(values) -> str:
    return "(" + ",".join(str(int(value)) for value in values) + ")"


def _save(fig, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(PROJECT / "figures" / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_axis(axis) -> None:
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    phase5 = _load("phase5_cm_population_summary.json")
    phase6 = _load("phase6_generic_controls_summary.json")
    phase7 = _load("phase7_equal_distance_controls_summary.json")
    phase8 = _load("phase8_adversarial_search_summary.json")
    phase9 = _load("phase9_gate_robustness_summary.json")
    phase10 = _load("phase10_posthoc_cm_comparison.json")
    phase10_audit = _load("phase10_high_precision_audit.json")
    known_cm = _load("strongest_known_cm_records.json")

    for rows in (phase5, phase6, phase7, phase8, phase9, phase10):
        for row in rows:
            row["type_label"] = _label(row["polarization_type"])
    known_by_type = {_label(row["polarization_type"]): row for row in known_cm}
    for row in phase10:
        known = known_by_type[row["type_label"]]
        row["strongest_known_cm_ell_squared"] = known["ell_squared_numeric"]
        row["blind_to_strongest_known_cm_ratio"] = (
            row["best_blind_ell_squared"] / known["ell_squared_numeric"]
        )

    # Figure 1: population-level distance/symmetry association.
    by_type5 = {row["type_label"]: row for row in phase5}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(TYPE_ORDER))
    mean_ratios = [
        by_type5[label]["mean_ell_squared_enhanced"]
        / by_type5[label]["mean_ell_squared_minimal_image"]
        for label in TYPE_ORDER
    ]
    axes[0].bar(x, mean_ratios, color=[TYPE_COLORS[label] for label in TYPE_ORDER])
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set(
        xticks=x,
        xticklabels=TYPE_ORDER,
        ylabel=r"mean $\ell^2$ (enhanced / minimal image)",
        title="Distance enhancement inside the CM population",
    )
    full = [by_type5[label]["extra_passive_symmetry_fraction"] for label in TYPE_ORDER]
    upper = [
        by_type5[label]["upper_quartile_extra_symmetry_fraction"]
        for label in TYPE_ORDER
    ]
    width = 0.37
    axes[1].bar(x - width / 2, full, width, label="all CM candidates", color="#9ecae1")
    axes[1].bar(x + width / 2, upper, width, label=r"top quartile of $\ell^2$", color="#08519c")
    axes[1].set(
        xticks=x,
        xticklabels=TYPE_ORDER,
        ylabel="fraction with enhanced passive image",
        title="Enhanced symmetry is enriched in the high-distance tail",
        ylim=(0, max(upper) * 1.18),
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        _style_axis(axis)
    _save(fig, "cm_population_distance_symmetry")

    # Figure 2: exactly equal-distance matched controls.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    offsets = {0.02: -0.12, 0.10: 0.12}
    markers = {0.02: "o", 0.10: "s"}
    for radius in (0.02, 0.10):
        subset = {
            row["type_label"]: row
            for row in phase7
            if abs(float(row["target_rms_geodesic_distance"]) - radius) < 1e-12
        }
        values = [subset[label]["mean_paired_ell_difference"] for label in TYPE_ORDER]
        low = [subset[label]["paired_difference_ci95_low"] for label in TYPE_ORDER]
        high = [subset[label]["paired_difference_ci95_high"] for label in TYPE_ORDER]
        error = np.asarray([np.asarray(values) - low, np.asarray(high) - values])
        ax.errorbar(
            x + offsets[radius],
            values,
            yerr=error,
            marker=markers[radius],
            linestyle="none",
            capsize=4,
            markersize=7,
            label=f"intrinsic radius {radius:.2f}",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set(
        xticks=x,
        xticklabels=TYPE_ORDER,
        ylabel=r"mean paired $\Delta\ell^2$ (generic $-$ CM)",
        title="Equal-distance generic deformations reduce distance on average",
    )
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(fig, "equal_distance_cm_controls")

    # Figure 3: adversarial local search.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for label in TYPE_ORDER:
        rows = sorted(
            (row for row in phase8 if row["type_label"] == label),
            key=lambda row: row["radius"],
        )
        ax.plot(
            [row["radius"] for row in rows],
            [row["overall_best_ratio"] for row in rows],
            marker="o",
            linewidth=2,
            color=TYPE_COLORS[label],
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="CM baseline")
    ax.set(
        xlabel="intrinsic tangent-sphere radius",
        ylabel=r"best searched $\ell^2 / \ell^2_{\rm CM}$",
        title="Adversarial local search does not beat the tested Phase-5 champions",
        ylim=(0.86, 1.006),
    )
    ax.legend(ncol=2, frameon=False)
    _style_axis(ax)
    _save(fig, "adversarial_local_search")

    # Figure 4: approximate passive-gate retention.
    radii9 = sorted(set(float(row["radius"]) for row in phase9))
    best_mean = [
        mean(row["overall_best_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    worst_mean = [
        mean(row["overall_worst_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    best_min = [
        min(row["overall_best_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    best_max = [
        max(row["overall_best_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    worst_min = [
        min(row["overall_worst_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    worst_max = [
        max(row["overall_worst_retention"] for row in phase9 if row["radius"] == radius)
        for radius in radii9
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.fill_between(radii9, best_min, best_max, alpha=0.15, color="#1f77b4")
    ax.fill_between(radii9, worst_min, worst_max, alpha=0.12, color="#d62728")
    ax.plot(radii9, best_mean, marker="o", linewidth=2.3, color="#1f77b4", label="best-retention directions")
    ax.plot(radii9, worst_mean, marker="s", linewidth=2.3, color="#d62728", label="worst-retention directions")
    ax.set(
        xlabel="intrinsic tangent-sphere radius",
        ylabel=r"approximate enhanced-gate retention $R_{0.02}$",
        title="Enhanced passive gates are fragile but direction-dependent",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(fig, "passive_gate_retention")

    # Figure 5: blind expanding-ball search.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for label in TYPE_ORDER:
        rows = sorted(
            (row for row in phase10 if row["type_label"] == label),
            key=lambda row: row["radius"],
        )
        ax.plot(
            [row["radius"] for row in rows],
            [row["blind_to_strongest_known_cm_ratio"] for row in rows],
            marker="o",
            linewidth=2,
            color=TYPE_COLORS[label],
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="strongest known CM record")
    ax.set(
        xlabel="intrinsic search-ball radius",
        ylabel=r"best blind $\ell^2$ / strongest known CM $\ell^2$",
        title="Blind bounded search does not reach the strongest known CM records",
        ylim=(0.25, 1.03),
    )
    ax.legend(ncol=2, frameon=False)
    _style_axis(ax)
    _save(fig, "blind_bounded_global_search")

    # One four-panel headline figure for README and presentations.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    # A: equal-distance controls.
    for radius in (0.02, 0.10):
        subset = {
            row["type_label"]: row
            for row in phase7
            if abs(float(row["target_rms_geodesic_distance"]) - radius) < 1e-12
        }
        values = [subset[label]["mean_paired_ell_difference"] for label in TYPE_ORDER]
        low = [subset[label]["paired_difference_ci95_low"] for label in TYPE_ORDER]
        high = [subset[label]["paired_difference_ci95_high"] for label in TYPE_ORDER]
        error = np.asarray([np.asarray(values) - low, np.asarray(high) - values])
        axes[0, 0].errorbar(
            x + offsets[radius], values, yerr=error, marker=markers[radius],
            linestyle="none", capsize=3, label=f"radius {radius:.2f}"
        )
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(xticks=x, xticklabels=TYPE_ORDER, ylabel=r"mean paired $\Delta\ell^2$", title="A. Equal-distance generic controls")
    axes[0, 0].legend(frameon=False, fontsize=9)
    # B: local adversarial.
    for label in TYPE_ORDER:
        rows = sorted((row for row in phase8 if row["type_label"] == label), key=lambda row: row["radius"])
        axes[0, 1].plot([row["radius"] for row in rows], [row["overall_best_ratio"] for row in rows], marker="o", color=TYPE_COLORS[label], label=label)
    axes[0, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="intrinsic radius", ylabel=r"best searched / CM $\ell^2$", title="B. Adversarial local search", ylim=(0.86, 1.006))
    # C: gates.
    axes[1, 0].fill_between(radii9, best_min, best_max, alpha=.15, color="#1f77b4")
    axes[1, 0].fill_between(radii9, worst_min, worst_max, alpha=.12, color="#d62728")
    axes[1, 0].plot(radii9, best_mean, marker="o", color="#1f77b4", label="best direction")
    axes[1, 0].plot(radii9, worst_mean, marker="s", color="#d62728", label="worst direction")
    axes[1, 0].set(xlabel="intrinsic radius", ylabel=r"gate retention $R_{0.02}$", title="C. Approximate passive-gate retention", ylim=(-.03, 1.03))
    axes[1, 0].legend(frameon=False, fontsize=9)
    # D: blind global.
    for label in TYPE_ORDER:
        rows = sorted((row for row in phase10 if row["type_label"] == label), key=lambda row: row["radius"])
        axes[1, 1].plot([row["radius"] for row in rows], [row["blind_to_strongest_known_cm_ratio"] for row in rows], marker="o", color=TYPE_COLORS[label], label=label)
    axes[1, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="intrinsic ball radius", ylabel=r"best blind / CM $\ell^2$", title="D. Blind bounded search", ylim=(.25, 1.03))
    axes[1, 1].legend(ncol=2, frameon=False, fontsize=8)
    for axis in axes.flat:
        _style_axis(axis)
    _save(fig, "headline_numerical_evidence")

    largest = [row for row in phase10 if abs(float(row["radius"]) - 1.5) < 1e-12]
    largest.sort(key=lambda row: TYPE_ORDER.index(row["type_label"]))
    headline_rows = [
        {
            "polarization_type": row["type_label"],
            "best_blind_ell_squared": row["best_blind_ell_squared"],
            "phase5_population_champion_ell_squared": row["cm_ell_squared"],
            "strongest_known_cm_ell_squared": row["strongest_known_cm_ell_squared"],
            "blind_to_strongest_known_cm_ratio": row["blind_to_strongest_known_cm_ratio"],
            "best_method": row["best_method"],
        }
        for row in largest
    ]
    csv_path = PROJECT / "data" / "consolidated_headline_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(headline_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(headline_rows)

    consolidated = {
        "claim": (
            "Numerical evidence that CM points are enriched near extremal regions "
            "for GKP distance and passive Clifford symmetry; no global-optimality claim."
        ),
        "polarization_types": list(TYPE_ORDER),
        "phase5": {
            "cm_candidate_count": sum(row["candidate_count"] for row in phase5),
            "type_summaries": phase5,
        },
        "phase6": {
            "generic_control_count": sum(row["control_count"] for row in phase6),
            "type_regime_summaries": phase6,
        },
        "phase7": {
            "equal_distance_control_count": sum(row["control_count"] for row in phase7),
            "all_mean_differences_negative": all(row["mean_paired_ell_difference"] < 0 for row in phase7),
            "all_descriptive_intervals_below_zero": all(row["paired_difference_ci95_high"] < 0 for row in phase7),
        },
        "phase8": {
            "objective_evaluations": 2400,
            "counterexamples_found": sum(bool(row["counterexample_found"]) for row in phase8),
            "bayesian_beats_sobol_count": sum(row["bayesian_best_ratio"] > row["sobol_best_ratio"] for row in phase8),
            "search_count": len(phase8),
        },
        "phase9": {
            "objective_evaluations": 3200,
            "nonzero_deformations_with_exact_enhanced_action": sum(
                int(row["best_retention_exact_enhanced_action_count"] > 0)
                + int(row["worst_retention_exact_enhanced_action_count"] > 0)
                for row in phase9
            ),
        },
        "phase10": {
            "objective_evaluations": 5760,
            "blind_counterexamples_found": sum(bool(row["blind_beats_cm"]) for row in phase10),
            "largest_radius_results": headline_rows,
            "maximum_high_precision_discrepancy": max(
                row["ell_absolute_discrepancy"] for row in phase10_audit
            ),
        },
        "claim_boundary": [
            "The CM populations are bounded computational samples.",
            "The generic controls are matched local real deformations and are non-CM almost surely; individual endomorphism rings are not all certified.",
            "The adversarial and blind searches are finite-budget bounded numerical searches.",
            "No theorem of local or global optimality is claimed.",
        ],
    }
    (PROJECT / "data" / "consolidated_results.json").write_text(
        json.dumps(consolidated, indent=2) + "\n"
    )
    print("Generated consolidated results and six figure sets")


if __name__ == "__main__":
    main()
