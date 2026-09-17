import argparse
import json

import numpy as np

from correctness_gate import stats


def table1(correct: np.ndarray, sizes=(25, 50, 100, 200), n_boot=5000,
           seed=1234) -> None:
    rng = np.random.default_rng(seed)
    print("\n| items scored | SE of accuracy | 95% spread (+/- 1.96 SE) |")
    print("|---|---|---|")
    for n in sizes:
        idx = rng.integers(0, len(correct), size=(n_boot, n))
        accs = correct[idx].mean(axis=1)
        se = accs.std(ddof=1)
        print(f"| {n} | {se:.3f} | +/- {1.96 * se:.3f} |")


def table2(p_disc: float, acc: float, alpha=0.05, power=0.80) -> None:
    print(f"\n(discordant rate {p_disc:.3f}, accuracy near {acc:.2f}, "
          f"alpha {alpha}, power {power})")
    print("\n| true drop | paired n (McNemar) | unpaired n (two-proportion) |")
    print("|---|---|---|")
    for delta in (0.02, 0.03, 0.05, 0.08, 0.10):
        if delta ** 2 >= p_disc:
            paired = "n/a (drop exceeds discordant budget)"
        else:
            paired = stats.required_n_paired(p_disc, delta, alpha, power)
        unpaired = stats.required_n_unpaired(acc, delta, alpha, power)
        print(f"| {delta:.2f} | {paired} | {unpaired} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run JSON from `cgate run`")
    ap.add_argument("--second", default=None,
                    help="second run JSON; measures the real discordant rate")
    ap.add_argument("--metric", default="correct_norm",
                    choices=["correct", "correct_norm"])
    a = ap.parse_args()

    run = json.loads(open(a.run).read())
    correct = np.array([it[a.metric] for it in run["items"]], dtype=float)
    print(f"run: {run.get('model', '?')}  n={len(correct)}  "
          f"acc={correct.mean():.3f}")
    table1(correct)

    if a.second:
        other = json.loads(open(a.second).read())
        if other["fingerprint"] != run["fingerprint"]:
            raise SystemExit("second run is on a different item set")
        cand = [bool(it[a.metric]) for it in other["items"]]
        pc = stats.paired_counts([bool(x) for x in correct], cand)
        p_disc = (pc.b + pc.c) / pc.n
        print(f"\nmeasured discordant rate vs {other.get('model', '?')}: "
              f"{p_disc:.3f} ({pc.b} broke, {pc.c} fixed)")
    else:
        p_disc = 0.15
        print("\nno second run given; table 2 uses a placeholder "
              "discordant rate of 0.15. Re-run with --second after M8.")
    table2(p_disc, float(correct.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
