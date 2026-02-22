#!/usr/bin/env python3
"""Analyze training run logs."""

import re


def main():
    with open("training_logs_full.txt", encoding="utf-8") as f:
        lines = f.readlines()

    # 1. Cost per phase
    print("=== COST PER PHASE ===")
    selfplay_costs = []
    training_costs = []
    eval_costs = []
    dollar = "$"
    for line in lines:
        m = re.search(r"Self-play phase cost \(est\): \$([\d.]+)", line)
        if m:
            selfplay_costs.append(float(m.group(1)))
        m = re.search(r"Training phase cost \(est\): \$([\d.]+)", line)
        if m:
            training_costs.append(float(m.group(1)))
        m = re.search(r"Evaluation phase cost \(est\): \$([\d.]+)", line)
        if m:
            eval_costs.append(float(m.group(1)))

    if selfplay_costs:
        print(
            f"  Self-play:  {len(selfplay_costs)} phases, total {dollar}{sum(selfplay_costs):.2f}, avg {dollar}{sum(selfplay_costs)/len(selfplay_costs):.2f}/iter"
        )
    if training_costs:
        print(
            f"  Training:   {len(training_costs)} phases, total {dollar}{sum(training_costs):.2f}, avg {dollar}{sum(training_costs)/len(training_costs):.2f}/iter"
        )
    if eval_costs:
        print(
            f"  Evaluation: {len(eval_costs)} phases, total {dollar}{sum(eval_costs):.2f}, avg {dollar}{sum(eval_costs)/len(eval_costs):.2f}/iter"
        )
    total = sum(selfplay_costs) + sum(training_costs) + sum(eval_costs)
    print(f"  GRAND TOTAL: {dollar}{total:.2f}")

    # 2. Iteration timing
    print("\n=== ITERATION TIMING ===")
    iter_times = []
    for line in lines:
        m = re.search(r"Iteration (\d+) complete in ([\d.]+) minutes", line)
        if m:
            iter_times.append((int(m.group(1)), float(m.group(2))))
    for it, t in iter_times:
        print(f"  Iter {it:2d}: {t:.1f} min")
    if iter_times:
        avg_time = sum(t for _, t in iter_times) / len(iter_times)
        total_time = sum(t for _, t in iter_times)
        print(f"  Average: {avg_time:.1f} min/iter")
        print(f"  Total:   {total_time:.1f} min ({total_time / 60:.1f} hours)")

    # 3. Promotions
    print("\n=== MODEL PROMOTIONS ===")
    promotion_chain = []
    for line in lines:
        m = re.search(r"Evaluation: (v\d+) win_rate=([\d.]+)% vs (v\d+)", line)
        if m:
            candidate, wr, champion = m.group(1), float(m.group(2)), m.group(3)
            if wr >= 55.0:
                promotion_chain.append((candidate, champion, wr))
                print(f"  PROMOTED: {candidate} beat {champion} ({wr}% WR)")

    # Find iteration for each promotion
    print("\n  Promotion chain: ", end="")
    champs = [p[1] for p in promotion_chain[:1]] + [p[0] for p in promotion_chain]
    seen = []
    for c in champs:
        if c not in seen:
            seen.append(c)
    print(" -> ".join(seen))

    # 4. Rejection stats
    promotions = len(promotion_chain)
    rejections = sum(1 for line_ in lines if "New model rejected" in line_)
    print(f"\n  Promotions: {promotions}")
    print(f"  Rejections: {rejections}")
    if promotions + rejections > 0:
        print(f"  Promotion rate: {promotions / (promotions + rejections) * 100:.1f}%")

    # 5. Win rate distribution
    print("\n=== WIN RATE DISTRIBUTION ===")
    win_rates = []
    for line in lines:
        m = re.search(r"Evaluation: v\d+ win_rate=([\d.]+)%", line)
        if m:
            win_rates.append(float(m.group(1)))
    if win_rates:
        print(
            f"  Min: {min(win_rates):.0f}%, Max: {max(win_rates):.0f}%, Mean: {sum(win_rates) / len(win_rates):.1f}%"
        )
        buckets = sorted({int(w) for w in win_rates})
        for wr_bucket in buckets:
            count = sum(1 for w in win_rates if int(w) == wr_bucket)
            bar = "#" * count
            print(f"  {wr_bucket}%: {bar} ({count})")

    # 6. Per-iteration detail table
    print("\n=== ITERATION DETAIL TABLE ===")
    print(
        f"{'Iter':>4} | {'Candidate':>10} | {'Champion':>10} | {'WinRate':>8} | {'Result':>10} | {'Time':>8}"
    )
    print("-" * 70)
    eval_results = []
    for line in lines:
        m = re.search(r"Evaluation: (v\d+) win_rate=([\d.]+)% vs (v\d+)", line)
        if m:
            eval_results.append((m.group(1), float(m.group(2)), m.group(3)))

    for i, (cand, wr, champ) in enumerate(eval_results):
        time_str = f"{iter_times[i][1]:.1f} min" if i < len(iter_times) else "?"
        result = "PROMOTED" if wr >= 55.0 else "rejected"
        print(f"{i:4d} | {cand:>10} | {champ:>10} | {wr:>7.1f}% | {result:>10} | {time_str:>8}")

    # 7. Training run summary
    print("\n=== FINAL SUMMARY ===")
    for line in lines:
        if (
            "Training loop complete" in line
            or "Results:" in line
            or "Final model" in line
            or "Total Estimated Cost" in line
        ):
            # Strip the [Orchestrator] prefix and timestamp
            m2 = re.search(r"INFO - (.+)", line)
            if m2:
                print(f"  {m2.group(1)}")

    # 8. Tournament info
    print("\n=== POST-RUN TOURNAMENT ===")
    for line in lines:
        if "Tournament:" in line and "models" in line:
            m2 = re.search(r"INFO - (.+)", line)
            if m2:
                print(f"  {m2.group(1)}")
        if "Device:" in line and "MCTS:" in line:
            m2 = re.search(r"INFO - (.+)", line)
            if m2:
                print(f"  {m2.group(1)}")

    # Check for tournament completion
    tournament_complete = any(
        "tournament" in line_.lower()
        and (
            "result" in line_.lower()
            or "ranking" in line_.lower()
            or "elo" in line_.lower()
            or "complete" in line_.lower()
        )
        for line_ in lines[-100:]
    )
    if tournament_complete:
        print("  Status: COMPLETE")
    else:
        print("  Status: STILL RUNNING (post-run tournament in progress)")

    # 9. Model versions in GCS
    print("\n=== MODELS PRODUCED ===")
    model_versions = set()
    for line in lines:
        m = re.search(r"Found latest model version: (\d+)", line)
        if m:
            model_versions.add(int(m.group(1)))
        m = re.search(r"Loaded model (v\d+)", line)
        if m:
            model_versions.add(int(m.group(1)[1:]))
    if model_versions:
        print(f"  Range: v{min(model_versions)} to v{max(model_versions)}")
        print(f"  Total model versions seen: {len(model_versions)}")


if __name__ == "__main__":
    main()
