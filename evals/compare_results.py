
import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_FILE = Path(__file__).parent / "results" / "results.json"

with open(RESULTS_FILE, encoding="utf-8") as f:
    report = json.load(f)

models = report["comparison"]

names = [m["model"] for m in models]
costs = [m["total_cost_usd"] for m in models]
latencies = [m["avg_latency_seconds"] for m in models]
pass_rates = [m["pass_rate"] * 100 for m in models]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(names, costs)
axes[0].set_title("Total Cost (USD)")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(names, latencies)
axes[1].set_title("Average Latency (seconds)")
axes[1].tick_params(axis="x", rotation=20)

axes[2].bar(names, pass_rates)
axes[2].set_title("Pass Rate (%)")
axes[2].set_ylim(0, 100)
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()

output = Path(__file__).parent / "results" / "model_comparison.png"
plt.savefig(output, dpi=150)
print(f"Graph saved to: {output}")
