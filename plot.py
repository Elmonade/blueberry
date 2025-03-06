import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/plot.csv")

plt.figure(figsize=(12, 8))
bars = plt.barh(df["Algorithm"], df["Time_ms"], color="skyblue")

for bar in bars:
    width = bar.get_width()
    label_x_pos = width * 1.01
    plt.text(
        label_x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f} ms",
        va="center",
        fontsize=9,
    )

plt.title("Matrix Operation Performance Comparison", fontsize=16)
plt.xlabel("Execution Time (ms) ", fontsize=12)
plt.ylabel("Algorithm", fontsize=12)

# Add grid lines for better readability
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Tight layout
plt.tight_layout()

# Show plot
plt.savefig("matrix_performance_Ofast.png", dpi=300)
plt.show()

# Log scale
plt.figure(figsize=(12, 8))

# Sort by execution time for the remaining plots
df_sorted = df.sort_values("Time_ms")
plt.barh(df_sorted["Algorithm"], df_sorted["Time_ms"], color="lightgreen")
plt.xscale("log")  # Use logarithmic scale

# Add title and labels
plt.title("Matrix Operation Performance Comparison (Log Scale)", fontsize=16)
plt.xlabel("Execution Time (ms) - Log Scale ", fontsize=12)
plt.ylabel("Algorithm", fontsize=12)

plt.grid(axis="x", linestyle="--", alpha=0.7)

# Add data labels
for i, v in enumerate(df_sorted["Time_ms"]):
    plt.text(v * 1.1, i, f"{v:.1f} ms", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("matrix_performance_log_Ofast.png", dpi=300)
plt.show()

# Calculate speedup relative to the baseline
baseline = df.loc[df["Algorithm"] == "Normal", "Time_ms"].values[0]
df_sorted["Speedup"] = baseline / df_sorted["Time_ms"]

# Speed Up
plt.figure(figsize=(12, 8))
speedup_bars = plt.barh(df_sorted["Algorithm"], df_sorted["Speedup"], color="salmon")

# Add data labels
for bar in speedup_bars:
    width = bar.get_width()
    plt.text(
        width * 1.01,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}x",
        va="center",
        fontsize=9,
    )

# Add title and labels
plt.title("Speedup Relative to Blocked Algorithm", fontsize=16)
plt.xlabel("Speedup Factor", fontsize=12)
plt.ylabel("Algorithm", fontsize=12)
plt.axvline(x=1, color="black", linestyle="--", alpha=0.7)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("matrix_speedup_Ofast.png", dpi=300)
plt.show()
