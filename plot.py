import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/plot.csv")

plt.figure(figsize=(12, 8))

# Create line plot
plt.plot(df["Algorithm"], df["Time_ms"], marker='o', linestyle='-', linewidth=2, color="skyblue")

# Add data points and labels
for i, (algo, time) in enumerate(zip(df["Algorithm"], df["Time_ms"])):
    plt.text(
        i,
        time * 1.05,  # Slightly above the point
        f"{time:.1f} ms",
        ha="center",
        fontsize=9,
    )

# Add titles and labels
plt.title("Matrix Operation Performance Comparison", fontsize=16)
plt.ylabel("Execution Time (ms)", fontsize=12)
plt.xlabel("Block Size", fontsize=12)

# Rotate x-axis labels if they overlap
plt.xticks(rotation=45, ha='right')

# Add grid lines for better readability
plt.grid(linestyle="--", alpha=0.7)

# Tight layout
plt.tight_layout()

# Save and show plot
plt.savefig("blockSizes.png", dpi=300)
plt.show()
