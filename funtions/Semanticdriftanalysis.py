import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- 1. FILE PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

input_path        = project_root / "data" / "CSV" / "hipHopDataset.csv"
yearly_out_path   = project_root / "data" / "CSV" / "similarity_by_year.csv"
decade_out_path   = project_root / "data" / "CSV" / "similarity_by_decade.csv"

# --- 2. LOAD & PARSE ---
print("Loading hip-hop dataset...")
df = pd.read_csv(input_path)

def parse_vector(val):
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=float)
    return np.array(val, dtype=float)

df["embeddings"] = df["embeddings"].apply(parse_vector)
print(f"  Loaded {len(df)} songs across {df['Year'].nunique()} years")

# --- 3. MEAN VECTOR PER YEAR ---
year_vectors = (
    df.groupby("Year")["embeddings"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
    .sort_index()
)
years = year_vectors.index.tolist()

# --- 4. YEAR-OVER-YEAR COSINE SIMILARITY ---
yearly_rows = []
for i in range(len(years) - 1):
    y_a, y_b = years[i], years[i + 1]
    sim = cosine_similarity(
        year_vectors[y_a].reshape(1, -1),
        year_vectors[y_b].reshape(1, -1)
    )[0][0]
    yearly_rows.append({"Year_A": y_a, "Year_B": y_b, "Cosine_Similarity": round(sim, 6)})

yearly_df = pd.DataFrame(yearly_rows)
yearly_df.to_csv(yearly_out_path, index=False)
print(f"  Year-over-year results saved to: {yearly_out_path.name}")

# --- 5. MEAN VECTOR PER DECADE ---
def get_decade(year):
    return f"{(year // 10) * 10}s"

df["Decade"] = df["Year"].apply(get_decade)

decade_vectors = (
    df.groupby("Decade")["embeddings"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
    .sort_index()
)
decades = decade_vectors.index.tolist()

# --- 6. DECADE-OVER-DECADE COSINE SIMILARITY ---
decade_rows = []
for i in range(len(decades) - 1):
    d_a, d_b = decades[i], decades[i + 1]
    sim = cosine_similarity(
        decade_vectors[d_a].reshape(1, -1),
        decade_vectors[d_b].reshape(1, -1)
    )[0][0]
    decade_rows.append({"Decade_A": d_a, "Decade_B": d_b, "Cosine_Similarity": round(sim, 6)})

decade_df = pd.DataFrame(decade_rows)
decade_df.to_csv(decade_out_path, index=False)
print(f"  Decade-over-decade results saved to: {decade_out_path.name}")

# --- 7. PLOT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle("Hip-Hop Lyric Semantic Drift (Cosine Similarity)",
             fontsize=16, fontweight="bold", y=0.99)

# ── 7a. LINE GRAPH — Year-over-Year ──
years_b = yearly_df["Year_B"].tolist()
sims    = yearly_df["Cosine_Similarity"].tolist()
avg     = np.mean(sims)
std     = np.std(sims)
low_threshold = avg - std

# Base line + fill
ax1.plot(years_b, sims, color="#4C72B0", linewidth=2, zorder=3)
ax1.fill_between(years_b, sims, alpha=0.1, color="#4C72B0")

# Reference lines
ax1.axhline(avg,           color="gray",    linestyle="--", linewidth=1,   label=f"Mean ({avg:.4f})")
ax1.axhline(avg - std,     color="#C44E52", linestyle=":",  linewidth=1.2, label=f"Mean − 1 SD ({avg-std:.4f})")

# Highlight notable dips in red
for year, sim in zip(years_b, sims):
    if sim < low_threshold:
        ax1.plot(year, sim, "o", color="#C44E52", markersize=8, zorder=5)
        ax1.annotate(f"{year}\n{sim:.4f}",
                     xy=(year, sim),
                     xytext=(0, -22), textcoords="offset points",
                     ha="center", fontsize=7.5, color="#C44E52", fontweight="bold")
    else:
        ax1.plot(year, sim, "o", color="#4C72B0", markersize=4, zorder=4)

ax1.set_title("Year-over-Year Semantic Drift  (red = notable shift  ≥ 1 SD below mean)", fontsize=12)
ax1.set_ylabel("Cosine Similarity")
ax1.set_xlabel("Year")
ax1.set_xlim(min(years_b) - 0.5, max(years_b) + 0.5)
ax1.set_ylim(min(sims) - 0.03, 1.0)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax1.legend(fontsize=9)
ax1.grid(axis="both", linestyle="--", alpha=0.3)

# ── 7b. CLUSTER SCATTER — Individual song similarity per decade ──
# For each song, compute its cosine similarity to the mean vector of its decade
# This shows the SPREAD within each decade, not just the transition between them

decade_palette = {
    "1990s": "#4C72B0",
    "2000s": "#55A868",
    "2010s": "#C44E52",
    "2020s": "#8172B2",
}

all_decades_ordered = sorted(df["Decade"].unique())
decade_x_map = {d: i for i, d in enumerate(all_decades_ordered)}

for decade in all_decades_ordered:
    decade_songs   = df[df["Decade"] == decade]
    decade_mean    = decade_vectors[decade]
    x_pos          = decade_x_map[decade]
    color          = decade_palette.get(decade, "#999999")

    # Cosine similarity of each song vs its decade mean vector
    song_sims = []
    for vec in decade_songs["embeddings"]:
        sim = cosine_similarity(vec.reshape(1, -1), decade_mean.reshape(1, -1))[0][0]
        song_sims.append(sim)

    # Jitter x so dots don't stack on top of each other
    jitter = np.random.uniform(-0.25, 0.25, size=len(song_sims))

    ax2.scatter(
        [x_pos + j for j in jitter],
        song_sims,
        c=color, s=25, alpha=0.55, zorder=3,
        edgecolors="white", linewidths=0.3,
        label=decade
    )

    # Mean line per cluster
    mean_sim = np.mean(song_sims)
    ax2.hlines(mean_sim, x_pos - 0.35, x_pos + 0.35,
               colors=color, linewidths=2.5, zorder=4)
    ax2.text(x_pos, mean_sim + 0.003, f"μ={mean_sim:.4f}",
             ha="center", fontsize=8.5, fontweight="bold", color=color)

    # Std band
    std_sim = np.std(song_sims)
    ax2.fill_between(
        [x_pos - 0.35, x_pos + 0.35],
        mean_sim - std_sim, mean_sim + std_sim,
        color=color, alpha=0.12, zorder=2
    )

ax2.set_title(
    "Song-Level Cosine Similarity to Decade Mean  (each dot = 1 song  |  bar = mean  |  band = ±1 SD)",
    fontsize=11
)
ax2.set_ylabel("Cosine Similarity to Decade Mean Vector")
ax2.set_xlabel("Decade")
ax2.set_xticks(list(decade_x_map.values()))
ax2.set_xticklabels(all_decades_ordered, fontsize=11)
ax2.set_xlim(-0.6, len(all_decades_ordered) - 0.4)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax2.legend(title="Decade", fontsize=9, loc="lower right")
ax2.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plot_path = project_root / "data" / "similarity_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Plot saved to: {plot_path}")
plt.show()