import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy.ma as ma
import scipy.stats as stats

# === AI INSIGHTS METHODS ===
st.markdown("""
### 🤖 AI Insights Methods
This dashboard uses multiple forms of AI to support insight generation and decision-making:
- **Rule-Based AI** (e.g., Chi-Square Test for statistical significance)
- **Generative AI (LLM)** (optional: GPT-based chart summaries, questions)
- **Predictive AI** (coming soon: ML to forecast cyber breaches)

Each AI output is labeled (🧠 Rule-Based, 🤖 GPT-Based). Golden Answers appear only when the data supports them.
""")

# === DASHBOARD TITLE ===
st.title("Air Force Cyber Breach Analysis Dashboard")

# === METHODS & LIMITATIONS ===
st.markdown("""
### 📘 Methods & Limitations
This dashboard uses seeded synthetic data to simulate cyber breach patterns across mission types and cyber risk levels.
- Color-coded heatmap with jittered data points.
- Fisher’s Exact Test to flag significant quadrant risks.
- Golden Questions & Answers auto-generated.
- Cyber Risk Levels are displayed as a legend (0 = Minimal Risk, 4 = Critical Risk).
""")

# === SEEDING & DATA REGEN BUTTON ===
st.markdown("#### 🔄 Regenerate Synthetic Data")
st.markdown("Click the button below to generate a new synthetic dataset for analysis. This will reset all visualizations and calculations.")
if "df" not in st.session_state or st.button("🔁 Regenerate Synthetic Data"):
    np.random.seed(42)
    st.session_state.df = pd.DataFrame({
        'Mission Type': np.random.choice(['Surveillance', 'Training', 'Combat', 'Logistics'], size=200),
        'Cyber Risk Level': np.random.randint(0, 5, size=200),
        'Cyber Breach History': np.random.choice([0, 1], size=200, p=[0.7, 0.3])
    })

df = st.session_state.df.copy()

# === PROBLEM STATEMENT ===
st.markdown("### 🧠 AI-Generated Smart Problem Statement")
top_breach = df.groupby(['Mission Type', 'Cyber Risk Level'])['Cyber Breach History'].mean().idxmax()
max_rate = df.groupby(['Mission Type', 'Cyber Risk Level'])['Cyber Breach History'].mean().max()
st.markdown(f"In the synthetic dataset, the highest cyber breach rate occurs for **{top_breach[0]} missions at Cyber Risk Level {top_breach[1]}**, with a breach rate of **{max_rate:.2%}**. This quadrant may represent the most critical operational vulnerability.")

# === PREPARE FOR HEATMAP ===
mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map)
df['y'] = df['Cyber Risk Level']

x_bins = np.linspace(-0.5, 3.5, 5)
y_bins = np.linspace(-0.5, 4.5, 6)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2
y_centers = (y_bins[:-1] + y_bins[1:]) / 2

heat_red, _, _ = np.histogram2d(df[df['Cyber Breach History'] == 1]['x'], df[df['Cyber Breach History'] == 1]['y'], bins=[x_bins, y_bins])
heat_blue, _, _ = np.histogram2d(df[df['Cyber Breach History'] == 0]['x'], df[df['Cyber Breach History'] == 0]['y'], bins=[x_bins, y_bins])
heat_total = heat_red + heat_blue

with np.errstate(divide='ignore', invalid='ignore'):
    proportion = np.true_divide(heat_red, heat_total)  # Compute proportions
    proportion[heat_total == 0] = np.nan              # Set NaN for empty cells

masked_proportion = ma.masked_invalid(proportion)     # Mask invalid values

# === JITTER ===
x_jitter, y_jitter = 0.1, 0.1
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter, size=len(df))

# === PLOT HEATMAP ===
fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)
im = ax.imshow(masked_proportion.T, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

mission_types = ['Surveillance', 'Training', 'Combat', 'Logistics']
coord_to_label = {(i, j): f"{mission_types[i]} @ {j}" for i in range(4) for j in range(5)}
significant_labels = []

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])   # Successes (cyber breaches)
        b = int(heat_blue[i, j]) # Failures (no breaches)
        total = r + b

        if total > 0:
            ax.text(x - 0.45,
                    y + 0.4,
                    f"{b}/{r}", ha='left',
                    va='top',
                    fontsize=8,
                    color='black',
                    alpha=0.9)

            other_r = heat_red.sum() - r  # Successes in all other quadrants
            other_b = heat_blue.sum() - b  # Failures in all other quadrants

# === LEGENDS ===

# Add red/blue legend for breach categories above heatmap
ax.legend(['No Cyber Breach', 'Cyber Breach'], loc='upper right', bbox_to_anchor=(1.25, 1.0))

# Add a single Risk Level Legend below the red/blue legend
legend_text_risk_levels = "\n".join([
    "📊 Cyber Risk Levels:",
    "0: Minimal - No major vulnerabilities.",
    "1: Low - Minor vulnerabilities.",
    "2: Moderate - Some vulnerabilities.",
    "3: High - Significant vulnerabilities.",
    "4: Critical - Severe vulnerabilities."
])

# Position Risk Level Legend below red/blue legend
ax.text(4.5, 0.5, legend_text_risk_levels, fontsize=8, verticalalignment='top', horizontalalignment='left')

# Position Risk Level Legend below red/blue legend
ax.text(4.5, 0.5, legend_text_risk_levels, fontsize=8, verticalalignment='top', horizontalalignment='left')

            # Create a valid 2x2 contingency table
            contingency_table = [[r, b], [other_r, other_b]]

            # Perform Fisher's Exact Test
            _, p_value = stats.fisher_exact(contingency_table)

            if p_value < 0.05:
                ax.text(x + 0.45,
                        y + 0.45,
                        f"p={p_value:.3f}", ha='right',
                        va='top',
                        fontsize=8,
                        color='green')
# Perform Chi-Square test
_, p_value, _, _ = stats.chi2_contingency([[r, b], [other_r, other_b]])
