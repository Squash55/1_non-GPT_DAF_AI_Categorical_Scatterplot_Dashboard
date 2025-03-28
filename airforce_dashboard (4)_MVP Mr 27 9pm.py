import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import chi2_contingency

# === AI INSIGHTS METHODS ===
st.markdown("### 🤖 AI Insights Methods\nThis dashboard uses multiple forms of AI to support insight generation and decision-making:\n- **Rule-Based AI** (e.g. Chi-Square Test, heuristics)\n- **Generative AI (LLM)** (optional: GPT-based chart summaries, questions)\n- **Predictive AI** (coming soon: ML to forecast cyber breaches)\n\nEach AI output is labeled (🧠 Rule-Based, 🤖 GPT-Based). Golden Answers appear only when the data supports them.")

# === DASHBOARD TITLE ===
st.title("Air Force Cyber Breach Analysis Dashboard")

# === METHODS & LIMITATIONS ===
st.markdown("### 📘 Methods & Limitations\nThis dashboard uses seeded synthetic data to simulate cyber breach patterns across mission types and cyber risk levels.\n- Color-coded heatmap with jittered data points\n- Chi-Square Test to flag significant quadrant risks\n- Golden Questions & Answers auto-generated\n- Cyber Risk Levels are displayed as a legend (0 = Minimal, 4 = Critical)")

# === SEEDING & DATA REGEN BUTTON ===
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

masked_proportion = np.ma.fix_invalid(proportion)     # Mask invalid values

# === JITTER ===
jitter = st.slider("Jitter Amount", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
df['x_jittered'] = df['x'] + np.random.normal(0, jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, jitter, size=len(df))

# === PLOT HEATMAP ===
fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)
im = ax.imshow(masked_proportion.T, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

mission_types = ['Surveillance', 'Training', 'Combat', 'Logistics']
significant_labels = []

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])   # Successes (cyber breaches)
        b = int(heat_blue[i, j]) # Failures (no breaches)
        total = r + b
        
        if total >= 5: # Only consider cells with at least 5 observations
            other_r = heat_red.sum() - r
            other_b = heat_blue.sum() - b
            
            # Create contingency table
            contingency_table = [[r, b], [other_r, other_b]]
            
            # Perform Chi-Square test
            chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
            
            if p_value < 0.05: # Significant result
                ax.text(x + 0.45, y + 0.45,
                        f"p={p_value:.3f}", ha='right',
                        va='top', fontsize=8,
                        color='green')
                significant_labels.append(f"{mission_types[i]} @ Risk Level {j} (p={p_value:.3f})")
        
        # Display proportions in each quadrant consistently in black text
        ax.text(x - 0.45,
                y + 0.4,
                f"{r}/{b}" if total > 0 else "N/A",
                ha='left',
                va='top',
                fontsize=8,
                color='black')

for label in [0, 1]:
    subset_color = ['blue', 'red'][label]
    subset_data = df[df['Cyber Breach History'] == label]
    ax.scatter(subset_data['x_jittered'],
               subset_data['y_jittered'],
               color=subset_color,
               edgecolors='white',
               linewidth=0.5)

ax.set_xticks(range(4))
ax.set_xticklabels(mission_types)
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Heatmap of Cyber Breach Proportions')
ax.legend(['No Cyber Breach', 'Cyber Breach'], loc='upper left')

st.pyplot(fig)

# === SIGNIFICANT RESULTS WITH EXPLANATION ===
st.markdown("### 🔬 Statistically Significant Quadrants")
if significant_labels:
    st.success(f"{len(significant_labels)} quadrant(s) show statistically significant cyber breach differences (p < 0.05):")
    for label in significant_labels:
        st.write("🔺", label)
else:
    st.info("No quadrant showed a statistically significant difference.")

# === PARETO CHART ===
st.subheader("📊 Cyber Breach Rate Pareto Chart")

grouped = df.groupby(['Mission Type', 'Cyber Risk Level'])
summary = grouped['Cyber Breach History'].agg(['mean', 'count']).reset_index()
summary['Label'] = summary['Mission Type'] + ' @ ' + summary['Cyber Risk Level'].astype(str)
summary['Cyber Breach %'] = (summary['mean'] * 100).round(1)
summary = summary.sort_values(by='Cyber Breach %', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(summary['Label'], summary['Cyber Breach %'], color='tomato', edgecolor='black')

# Corrected loop to annotate bars
for bar, count in zip(bars, summary['count']):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{count} pts", va='center', fontsize=8)

ax2.set_xlabel('Cyber Breach Percentage (%)')
ax2.set_title('Pareto Chart: Cyber Breach Rate by Mission × Risk Level')
ax2.invert_yaxis()

st.pyplot(fig2)

