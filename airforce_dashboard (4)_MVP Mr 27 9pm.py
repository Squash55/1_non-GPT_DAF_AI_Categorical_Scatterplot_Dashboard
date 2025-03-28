import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import chi2_contingency

# === AI INSIGHTS METHODS ===
st.markdown("""
### 🤖 AI Insights Methods
This dashboard uses multiple forms of AI to support insight generation and decision-making:

- **Rule-Based AI**: This includes statistical methods like the Chi-Square Test of Independence to identify significant patterns in cyber breach data. For example:
    - The heatmap uses statistical tests to flag quadrants where breach rates are disproportionately high compared to expected proportions.
    - The Pareto chart highlights categories that contribute most to cyber breaches, helping prioritize mitigation efforts.

- **Generative AI (GPT)**: This provides natural language interpretations of visualizations, offering insights in a conversational tone. These GPT-generated insights complement rule-based findings by summarizing trends and suggesting actionable strategies.

Each AI output is labeled:
- 🧠 Rule-Based for insights derived from statistical methods.
- 🤖 GPT-Based for generative AI explanations.
""")

# === DASHBOARD TITLE ===
st.title("Air Force Cyber Breach Analysis Dashboard")

# === METHODS & LIMITATIONS ===
st.markdown("""
### 📘 Methods & Limitations
This dashboard uses seeded synthetic data to simulate cyber breach patterns across mission types and cyber risk levels.
- Color-coded heatmap with jittered data points.
- Chi-Square Test to flag significant quadrant risks.
- Golden Questions & Answers auto-generated.
- Cyber Risk Levels are displayed as a legend (0 = Minimal Risk, 4 = Critical Risk).
""")

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
jitter = st.slider("Jitter Amount", min_value=0.0, max_value=0.5, value=0.1, step=0.
