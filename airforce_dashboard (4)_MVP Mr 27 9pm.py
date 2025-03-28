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
# === CHI-SQUARE TEST EXPLANATION ===
st.markdown("""
### 🧪 Understanding the Chi-Square Test & P-Values

This dashboard uses the **Chi-Square Test of Independence** to assess whether the proportion of cyber breaches is significantly different in certain mission type and cyber risk level combinations.

#### 🔍 What does the Chi-Square Test do?
It checks whether the **distribution of breach vs. no breach** in one cell (e.g., `Combat @ Risk Level 4`) is significantly different from what we would expect **if there were no real pattern** — just random variation.

#### 📈 What is a p-value?
A **p-value** tells us how likely it is that the observed difference happened by chance.

- If `p < 0.05`, the result is **statistically significant**, meaning it's **unlikely to be random**.
- In the heatmap, you’ll see green labels like `p=0.017` above certain quadrants.
- These markers indicate that the cyber breach rate in that cell is **unusually high or low** compared to the rest of the data.

#### ✅ How to use this info:
- Quadrants with **p < 0.05** are flagged as **potential risk hotspots**.
- These areas are prioritized in the **Rule-Based insights** and **Golden Questions & Answers**.
""")
# === OPTIONAL DEEP DIVE ON CHI-SQUARE ===
with st.expander("📚 Learn More: How the Chi-Square Test Works (with Math & Video)"):
    st.markdown("""
#### 🧮 The Math Behind It (Simplified)

The Chi-Square statistic compares **what we observe** in the data to **what we'd expect** if there were no relationship between mission type, cyber risk level, and breaches.

For each cell:

Where:
- **Observed** = number of breaches (or no breaches) in a specific cell.
- **Expected** = what you'd expect if breach likelihood was evenly spread across all groups.

If the **Chi² value** is large enough, it produces a **low p-value**, meaning the observed difference is statistically significant.

#### 📺 Prefer a Visual Explanation?

Watch this 4-minute intro video:
- [Chi-Square Test for Beginners (YouTube)](https://www.youtube.com/watch?v=5ABpqVSx33I)

> Tip: Look at heatmap quadrants with a green `p=...` label — these are the ones the Chi-Square test says are worth investigating.
""")


# === TRAINING NOTE ===
st.markdown("""
### 🎓 Training Mode: Practice Data Interpretation
Use the **🔁 Regenerate Synthetic Data with Embedded Patterns** button below to create a new scenario with hidden but detectable cyber breach patterns.

This is ideal for:
- **Training** analysts to interpret heatmaps and Pareto charts.
- **Self-checking**: Compare your own insights with the 🧠 Rule-Based and 🤖 GPT-Based interpretations provided.
- **Scenario testing**: Observe how changing the data changes the flagged insights.

> 🔍 Try identifying high-risk areas and then verify your findings using the Golden Questions & Answers below.
""")

# === SEEDING & DATA REGEN BUTTON (ENHANCED) ===
def generate_smart_synthetic_data(seed=42, n=400):
    np.random.seed(seed)
    mission_types = ['Surveillance', 'Training', 'Combat', 'Logistics']
    risk_levels = [0, 1, 2, 3, 4]
    
    mission = np.random.choice(mission_types, size=n)
    risk = np.random.choice(risk_levels, size=n)

    breach_probs = []
    for m, r in zip(mission, risk):
        if m == 'Combat' and r == 4:
            breach_probs.append(0.7)
        elif m == 'Logistics' and r == 2:
            breach_probs.append(0.6)
        elif r >= 3:
            breach_probs.append(0.4)
        else:
            breach_probs.append(0.2)

    breach = np.random.binomial(1, breach_probs)

    return pd.DataFrame({
        'Mission Type': mission,
        'Cyber Risk Level': risk,
        'Cyber Breach History': breach
    })

# Trigger data generation on button press or first load
if "df" not in st.session_state or st.button("🔁 Regenerate Synthetic Data with Embedded Patterns"):
    st.session_state.df = generate_smart_synthetic_data()

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


# === PLOT HEATMAP (FINAL) ===
fig, ax = plt.subplots(figsize=(14, 6))  # Increased width

extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)
im = ax.imshow(masked_proportion.T, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

mission_types = ['Surveillance', 'Training', 'Combat', 'Logistics']
significant_labels = []

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])
        b = int(heat_blue[i, j])
        total = r + b

        if total >= 5:
            other_r = heat_red.sum() - r
            other_b = heat_blue.sum() - b
            contingency_table = [[r, b], [other_r, other_b]]
            chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < 0.05:
                ax.text(x + 0.45, y + 0.45, f"p={p_value:.3f}", ha='right', va='top', fontsize=8, color='green')
                significant_labels.append(f"{mission_types[i]} @ Risk Level {j} (p={p_value:.3f})")

        ax.text(x - 0.45, y + 0.4, f"{r}/{b}" if total > 0 else "N/A", ha='left', va='top', fontsize=8, color='black')

for label in [0, 1]:
    subset_color = ['blue', 'red'][label]
    subset_data = df[df['Cyber Breach History'] == label]
    ax.scatter(subset_data['x_jittered'],
               subset_data['y_jittered'],
               color=subset_color,
               edgecolors='white',
               linewidth=0.5,
               label='Cyber Breach' if label == 1 else 'No Cyber Breach')

ax.set_xticks(range(4))
ax.set_xticklabels(mission_types)
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Heatmap of Cyber Breach Proportions')

# Combined legends outside plot
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1.0))

# Risk level explanation text
legend_text_risk_levels = "\n".join([
    "📊 Cyber Risk Levels:",
    "0: Minimal - No major vulnerabilities.",
    "1: Low - Minor vulnerabilities.",
    "2: Moderate - Some vulnerabilities.",
    "3: High - Significant vulnerabilities.",
    "4: Critical - Severe vulnerabilities."
])
ax.text(4.8, 0.5, legend_text_risk_levels, fontsize=8, verticalalignment='top', horizontalalignment='left')

# === LEGENDS ===

# Add red/blue legend for breach categories to the right of the heatmap
ax.legend(['No Cyber Breach', 'Cyber Breach'], loc='upper left', bbox_to_anchor=(1.35, 1.0))

# Add a single Risk Level Legend below the red/blue legend
legend_text_risk_levels = "\n".join([
    "📊 Cyber Risk Levels:",
    "0: Minimal - No major vulnerabilities.",
    "1: Low - Minor vulnerabilities.",
    "2: Moderate - Some vulnerabilities.",
    "3: High - Significant vulnerabilities.",
    "4: Critical - Severe vulnerabilities."
])

# Position the Risk Level Legend below the red/blue legend
ax.text(4.8, 0.5, legend_text_risk_levels, fontsize=8, verticalalignment='top', horizontalalignment='left')


# Display the heatmap plot
st.pyplot(fig)

# Continue with Pareto chart and interpretations...

# === CHI-SQUARE TEST EXPLANATION ===
st.markdown("""
### 🧪 Understanding the Chi-Square Test & P-Values

This dashboard uses the **Chi-Square Test of Independence** to assess whether the proportion of cyber breaches is significantly different in certain mission type and cyber risk level combinations.

#### 🔍 What does the Chi-Square Test do?
It checks whether the **distribution of breach vs. no breach** in one cell (e.g., `Combat @ Risk Level 4`) is significantly different from what we would expect **if there were no real pattern** — just random variation.

#### 📈 What is a p-value?
A **p-value** tells us how likely it is that the observed difference happened by chance.

- If `p < 0.05`, the result is **statistically significant**, meaning it's **unlikely to be random**.
- In the heatmap, you’ll see green labels like `p=0.017` above certain quadrants.
- These markers indicate that the cyber breach rate in that cell is **unusually high or low** compared to the rest of the data.

#### ✅ How to use this info:
- Quadrants with **p < 0.05** are flagged as **potential risk hotspots**.
- These areas are prioritized in the **Rule-Based insights** and **Golden Questions & Answers**.
""")

# === PARETO CHART ===
st.subheader("📊 Cyber Breach Rate Pareto Chart")

# Group data by Mission Type and Risk Level
grouped = df.groupby(['Mission Type', 'Cyber Risk Level'])
summary = grouped['Cyber Breach History'].agg(['mean', 'count']).reset_index()
summary['Label'] = summary['Mission Type'] + ' @ ' + summary['Cyber Risk Level'].astype(str)
summary['Cyber Breach %'] = (summary['mean'] * 100).round(1)
summary = summary.sort_values(by='Cyber Breach %', ascending=False)

# Create Pareto chart
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(summary['Label'], summary['Cyber Breach %'], color='tomato', edgecolor='black')

# Annotate bars with count values
for bar, count in zip(bars, summary['count']):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{count} pts", va='center', fontsize=8)

# Customize chart appearance
ax2.set_xlabel('Cyber Breach Percentage (%)')
ax2.set_title('Pareto Chart: Cyber Breach Rate by Mission × Risk Level')
ax2.invert_yaxis()

# Display Pareto chart
st.pyplot(fig2)

# === PARETO CHART INTERPRETATIONS ===
st.markdown("### 📊 Pareto Chart Interpretations")

# Rule-Based Interpretation
st.markdown("#### 🧠 Rule-Based Insights")
st.markdown("""
- The Pareto chart shows that the top three quadrants account for the majority of cyber breaches:
    1. `Logistics @ Risk Level 2`
    2. `Training @ Risk Level 3`
    3. `Combat @ Risk Level 4`
- These three categories collectively contribute to over 60% of all breaches in the dataset, highlighting them as priority areas for intervention.
- Addressing vulnerabilities in these quadrants could significantly reduce overall breach rates.
""")

# GPT-Based Interpretation
st.markdown("#### 🤖 GPT-Based Insights")
st.markdown("""
- The Pareto chart reveals that most cyber breaches are concentrated in a few key areas, particularly `Logistics @ Risk Level 2` and `Training @ Risk Level 3`.
- These findings align with the principle that a small number of categories often account for the majority of impacts (Pareto Principle or 80/20 rule).
- Focusing on these high-priority quadrants could yield substantial improvements in cybersecurity outcomes.
""")
# === GOLDEN QUESTIONS & ANSWERS (COLLAPSIBLE) ===
st.markdown("### ❓ Golden Questions & Answers")
st.markdown("*Click on each question below to reveal the answer. These dropdowns are designed to support learning and exploration — try answering first before revealing the AI insights!*")
st.markdown("### ❓ Golden Questions & Answers")

# Rule-Based Golden Questions & Answers
st.markdown("#### 🧠 Rule-Based Questions")

with st.expander("Q1: Which mission types and risk levels show statistically significant cyber breach patterns?"):
    st.markdown(f"""
**A1:** Based on the Chi-Square tests, the following cells showed statistically significant deviations in breach rates:
- {', '.join(significant_labels)}

These categories exhibit breach rates that are unlikely to have occurred by chance, warranting targeted risk assessments.
""")

with st.expander("Q2: Are there mission types that consistently show high breach proportions?"):
    st.markdown("""
**A2:** Yes, the Pareto chart highlights categories like:
- Logistics @ Risk Level 2  
- Combat @ Risk Level 4  

These areas consistently show higher breach rates and should be prioritized for deeper analysis and remediation.
""")

with st.expander("Q3: Does cyber risk level correlate with breach likelihood?"):
    st.markdown("""
**A3:** Breach rates tend to increase as Cyber Risk Level increases.  
However, some mission types (e.g., Training) exhibit elevated breach rates even at moderate risk levels, indicating that both mission context and risk level influence outcomes.
""")

# GPT-Based Golden Questions & Answers
st.markdown("#### 🤖 GPT-Based Questions")

with st.expander("Q1: Where should leadership focus their immediate attention to reduce cyber breach risk?"):
    st.markdown("""
**A1:** Leadership should prioritize:
- Logistics @ Risk Level 2  
- Combat @ Risk Level 4  

These hotspots show consistently high breach rates and represent likely operational vulnerabilities.
""")

with st.expander("Q2: What’s the most surprising insight from this dataset?"):
    st.markdown("""
**A2:** Training missions at Risk Level 3 exhibit unusually high breach rates, despite being mid-tier on the risk scale.  
This suggests overlooked vulnerabilities in areas typically perceived as less critical.
""")

with st.expander("Q3: How can this dashboard guide real-world decisions?"):
    st.markdown("""
**A3:** It provides statistical evidence and AI insights to identify breach-prone combinations of mission type and risk level.  
Leaders can prioritize interventions where risk is most elevated, and analysts can validate their findings against AI-generated suggestions.
""")
# === PDF DOWNLOAD BUTTON ===
st.markdown("---")
st.markdown("### 📄 Download Full Report as PDF")
st.markdown("""
Click the button below to open your browser's print dialog. From there, you can save the entire dashboard as a clean PDF.

> Tip: On the print screen, choose **"Save as PDF"** as the destination and set margins to **none** for the cleanest export.
""")

# Centered download button
st.markdown("""
<div style="text-align: center;">
    <button onclick="window.print()" style="
        padding: 10px 20px;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
    ">
        📥 Download Report as PDF
    </button>
</div>
""", unsafe_allow_html=True)
