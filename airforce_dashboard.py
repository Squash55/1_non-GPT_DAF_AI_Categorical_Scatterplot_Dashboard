import streamlit as st
import pandas as pd
import numpy as np

from chart_modules.chart_quadrant import show_chart_quadrant

def generate_data(seed=42, n=400):
    np.random.seed(seed)
    missions = ['Surveillance', 'Training', 'Combat', 'Logistics']
    risks = list(range(5))
    mission = np.random.choice(missions, size=n)
    risk = np.random.choice(risks, size=n)
    breach_probs = [0.7 if m == 'Combat' and r == 4 else 0.6 if m == 'Logistics' and r == 2 else 0.4 if r >= 3 else 0.2
                    for m, r in zip(mission, risk)]
    breach = np.random.binomial(1, breach_probs)
    return pd.DataFrame({'Mission Type': mission, 'Cyber Risk Level': risk, 'Cyber Breach History': breach})

df = generate_data()
show_chart_quadrant(df)
