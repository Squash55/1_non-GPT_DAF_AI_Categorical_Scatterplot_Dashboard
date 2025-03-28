import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def show_chart_quadrant(df):
    st.markdown("## 📊 Multi-Chart Risk Analytics Quadrant")
    col1, col2 = st.columns(2)

    # Precompute for charts
    bubble_df = df.groupby(['Mission Type', 'Cyber Risk Level']).agg(
        breach_rate=('Cyber Breach History', 'mean'),
        count=('Cyber Breach History', 'count')
    ).reset_index()

    # === 1. Radar Chart ===
    with col1:
        st.markdown("#### 🕸️ Radar Chart: Breach Risk by Mission Type")
        radar_df = df.groupby('Mission Type')['Cyber Breach History'].mean().reset_index()
        radar_df.columns = ['Mission Type', 'Breach Rate']
        radar_df = pd.concat([radar_df, radar_df.iloc[[0]]])  # Close loop

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df['Breach Rate'],
            theta=radar_df['Mission Type'],
            fill='toself',
            name='Breach Rate'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)
        with st.expander("🧠 Radar Insights & Golden Q&A"):
            st.markdown("- Outer spikes = higher risk.
- Use radar shape to identify standout mission types.")
            st.markdown("**Q: Which mission type has highest breach rate?**
A: Usually 'Combat' or 'Logistics' based on data.")

    # === 2. Bubble Chart ===
    with col2:
        st.markdown("#### 🔵 Bubble Chart: Breach Rate vs. Mission & Risk")
        fig_bubble = px.scatter(
            bubble_df,
            x='Mission Type',
            y='Cyber Risk Level',
            size='count',
            color='breach_rate',
            color_continuous_scale='RdBu',
            labels={'breach_rate': 'Breach %'}
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        with st.expander("🧠 Bubble Chart Insights & Golden Q&A"):
            st.markdown("- Large red bubbles = high-risk zones.
- Use bubble size and color together to prioritize.")
            st.markdown("**Q: Which mission/risk combo is most breach-prone?**
A: Follow the largest darkest bubbles.")

    # === 3. Decision Tree Graphic ===
    with col1:
        st.markdown("#### 🌳 Decision Tree: Risk Guidance View")
        fig_tree, ax_tree = plt.subplots(figsize=(5, 4))
        ax_tree.axis('off')
        ax_tree.set_title("Decision Tree View")
        ax_tree.text(0.1, 0.5, "Low Risk
(<=2)", ha='center', va='center', bbox=dict(boxstyle="round", fc="lightblue"))
        ax_tree.text(0.1, 0.2, "⬅ Few Breaches", ha='center')
        ax_tree.text(0.5, 0.5, "Moderate Risk
(Level 3)", ha='center', va='center', bbox=dict(boxstyle="round", fc="khaki"))
        ax_tree.text(0.5, 0.2, "↔ Mixed Results", ha='center')
        ax_tree.text(0.9, 0.5, "High Risk
(Level 4)", ha='center', va='center', bbox=dict(boxstyle="round", fc="salmon"))
        ax_tree.text(0.9, 0.2, "➡ Mostly Breaches", ha='center')
        st.pyplot(fig_tree)
        with st.expander("🧠 Decision Tree Insights & Golden Q&A"):
            st.markdown("- Progresses left to right by risk level.
- Use as a decision-making guide.")
            st.markdown("**Q: How does risk level relate to breach?**
A: Breaches increase consistently from left to right.")

    # === 4. Sankey Diagram ===
    with col2:
        st.markdown("#### 🔁 Sankey Diagram: Mission → Risk → Outcome")
        sankey_df = pd.DataFrame({
            'source': df['Mission Type'],
            'intermediate': df['Cyber Risk Level'].astype(str),
            'target': df['Cyber Breach History'].replace({0: 'No Breach', 1: 'Breach'})
        })

        link_1 = sankey_df.groupby(['source', 'intermediate']).size().reset_index(name='count')
        link_2 = sankey_df.groupby(['intermediate', 'target']).size().reset_index(name='count')

        labels = list(pd.unique(sankey_df[['source', 'intermediate', 'target']].values.ravel()))
        label_map = {label: i for i, label in enumerate(labels)}

        links = []
        for _, row in link_1.iterrows():
            links.append(dict(source=label_map[row['source']], target=label_map[row['intermediate']], value=row['count']))
        for _, row in link_2.iterrows():
            links.append(dict(source=label_map[row['intermediate']], target=label_map[row['target']], value=row['count']))

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
            link=links
        )])
        sankey_fig.update_layout(font_size=10)
        st.plotly_chart(sankey_fig, use_container_width=True)
        with st.expander("🧠 Sankey Flow Insights & Golden Q&A"):
            st.markdown("- Shows how breaches flow from mission type through risk levels to outcomes.")
            st.markdown("**Q: Where does breach risk concentrate?**
A: Follow thickest paths toward 'Breach'.")
