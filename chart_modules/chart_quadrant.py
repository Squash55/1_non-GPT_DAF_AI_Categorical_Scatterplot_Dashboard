import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def show_chart_quadrant(df):
    st.markdown("## 📊 Multi-Chart Risk Analytics Quadrant")
    col1, col2 = st.columns(2)

    bubble_df = df.groupby(['Mission Type', 'Cyber Risk Level']).agg(
        breach_rate=('Cyber Breach History', 'mean'),
        count=('Cyber Breach History', 'count')
    ).reset_index()

    with col1:
        st.markdown("#### 🕸️ Radar Chart")
        radar_df = df.groupby('Mission Type')['Cyber Breach History'].mean().reset_index()
        radar_df = pd.concat([radar_df, radar_df.iloc[[0]]])
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df['Cyber Breach History'],
            theta=radar_df['Mission Type'],
            fill='toself'
        ))
        fig_radar.update_layout(showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("#### 🔵 Bubble Chart")
        fig_bubble = px.scatter(
            bubble_df, x='Mission Type', y='Cyber Risk Level',
            size='count', color='breach_rate',
            color_continuous_scale='RdBu', labels={'breach_rate': 'Breach %'}
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col1:
        st.markdown("#### 🌳 Decision Tree")
        fig_tree, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.2, 0.7, "Low Risk
(<=2)", ha='center', bbox=dict(boxstyle="round", fc="lightblue"))
        ax.text(0.5, 0.7, "Moderate
(Level 3)", ha='center', bbox=dict(boxstyle="round", fc="khaki"))
        ax.text(0.8, 0.7, "High Risk
(Level 4)", ha='center', bbox=dict(boxstyle="round", fc="salmon"))
        st.pyplot(fig_tree)

    with col2:
        st.markdown("#### 🔁 Sankey Diagram")
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

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
            link=links
        )])
        st.plotly_chart(fig_sankey, use_container_width=True)
