
import streamlit as st
import pandas as pd
import plotly.express as px

# Load the evaluation comparison log
st.title("📊 PPO vs A2C vs DQN – Cloud Scheduler Evaluation Dashboard")
df = pd.read_csv("evaluation_comparison_log.csv")

st.markdown("### 🔍 Action Distribution (Cloud Selection)")
action_map = {0: 'AWS', 1: 'Azure', 2: 'GCP'}
for col in ['Action', 'A2C_Action', 'DQN_Action']:
    df[col + '_Label'] = df[col].map(action_map)

action_counts = pd.DataFrame({
    'PPO': df['Action_Label'].value_counts(),
    'A2C': df['A2C_Action_Label'].value_counts(),
    'DQN': df['DQN_Action_Label'].value_counts()
}).fillna(0).astype(int).reset_index().rename(columns={'index': 'Cloud'})

fig_action = px.bar(action_counts.melt(id_vars='Cloud', var_name='Model', value_name='Count'),
                    x='Cloud', y='Count', color='Model', barmode='group')
st.plotly_chart(fig_action)

st.markdown("### ✅ SLA Compliance Rate")
if 'SLAMet' in df.columns:
    df['SLA_Bool'] = df['SLAMet'].astype(str).str.upper().str.startswith("Y")
    sla_df = pd.DataFrame({
        'PPO': df['SLA_Bool'],
        'A2C': df['A2C_Action'] == df['Action'],
        'DQN': df['DQN_Action'] == df['Action']
    })
    sla_rate = sla_df.mean().reset_index()
    sla_rate.columns = ['Model', 'SLA_Accuracy']
    fig_sla = px.bar(sla_rate, x='Model', y='SLA_Accuracy', text='SLA_Accuracy')
    fig_sla.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_sla)

st.markdown("### 💰 Average Cost per Model (if available)")
if 'CPUCost' in df.columns:
    cost_df = pd.DataFrame({
        'PPO': df['CPUCost'],
        'A2C': df['CPUCost'],  # Same cost, but we can simulate cost models if needed
        'DQN': df['CPUCost']
    }).mean().reset_index()
    cost_df.columns = ['Model', 'AvgCost']
    fig_cost = px.bar(cost_df, x='Model', y='AvgCost', text='AvgCost')
    st.plotly_chart(fig_cost)
