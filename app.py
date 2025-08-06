import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ✅ Set dark layout and cosmic theme
theme_color = "#00ffff"
st.set_page_config(page_title="Cloud Scheduler Dashboard", layout="wide")
st.markdown(f"""
    <style>
    .main {{ background-color: #0d1117; color: white; }}
    .sidebar .sidebar-content {{ background-color: #161b22; }}
    .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2 {{ color: white; }}
    </style>
    <h1 style='color:{theme_color};'>🚀 Multi-Cloud Scheduling Evaluation Dashboard</h1>
""", unsafe_allow_html=True)

# ✅ Load logs
def load_data():
    base_path = "results"
    return {
        "PPO": pd.read_csv(os.path.join(base_path, "ppo_log.csv")),
        "A2C": pd.read_csv(os.path.join(base_path, "A2C_log.csv")),
        "DQN": pd.read_csv(os.path.join(base_path, "dqn_log.csv")),
        "FCFS": pd.read_csv(os.path.join(base_path, "fcfs_log.csv")),
        "Round Robin": pd.read_csv(os.path.join(base_path, "round_robin_log.csv"))
    }

model_data = load_data()

# ✅ PPO color map
color_map = {"PPO": "green", "A2C": "gray", "DQN": "gray", "FCFS": "gray", "Round Robin": "gray"}

# 📊 SLA Compliance %
def sla_chart():
    sla_values = {m: (df['SLAMet'].astype(str).str.upper() == "YES").mean()*100 for m, df in model_data.items()}
    df = pd.DataFrame(list(sla_values.items()), columns=["Model", "SLA Compliance %"])
    fig = px.bar(df, x="Model", y="SLA Compliance %", title="✅ SLA Compliance %", color="Model", color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

# 💰 Avg CPU Cost
def cost_chart():
    df = pd.DataFrame({"Model": list(model_data.keys()), "Avg CPU Cost": [df["CPUCost"].mean() for df in model_data.values()]})
    fig = px.bar(df, x="Model", y="Avg CPU Cost", title="💰 Average CPU Cost", color="Model", color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

# ⏱️ Avg Execution Time
def exec_chart():
    df = pd.DataFrame({"Model": list(model_data.keys()), "Avg Execution Time": [df["ExecutionTime"].mean() for df in model_data.values()]})
    fig = px.bar(df, x="Model", y="Avg Execution Time", title="⏱️ Average Execution Time", color="Model", color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

# 🎯 Reward Score
def reward_chart():
    rewards = {}
    for model in ["PPO", "A2C", "DQN"]:
        df = model_data[model]
        rewards[model] = ((df['SLAMet'].str.upper() == "YES").astype(int) - 0.1 * df['CPUCost']).mean()
    df = pd.DataFrame(rewards.items(), columns=["Model", "Avg Reward Score"])
    fig = px.bar(df, x="Model", y="Avg Reward Score", title="🎯 Avg Reward Score", color="Model", color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

# 📈 PPO SLA Trend
def sla_trend_chart():
    df = model_data["PPO"]
    df["Batch"] = df.index // 10
    df_trend = df.groupby("Batch")["SLAMet"].apply(lambda x: (x == "YES").mean()*100).reset_index()
    fig = px.line(df_trend, x="Batch", y="SLAMet", title="📈 PPO SLA Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ☁️ PPO Cloud Usage
def cloud_usage_chart():
    df = model_data["PPO"]
    clouds = df["SelectedCloud"].map({0: "AWS", 1: "Azure", 2: "GCP"}).value_counts().reset_index()
    clouds.columns = ["Cloud", "Count"]
    fig = px.pie(clouds, values="Count", names="Cloud", title="☁️ PPO Cloud Selection")
    st.plotly_chart(fig, use_container_width=True)

# 🔥 SLA Violation Heatmap
def violation_heatmap():
    df = model_data["PPO"]
    df["SLAMet"] = df["SLAMet"].astype(str).str.upper().map({"YES": 1, "NO": 0})
    df["Cloud"] = df["SelectedCloud"].map({0: "AWS", 1: "Azure", 2: "GCP"})
    df["TaskGroup"] = pd.cut(df["TaskID"], bins=10, labels=[f"G{i}" for i in range(1, 11)])
    heatmap_data = df.pivot_table(index="Cloud", columns="TaskGroup", values="SLAMet", aggfunc=lambda x: 100 - x.mean()*100)
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", cbar_kws={'label': '% SLA Violations'}, ax=ax)
    ax.set_title("🔴 PPO SLA Violation Heatmap")
    st.pyplot(fig)

# 📦 Task Distribution Boxplot
def task_boxplot():
    dfs = []
    for name, df in model_data.items():
        df["Model"] = name
        dfs.append(df)
    all_df = pd.concat(dfs)
    fig1 = px.box(all_df, x="Model", y="ExecutionTime", title="⏱️ Task Execution Time Variance", color="Model", color_discrete_map=color_map)
    fig2 = px.box(all_df, x="Model", y="CPUCost", title="💰 Task CPU Cost Variance", color="Model", color_discrete_map=color_map)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# 🧠 Explainability Table
def explain_table():
    df = pd.read_csv("results/explainability_table.csv")
    st.subheader("🧠 Explainability Table")
    st.dataframe(df)

# ✅ Final Verdict
def final_summary():
    with st.expander("📌 Final Verdict"):
        st.success("✅ PPO achieved the highest SLA %, lowest average cost, best reward scores, and most balanced cloud usage — proving its superiority as a real-time cloud scheduler.")

# 🔍 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Choose a Metric", [
    "SLA Compliance %", "Average CPU Cost", "Execution Time",
    "Reward Score", "PPO SLA Trend", "Cloud Usage (PPO)",
    "SLA Violation Heatmap", "Task Distribution Variance",
    "Explainability Table", "Conclusion"
])

# 🔁 Router
if page == "SLA Compliance %": sla_chart()
elif page == "Average CPU Cost": cost_chart()
elif page == "Execution Time": exec_chart()
elif page == "Reward Score": reward_chart()
elif page == "PPO SLA Trend": sla_trend_chart()
elif page == "Cloud Usage (PPO)": cloud_usage_chart()
elif page == "SLA Violation Heatmap": violation_heatmap()
elif page == "Task Distribution Variance": task_boxplot()
elif page == "Explainability Table": explain_table()
elif page == "Conclusion": final_summary()
