import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(page_title="学科AI成熟度诊断工具", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
.title-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem 3rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(102,126,234,0.3); }
.title-container h1 { color: white !important; font-size: 2.8rem !important; font-weight: 800 !important; }
.title-container p { color: rgba(255,255,255,0.9) !important; font-size: 1.1rem !important; }
.custom-card { background: white; padding: 1.5rem 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
.custom-card h3 { color: #333; font-weight: 700; margin-bottom: 1rem; border-bottom: 3px solid #667eea; display: inline-block; padding-bottom: 0.5rem; }
.metric-box { text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9ff, #eef0ff); border-radius: 12px; margin: 0.5rem 0; }
.metric-box .value { font-size: 1.6rem; font-weight: 700; color: #667eea; }
.metric-box .label { font-size: 0.85rem; color: #888; margin-top: 0.3rem; }
.footer { text-align: center; padding: 2rem; color: #aaa; font-size: 0.9rem; border-top: 1px solid rgba(0,0,0,0.05); margin-top: 3rem; }
.stage-badge { display: inline-block; padding: 0.6rem 2rem; border-radius: 30px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-container"><h1>🧠 学科AI成熟度诊断工具</h1><p>基于 Bass 扩散模型的学科人工智能融合程度智能诊断系统</p></div>', unsafe_allow_html=True)

def bass_cumulative(t, p, q):
    return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def bass_cumulative_with_M(t, p, q, M):
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

uploaded_file = st.file_uploader("上传CSV文件（需包含 year、ai_count、total_count 三列）", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not all(col in df.columns for col in ["year", "ai_count", "total_count"]):
        st.error("❌ CSV文件必须包含 year、ai_count、total_count 三列")
        st.stop()
    
    df = df.sort_values("year").reset_index(drop=True)
    df["cum_ai"] = df["ai_count"].cumsum()
    df["cum_total"] = df["total_count"].cumsum()
    df["F_actual"] = df["cum_ai"] / df["cum_total"]
    
    if len(df) < 5:
        st.error("❌ 数据不足5年，Bass模型无法稳定拟合")
        st.stop()
    
    st.success(f"✅ 数据加载成功：共 {len(df)} 年（{df['year'].min()}—{df['year'].max()}）")
    
    t = (df["year"] - df["year"].min()).values.astype(float)
    F_actual = df["F_actual"].values
    
    try:
        M_init = min(1.0, F_actual[-1] * 1.5)
        if M_init <= 0:
            M_init = 0.5
        popt, _ = curve_fit(bass_cumulative_with_M, t, F_actual, p0=[0.001, 0.1, M_init], bounds=([0, 0, 0.01], [0.5, 1.5, 1.0]), maxfev=10000)
        p_fit, q_fit, M_fit = popt
        F_fitted = bass_cumulative_with_M(t, p_fit, q_fit, M_fit)
    except Exception:
        popt, _ = curve_fit(bass_cumulative, t, F_actual, p0=[0.001, 0.1], maxfev=10000)
        p_fit, q_fit = popt
        M_fit = 1.0
        F_fitted = bass_cumulative(t, p_fit, q_fit)
    
    ss_res = np.sum((F_actual - F_fitted) ** 2)
    ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
    year_star = df["year"].min() + t_star
    
    st.markdown('<div class="custom-card"><h3>📊 数据概览</h3></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="value">{df["year"].min()}—{df["year"].max()}</div><div class="label">数据年份范围</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="value">{len(df)} 年</div><div class="label">总年数</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="value">{df["total_count"].sum():,}</div><div class="label">总文献量</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="value">{df["ai_count"].sum():,}</div><div class="label">AI文献量</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card"><h3>🔬 模型参数</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="value">{p_fit:.6f}</div><div class="label">创新系数 p</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="value">{q_fit:.6f}</div><div class="label">模仿系数 q</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="value">{M_fit*100:.1f}%</div><div class="label">饱和水平 M</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="value">{int(round(year_star))}</div><div class="label">拐点年份</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-box"><div class="value">{r2:.4f}</div><div class="label">拟合优度 R²</div></div>', unsafe_allow_html=True)
    
    F_current = df["F_actual"].iloc[-1]
    current_year = df["year"].max()
    dist_to_star = abs(current_year - year_star)
    
    if F_current < 0.05:
        stage = "萌芽期"
        desc = f"AI技术渗透率仅为{F_current*100:.1f}%，远低于10%的创新鸿沟阈值。"
    elif F_current < 0.10 and current_year < year_star:
        stage = "萌芽后期"
        desc = f"渗透率{F_current*100:.1f}%，接近10%临界点。预计{int(year_star)}年前后将迎来快速增长拐点。"
    elif F_current >= 0.10 and dist_to_star <= 3:
        stage = "爆发期"
        desc = f"渗透率{F_current*100:.1f}%，已跨过10%临界点，当前处于理论拐点附近。"
    elif F_current >= 0.40 and dist_to_star > 3:
        stage = "成熟期"
        desc = f"渗透率{F_current*100:.1f}%，超过40%。AI已成为该学科主流研究方法之一。"
    elif F_current >= 0.75:
        stage = "饱和期"
        desc = f"渗透率高达{F_current*100:.1f}%，接近饱和水平(M={M_fit*100:.1f}%)。"
    else:
        stage = "发展期"
        desc = f"渗透率{F_current*100:.1f}%，处于持续增长阶段。距离理论拐点还有{int(dist_to_star)}年。"
    
    if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
        dtype = "创新主导型"
    elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
        dtype = "模仿主导型"
    else:
        dtype = "均衡扩散型"
    
    st.markdown(f'<div class="custom-card"><h3>🏥 诊断结果</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;padding:1.5rem;"><div class="stage-badge" style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);color:#2e7d32;">{stage}</div><p style="margin-top:1rem;color:#666;">{desc}</p><p style="color:#999;font-size:0.9rem;">扩散类型：{dtype} | 累计渗透率：{F_current*100:.2f}%</p></div>', unsafe_allow_html=True)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["year"], y=df["ai_count"], mode="lines+markers", name="AI发文量", line=dict(color="#667eea", width=3)))
    fig1.update_layout(title="年度 AI 发文量趋势", height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["year"], y=F_actual, mode="markers", name="实际值", marker=dict(color="#ff6b6b", size=8)))
    fig2.add_trace(go.Scatter(x=df["year"], y=F_fitted, mode="lines", name="Bass拟合", line=dict(color="#667eea", width=3)))
    fig2.update_layout(title="Bass 模型拟合效果", height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    
    df["annual_F"] = df["ai_count"] / df["total_count"]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df["year"], y=df["annual_F"] * 100, name="年度渗透率", marker_color="rgba(102,126,234,0.6)"))
    fig3.add_trace(go.Scatter(x=df["year"], y=df["F_actual"] * 100, mode="lines+markers", name="累计渗透率", line=dict(color="#764ba2", width=3)))
    fig3.update_layout(title="渗透率变化趋势", height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)
    
    future_years = np.arange(df["year"].max() + 1, 2051)
    t_future = future_years - df["year"].min()
    F_future = bass_cumulative_with_M(t_future.astype(float), p_fit, q_fit, M_fit)
    pred_df = pd.DataFrame({"年份": future_years, "累计渗透率预测": [f"{v*100:.2f}%" for v in F_future]})
    st.markdown('<div class="custom-card"><h3>🔮 未来预测（2026—2050）</h3></div>', unsafe_allow_html=True)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    report = f"# 学科AI成熟度诊断报告\n\n## 基本信息\n- 数据范围：{df['year'].min()}—{df['year'].max()}（共{len(df)}年）\n- 总文献量：{df['total_count'].sum():,}\n- AI相关文献：{df['ai_count'].sum():,}\n\n## 模型参数\n- 创新系数 p：{p_fit:.6f}\n- 模仿系数 q：{q_fit:.6f}\n- 饱和水平 M：{M_fit*100:.2f}%\n- 拐点年份：{int(year_star)}\n- 拟合优度 R²：{r2:.4f}\n\n## 诊断结论\n- 当前阶段：{stage}\n- 当前累计渗透率：{F_current*100:.2f}%\n- 扩散动力类型：{dtype}"
    st.download_button("📥 下载诊断报告", report.encode("utf-8"), file_name=f"学科AI诊断报告_{df['year'].max()}.md")

else:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem; background:white; border-radius:20px; box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
        <div style='font-size:5rem; margin-bottom:1rem;'>📤</div>
        <h2 style='color:#667eea; margin-bottom:1rem;'>欢迎使用学科AI成熟度诊断工具</h2>
        <p style='color:#666; font-size:1.1rem;'>请从上方上传包含学科年度发文量的 CSV 文件</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("year,ai_count,total_count\n2000,331,50000\n2001,386,52000\n2002,380,54000", language="csv")

st.markdown('<div class="footer"><p>🧠 学科AI成熟度诊断工具 © 2026</p></div>', unsafe_allow_html=True)