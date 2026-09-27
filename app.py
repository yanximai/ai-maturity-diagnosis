import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import scipy.stats as stats

# ============ 页面配置 ============
st.set_page_config(
    page_title="学科AI成熟度诊断工具",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 全局美化样式 ============
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stTitle { color: #333 !important; font-weight: 800 !important; }
    div[data-testid="stBlock"] {
        background: white; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06); margin-bottom: 1.5rem;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    div[data-testid="stBlock"]:hover {
        transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9ff, #eef0ff); border-radius: 12px;
        padding: 1rem; text-align: center; border: none;
        box-shadow: 0 2px 8px rgba(102,126,234,0.1);
    }
    div[data-testid="metric-container"] label { color: #888 !important; font-size: 0.85rem !important; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #667eea !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    .stSubheader { color: #444 !important; font-weight: 700 !important;
        border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;
        display: inline-block; margin-bottom: 1rem !important; }
    .stAlert, .stInfo, .stError { border-radius: 12px !important; }
    .stInfo { background: #f8f9ff !important; }
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #667eea, #764ba2); color: white;
        border: none; border-radius: 10px; padding: 0.5rem 2rem;
        font-weight: 600; transition: all 0.3s;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8f9ff, #eef0ff);
        border: 2px dashed #667eea; border-radius: 15px; padding: 1rem;
    }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e8e8e8; }
    div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid #e8e8e8; background: white; }
    div[data-testid="stTabs"] button { border-radius: 10px 10px 0 0 !important; font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important;
    }
    .stCode { border-radius: 12px !important; border: 1px solid #e8e8e8 !important; }
    hr { border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 2rem 0; }
    .footer { text-align: center; padding: 2rem; color: #aaa; font-size: 0.9rem;
        border-top: 1px solid rgba(0,0,0,0.05); margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# ============ 模型函数 ============
def bass_cumulative(t, p, q):
    """标准 Bass 模型（饱和水平归一化为 1）"""
    return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def bass_cumulative_with_M(t, p, q, M):
    """带饱和水平的 Bass 模型"""
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def logistic_cumulative(t, a, b, M):
    """Logistic 模型"""
    return M / (1 + np.exp(-a * (t - b)))


def gompertz_cumulative(t, a, b, M):
    """Gompertz 模型"""
    return M * np.exp(-b * np.exp(-a * t))


# ============ 政策建议生成 ============
def generate_recommendations(stage, dtype, F_current, M_fit):
    recommendations = []
    if stage == "萌芽期":
        recommendations.append("🎯 **设立种子基金**：资助 AI+学科 交叉研究项目，降低初期探索门槛")
        recommendations.append("📚 **建设知识库**：整理 AI 在该学科的成功案例，降低学习成本")
        recommendations.append("🤝 **举办研讨会**：邀请先行者分享经验，促进学术交流")
    elif stage == "萌芽后期":
        recommendations.append("🚀 **加大投入**：增加 AI 基础设施投入，迎接即将到来的爆发期")
        recommendations.append("🎓 **人才培养**：开设 AI+学科 培训课程，储备人才")
        recommendations.append("📋 **制定标准**：建立 AI 在该学科的应用规范和评价标准")
    elif stage == "爆发期":
        recommendations.append("⚡ **搭建共享平台**：建设算力、数据、模型共享平台")
        recommendations.append("📝 **加强监管**：制定 AI 伦理准则，防范滥用风险")
        recommendations.append("🌐 **促进合作**：鼓励跨机构、跨学科协作研究")
    elif stage == "成熟期":
        recommendations.append("💡 **深化应用**：从简单应用到复杂系统集成，提升 AI 应用深度")
        recommendations.append("📊 **质量评估**：建立 AI 研究成果的质量评估体系")
        recommendations.append("🔍 **寻找新方向**：探索 AI 与其他前沿技术的结合")
    elif stage == "饱和期":
        recommendations.append("🔄 **范式创新**：寻找突破性技术方向，避免内卷")
        recommendations.append("🌍 **拓展应用**：将 AI 能力迁移到新的子领域")
        recommendations.append("📈 **国际合作**：引入外部资源，开辟新增长空间")
    if dtype == "创新主导型":
        recommendations.append("💪 **强化创新生态**：保护原创性研究，建立创新激励机制")
    elif dtype == "模仿主导型":
        recommendations.append("🔄 **培育创新能力**：在模仿基础上逐步培养自主创新能力")
    return recommendations


# ============ 单个学科分析函数（供批量对比调用） ============
def analyze_discipline(df, discipline_name=None):
    """对单个学科数据做完整分析，返回结果字典"""
    df = df.copy()
    if not all(col in df.columns for col in ["year", "ai_count", "total_count"]):
        return None
    df = df.sort_values("year").reset_index(drop=True)
    df["cum_ai"] = df["ai_count"].cumsum()
    df["cum_total"] = df["total_count"].cumsum()
    df["F_actual"] = df["cum_ai"] / df["cum_total"]
    if len(df) < 5:
        return None

    t = (df["year"] - df["year"].min()).values.astype(float)
    F_actual = df["F_actual"].values
    ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)

    # Bass 模型
    bass_result, bass_r2 = None, -999
    try:
        M_init = min(1.0, F_actual[-1] * 1.5)
        if M_init <= 0:
            M_init = 0.5
        popt, _ = curve_fit(
            bass_cumulative_with_M, t, F_actual,
            p0=[0.001, 0.1, M_init],
            bounds=([0, 0, 0.01], [0.5, 1.5, 1.0]), maxfev=10000
        )
        p_fit, q_fit, M_fit = popt
        F_bass = bass_cumulative_with_M(t, p_fit, q_fit, M_fit)
        bass_r2 = 1 - np.sum((F_actual - F_bass) ** 2) / ss_tot if ss_tot > 0 else 0
        bass_result = {"params": {"p": p_fit, "q": q_fit, "M": M_fit}, "fitted": F_bass, "r2": bass_r2}
    except Exception:
        pass

    # 回退：标准 Bass
    if bass_result is None:
        try:
            popt, _ = curve_fit(bass_cumulative, t, F_actual, p0=[0.001, 0.1], maxfev=10000)
            p_fit, q_fit = popt
            M_fit = 1.0
            F_bass = bass_cumulative(t, p_fit, q_fit)
            bass_r2 = 1 - np.sum((F_actual - F_bass) ** 2) / ss_tot if ss_tot > 0 else 0
            bass_result = {"params": {"p": p_fit, "q": q_fit, "M": M_fit}, "fitted": F_bass, "r2": bass_r2}
        except Exception:
            return None

    p_fit, q_fit, M_fit = bass_result["params"]["p"], bass_result["params"]["q"], bass_result["params"]["M"]
    F_fitted = bass_result["fitted"]
    r2 = bass_result["r2"]

    # 拐点
    t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
    year_star = df["year"].min() + t_star

    # 阶段
    F_current = df["F_actual"].iloc[-1]
    current_year = df["year"].max()
    dist_to_star = abs(current_year - year_star)
    if F_current < 0.05:
        stage = "萌芽期"
    elif F_current < 0.10 and current_year < year_star:
        stage = "萌芽后期"
    elif F_current >= 0.10 and dist_to_star <= 3:
        stage = "爆发期"
    elif F_current >= 0.40 and dist_to_star > 3:
        stage = "成熟期"
    elif F_current >= 0.75:
        stage = "饱和期"
    else:
        stage = "发展期"

    # 扩散动力
    if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
        dtype = "创新主导型"
    elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
        dtype = "模仿主导型"
    else:
        dtype = "均衡扩散型"

    # 自创指标
    df["annual_F"] = df["ai_count"] / df["total_count"]
    df["渗透率变化率"] = df["annual_F"].diff()
    df["渗透加速度"] = df["渗透率变化率"].diff()
    acceleration = df["渗透加速度"].iloc[-1] if not pd.isna(df["渗透加速度"].iloc[-1]) else 0
    cv = df["total_count"].std() / df["total_count"].mean() if df["total_count"].mean() > 0 else 0
    activity_index = 1 / (1 + cv)
    lock_in_risk = F_current / M_fit if M_fit > 0 else 0
    if lock_in_risk > 0.9:
        risk_level = "高风险"
    elif lock_in_risk > 0.7:
        risk_level = "中等风险"
    else:
        risk_level = "低风险"

    return {
        "name": discipline_name or "当前学科",
        "df": df,
        "F_current": F_current,
        "stage": stage,
        "dtype": dtype,
        "p_fit": p_fit,
        "q_fit": q_fit,
        "M_fit": M_fit,
        "year_star": year_star,
        "r2": r2,
        "F_fitted": F_fitted,
        "acceleration": acceleration,
        "activity_index": activity_index,
        "lock_in_risk": lock_in_risk,
        "risk_level": risk_level,
    }


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📤 数据上传")
    uploaded_files = st.file_uploader(
        "上传CSV文件（支持多选，批量对比）",
        type=["csv"],
        accept_multiple_files=True,
        help="每个文件需包含 year、ai_count、total_count 三列"
    )
    st.markdown("---")
    st.markdown("### 📋 数据格式要求")
    st.info("""
    **必需列：**
    - year：年份
    - ai_count：AI相关文献数
    - total_count：总文献数
    
    **示例：**
    ```
    year,ai_count,total_count
    2000,331,50000
    2001,386,52000
    ```
    """)
    st.markdown("---")
    st.markdown("### 💡 使用步骤")
    st.caption("""
    1️⃣ 准备学科年度数据  
    2️⃣ 上传 CSV 文件  
    3️⃣ 自动生成诊断报告  
    4️⃣ 可下载分析结果
    """)

# ============ 标题 ============
st.markdown(
    '<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);'
    'padding:2rem 3rem;border-radius:20px;margin-bottom:2rem;'
    'box-shadow:0 10px 30px rgba(102,126,234,0.3);">'
    '<h1 style="color:white;font-size:2.8rem;font-weight:800;margin:0 0 0.5rem 0;">'
    '🧠 学科AI成熟度诊断工具</h1>'
    '<p style="color:rgba(255,255,255,0.9);font-size:1.1rem;margin:0;">'
    '基于 Bass / Logistic / Gompertz 多模型对比的学科人工智能融合程度智能诊断系统</p>'
    '</div>',
    unsafe_allow_html=True
)

# ============ 主逻辑 ============
if uploaded_files:
    results = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            res = analyze_discipline(df, discipline_name=f.name.replace(".csv", ""))
            if res is not None:
                results.append(res)
            else:
                st.error(f"❌ 文件 {f.name} 数据不完整或无法拟合，已跳过")
        except Exception as e:
            st.error(f"❌ 文件 {f.name} 读取失败：{e}")

    if not results:
        st.stop()

    # 批量对比（>=2 个学科）
    if len(results) >= 2:
        st.markdown('<div class="custom-card"><h3>📊 多学科学期对比排名</h3></div>', unsafe_allow_html=True)
        sorted_r = sorted(results, key=lambda x: x["F_current"], reverse=True)
        rank_df = pd.DataFrame([
            {"排名": i + 1, "学科": r["name"], "当前渗透率": f"{r['F_current']*100:.2f}%",
             "阶段": r["stage"], "拟合优度 R²": f"{r['r2']:.4f}",
             "技术锁定风险": r["risk_level"]}
            for i, r in enumerate(sorted_r)
        ])
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="custom-card"><h3>🕸️ 多学科健康度蛛网图对比</h3></div>', unsafe_allow_html=True)
        cats = ["渗透深度", "增长速度", "发展潜力", "稳定性", "成熟度"]
        fig_cmp = go.Figure()
        for r in results:
            vals = [
                min(r["F_current"] * 5, 1),
                min(abs(r["acceleration"]) * 100, 1),
                min(1 - r["lock_in_risk"], 1),
                r["activity_index"],
                min(r["F_current"], 1),
            ]
            fig_cmp.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=r["name"]))
        fig_cmp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=500)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # 逐个学科展示详情
    for res in results:
        df = res["df"]
        st.markdown(f"## 📚 {res['name']}")

        # 数据概览
        st.markdown('<div class="custom-card"><h3>📊 数据概览</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">'
                        f'{df["year"].min()}—{df["year"].max()}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">数据年份范围</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{len(df)} 年</div>'
                        f'<div style="font-size:0.85rem;color:#888;">总年数</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{df["total_count"].sum():,}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">总文献量</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{df["ai_count"].sum():,}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">AI文献量</div></div>', unsafe_allow_html=True)

        # 模型参数
        st.markdown('<div class="custom-card"><h3>🔬 模型参数</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{res["p_fit"]:.6f}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">创新系数 p</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{res["q_fit"]:.6f}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">模仿系数 q</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{res["M_fit"]*100:.1f}%</div>'
                        f'<div style="font-size:0.85rem;color:#888;">饱和水平 M</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{int(round(res["year_star"]))}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">拐点年份</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div style="text-align:center;padding:1rem;background:linear-gradient(135deg,#f8f9ff,#eef0ff);'
                        f'border-radius:12px;"><div style="font-size:1.6rem;font-weight:700;color:#667eea;">{res["r2"]:.4f}</div>'
                        f'<div style="font-size:0.85rem;color:#888;">拟合优度 R²</div></div>', unsafe_allow_html=True)

        # 扩散动力
        st.markdown('<div class="custom-card"><h3>🔀 扩散动力类型</h3></div>', unsafe_allow_html=True)
        if res["p_fit"] > res["q_fit"] and (res["p_fit"] - res["q_fit"]) >= 0.02:
            st.info(f"**创新主导型**：学科内部自发的前沿探索是推动 AI 技术发展的主要动力。")
        elif res["q_fit"] > res["p_fit"] and (res["q_fit"] - res["p_fit"]) >= 0.02:
            st.warning(f"**模仿主导型**：学科对 AI 技术的采纳主要源于对同行或相关领域成功实践的模仿与学习。")
        else:
            st.info(f"**均衡扩散型**：创新与模仿两种力量在扩散过程中作用相当。")

        # 诊断结果
        st.markdown('<div class="custom-card"><h3>🏥 当前发展阶段</h3></div>', unsafe_allow_html=True)
        stage_color = {"萌芽期": "#2e7d32", "萌芽后期": "#e65100", "爆发期": "#c62828",
                      "发展期": "#1565c0", "成熟期": "#6a1b9a", "饱和期": "#37474f"}.get(res["stage"], "#333")
        st.markdown(
            f'<div style="text-align:center;padding:1.5rem;background:#f8f9ff;border-radius:15px;">'
            f'<div style="display:inline-block;padding:0.6rem 2rem;border-radius:30px;font-weight:700;'
            f'font-size:1.3rem;background:linear-gradient(135deg,#e8f5e9,#c8e6c9);color:{stage_color};'
            f'box-shadow:0 4px 10px rgba(0,0,0,0.1);">{res["stage"]}</div>'
            f'<p style="margin-top:1rem;color:#666;font-size:0.95rem;">'
            f'累计渗透率：{res["F_current"]*100:.2f}% | 扩散类型：{res["dtype"]} | '
            f'距拐点：{abs(df["year"].max()-res["year_star"]):.1f} 年</p></div>',
            unsafe_allow_html=True
        )

        # 自创诊断指标
        st.markdown('<div class="custom-card"><h3>📈 自创诊断指标</h3></div>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("AI渗透加速度", f"{res['acceleration']*10000:.2f}‱",
                   help="正值表示加速渗透，负值表示增速放缓")
        cc2.metric("学科活跃指数", f"{res['activity_index']:.3f}",
                   help="越接近1表示学科发文越稳定活跃")
        cc3.metric("技术锁定风险", res["risk_level"],
                   delta=f"{res['lock_in_risk']*100:.1f}%",
                   help="当前渗透率占饱和水平的比例")

        # 图表
        st.markdown('<div class="custom-card"><h3>📈 可视化分析</h3></div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 发文量趋势", "📉 模型拟合", "🔥 敏感性热力图", "🕸️ 健康度蛛网图"])

        with tab1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df["year"], y=df["ai_count"], mode="lines+markers",
                                      name="AI发文量", line=dict(color="#667eea", width=3)))
            fig1.update_layout(title="年度 AI 发文量趋势", xaxis_title="年份", yaxis_title="发文量",
                               height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["year"], y=df["F_actual"], mode="markers",
                                      name="实际值", marker=dict(color="#ff6b6b", size=8)))
            fig2.add_trace(go.Scatter(x=df["year"], y=res["F_fitted"], mode="lines",
                                      name="Bass拟合", line=dict(color="#667eea", width=3)))
            fig2.update_layout(title="Bass 模型拟合效果", xaxis_title="年份", yaxis_title="累计渗透率 F(t)",
                               height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            t = (df["year"] - df["year"].min()).values.astype(float)
            F_actual = df["F_actual"].values
            ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
            p_r = np.linspace(res["p_fit"] * 0.5, res["p_fit"] * 1.5, 20)
            q_r = np.linspace(res["q_fit"] * 0.5, res["q_fit"] * 1.5, 20)
            P, Q = np.meshgrid(p_r, q_r)
            R2 = np.zeros_like(P)
            for i in range(len(p_r)):
                for j in range(len(q_r)):
                    try:
                        Ft = bass_cumulative_with_M(t, P[i, j], Q[i, j], res["M_fit"])
                        r2t = 1 - np.sum((F_actual - Ft) ** 2) / ss_tot if ss_tot > 0 else 0
                        R2[i, j] = r2t
                    except Exception:
                        R2[i, j] = 0
            fig_hm = go.Figure(data=go.Heatmap(
                z=R2, x=p_r, y=q_r, colorscale="Viridis",
                hovertemplate="p=%{x:.4f}<br>q=%{y:.4f}<br>R²=%{z:.4f}<extra></extra>"))
            fig_hm.update_layout(title="参数敏感性热力图", xaxis_title="创新系数 p",
                                 yaxis_title="模仿系数 q", height=400)
            st.plotly_chart(fig_hm, use_container_width=True)

        with tab4:
            cats = ["渗透深度", "增长速度", "发展潜力", "稳定性", "成熟度"]
            vals = [
                min(res["F_current"] * 5, 1),
                min(abs(res["acceleration"]) * 100, 1),
                min(1 - res["lock_in_risk"], 1),
                res["activity_index"],
                min(res["F_current"], 1),
            ]
            fig_rd = go.Figure()
            fig_rd.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                name=res["name"], line=dict(color="#667eea", width=2)))
            fig_rd.add_trace(go.Scatterpolar(
                r=[0.5] * 6, theta=cats + [cats[0]], mode="lines",
                name="基准线(0.5)", line=dict(color="gray", dash="dash")))
            fig_rd.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                 showlegend=True, height=400)
            st.plotly_chart(fig_rd, use_container_width=True)

        # 动态时间轴动画
        st.markdown('<div class="custom-card"><h3>▶️ 扩散过程动画</h3></div>', unsafe_allow_html=True)
        future_years = np.arange(df["year"].max() + 1, 2051)
        t_future = future_years - df["year"].min()
        F_future = bass_cumulative_with_M(t_future.astype(float), res["p_fit"], res["q_fit"], res["M_fit"])
        ymin, ymax = int(df["year"].min()), int(future_years[-1])
        year_sel = st.slider("选择年份", ymin, ymax, int(df["year"].max()), key=f"slider_{res['name']}")
        fig_anim = go.Figure()
        hist = df[df["year"] <= year_sel]
        if not hist.empty:
            fig_anim.add_trace(go.Scatter(x=hist["year"], y=hist["F_actual"],
                                          mode="lines+markers", name="历史数据",
                                          line=dict(color="#667eea", width=2)))
        fig_anim.add_trace(go.Scatter(x=[year_sel], y=[df[df["year"] == year_sel]["F_actual"].values[0]
                                if year_sel in df["year"].values else F_future[year_sel - df["year"].max() - 1]],
                                      mode="markers", name="当前位置",
                                      marker=dict(color="red", size=12)))
        if year_sel > df["year"].max():
            fi = year_sel - df["year"].max() - 1
            fig_anim.add_trace(go.Scatter(x=future_years[:fi + 1], y=F_future[:fi + 1],
                                          mode="lines", name="未来预测",
                                          line=dict(color="#ff9800", width=2, dash="dot")))
        cur_F = df[df["year"] == year_sel]["F_actual"].values[0] if year_sel in df["year"].values else F_future[year_sel - df["year"].max() - 1]
        fig_anim.update_layout(title=f"{year_sel}年 渗透率：{cur_F*100:.2f}%",
                              xaxis_title="年份", yaxis_title="累计渗透率", height=400)
        st.plotly_chart(fig_anim, use_container_width=True)

        # 未来预测
        st.markdown('<div class="custom-card"><h3>🔮 未来25年渗透率预测（2026—2050）</h3></div>', unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            "年份": future_years,
            "累计渗透率预测": [f"{v*100:.2f}%" for v in F_future]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # 政策建议
        st.markdown('<div class="custom-card"><h3>💡 政策建议</h3></div>', unsafe_allow_html=True)
        recs = generate_recommendations(res["stage"], res["dtype"], res["F_current"], res["M_fit"])
        for r in recs:
            st.markdown(r)

        # 导出报告
        st.markdown('<div class="custom-card"><h3>📥 导出诊断报告</h3></div>', unsafe_allow_html=True)
        report_lines = []
        report_lines.append(f"# 学科AI成熟度诊断报告 —— {res['name']}")
        report_lines.append("")
        report_lines.append("## 基本信息")
        report_lines.append(f"- 数据范围：{df['year'].min()}—{df['year'].max()}（共{len(df)}年）")
        report_lines.append(f"- 总文献量：{df['total_count'].sum():,}")
        report_lines.append(f"- AI相关文献：{df['ai_count'].sum():,}")
        report_lines.append("")
        report_lines.append("## 模型参数")
        report_lines.append(f"- 创新系数 p：{res['p_fit']:.6f}")
        report_lines.append(f"- 模仿系数 q：{res['q_fit']:.6f}")
        report_lines.append(f"- 饱和水平 M：{res['M_fit']*100:.2f}%")
        report_lines.append(f"- 拐点年份：{int(res['year_star'])}")
        report_lines.append(f"- 拟合优度 R²：{res['r2']:.4f}")
        report_lines.append("")
        report_lines.append("## 诊断结论")
        report_lines.append(f"- 当前阶段：{res['stage']}")
        report_lines.append(f"- 当前累计渗透率：{res['F_current']*100:.2f}%")
        report_lines.append(f"- 扩散动力类型：{res['dtype']}")
        report_lines.append(f"- AI渗透加速度：{res['acceleration']*10000:.2f}‱")
        report_lines.append(f"- 学科活跃指数：{res['activity_index']:.3f}")
        report_lines.append(f"- 技术锁定风险：{res['risk_level']}（{res['lock_in_risk']*100:.1f}%）")
        report_lines.append("")
        report_lines.append("## 政策建议")
        for r in recs:
            report_lines.append(f"- {r}")
        report_text = "\n".join(report_lines)
        st.download_button(
            label="⬇️ 下载报告 (.md)",
            data=report_text.encode("utf-8"),
            file_name=f"学科AI诊断报告_{res['name']}_{df['year'].max()}.md",
            mime="text/markdown",
            key=f"dl_{res['name']}"
        )

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;background:white;border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);">
        <div style="font-size:5rem;margin-bottom:1rem;">📤</div>
        <h2 style="color:#667eea;margin-bottom:1rem;">欢迎使用学科AI成熟度诊断工具</h2>
        <p style="color:#666;font-size:1.1rem;">请从左侧上传包含学科年度发文量的 CSV 文件</p>
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-top:2rem;">
            <div style="background:#f8f9ff;padding:1.5rem;border-radius:12px;min-width:150px;">
                <div style="font-size:2rem;">📊</div><div style="color:#666;margin-top:0.5rem;">多模型拟合</div></div>
            <div style="background:#f8f9ff;padding:1.5rem;border-radius:12px;min-width:150px;">
                <div style="font-size:2rem;">🏥</div><div style="color:#666;margin-top:0.5rem;">阶段诊断</div></div>
            <div style="background:#f8f9ff;padding:1.5rem;border-radius:12px;min-width:150px;">
                <div style="font-size:2rem;">🔮</div><div style="color:#666;margin-top:0.5rem;">未来预测</div></div>
            <div style="background:#f8f9ff;padding:1.5rem;border-radius:12px;min-width:150px;">
                <div style="font-size:2rem;">📥</div><div style="color:#666;margin-top:0.5rem;">报告导出</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.code("year,ai_count,total_count\n2000,331,50000\n2001,386,52000\n2002,380,54000", language="csv")

st.markdown('<div class="footer"><p>🧠 学科AI成熟度诊断工具 | 基于 Bass/Logistic/Gompertz 多模型对比</p>'
            '<p>© 2026 科研数据分析工具</p></div>', unsafe_allow_html=True)
