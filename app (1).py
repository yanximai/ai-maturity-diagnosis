import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# ============ 页面配置 ============
st.set_page_config(
    page_title="学科AI成熟度诊断工具",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义CSS美化 ============
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 标题区域 */
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .title-container h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .title-container p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    
    /* 卡片样式 */
    .custom-card {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: none;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .custom-card h3 {
        color: #333;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* 指标数值 */
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9ff, #eef0ff);
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-box .label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
    }
    
    /* 阶段标签 */
    .stage-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .stage-萌芽期 { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); color: #2e7d32; }
    .stage-萌芽后期 { background: linear-gradient(135deg, #fff3e0, #ffe0b2); color: #e65100; }
    .stage-爆发期 { background: linear-gradient(135deg, #ffebee, #ffcdd2); color: #c62828; }
    .stage-发展期 { background: linear-gradient(135deg, #e3f2fd, #bbdefb); color: #1565c0; }
    .stage-成熟期 { background: linear-gradient(135deg, #f3e5f5, #e1bee7); color: #6a1b9a; }
    .stage-饱和期 { background: linear-gradient(135deg, #eceff1, #cfd8dc); color: #37474f; }
    
    /* 上传区域美化 */
    .upload-area {
        background: linear-gradient(135deg, #f8f9ff, #eef0ff);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 页脚 */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #aaa;
        font-size: 0.9rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 3rem;
    }
    
    /* 分割线美化 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* 按钮美化 */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

# ============ 标题区域 ============
st.markdown("""
<div class="title-container">
    <h1>🧠 学科AI成熟度诊断工具</h1>
    <p>基于 Bass 扩散模型的学科人工智能融合程度智能诊断系统</p>
</div>
""", unsafe_allow_html=True)

# ============ 模型函数 ============
def bass_cumulative(t, p, q):
    """标准 Bass 模型（饱和水平归一化为 1）"""
    return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def bass_cumulative_with_M(t, p, q, M):
    """带饱和水平的 Bass 模型"""
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📤 数据上传")
    uploaded_file = st.file_uploader(
        "上传CSV文件",
        type=["csv"],
        help="文件需包含 year、ai_count、total_count 三列"
    )
    
    st.markdown("---")
    st.markdown("### 📋 数据格式要求")
    st.info("""
    **必需列：**
    - year：年份
    - ai_count：AI相关文献数
    - total_count：总文献数
    
    **示例：**
    """)

st.markdown("---")
st.markdown("### 💡 使用步骤")
st.caption("""
1️⃣ 准备学科年度数据  
2️⃣ 上传 CSV 文件  
3️⃣ 自动生成诊断报告  
4️⃣ 可下载分析结果
""")

# ============ 主内容区 ============
if uploaded_file is not None:
try:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ["year", "ai_count", "total_count"]):
        st.error("❌ CSV文件必须包含 'year'、'ai_count'、'total_count' 三列。")
        st.stop()

    df = df.sort_values("year").reset_index(drop=True)
    df["cum_ai"] = df["ai_count"].cumsum()
    df["cum_total"] = df["total_count"].cumsum()
    df["F_actual"] = df["cum_ai"] / df["cum_total"]

    if len(df) < 5:
        st.error("❌ 数据不足5年，Bass模型无法稳定拟合。")
        st.stop()

    # 数据概览卡片
    st.markdown("""
    <div class="custom-card">
        <h3>📊 数据概览</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{df['year'].min()}—{df['year'].max()}</div>
            <div class="label">数据年份范围</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{len(df)} 年</div>
            <div class="label">总年数</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{df['total_count'].sum():,}</div>
            <div class="label">总文献量</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{df['ai_count'].sum():,}</div>
            <div class="label">AI文献量</div>
        </div>
        """, unsafe_allow_html=True)

    # ============ 模型拟合 ============
    t = (df["year"] - df["year"].min()).values.astype(float)
    F_actual = df["F_actual"].values

    try:
        M_init = min(1.0, F_actual[-1] * 1.5)
        if M_init <= 0:
            M_init = 0.5

        popt, pcov = curve_fit(
            bass_cumulative_with_M, t, F_actual,
            p0=[0.001, 0.1, M_init],
            bounds=([0, 0, 0.01], [0.5, 1.5, 1.0]),
            maxfev=10000
        )
        p_fit, q_fit, M_fit = popt
        F_fitted = bass_cumulative_with_M(t, p_fit, q_fit, M_fit)

    except Exception:
        try:
            popt, pcov = curve_fit(
                bass_cumulative, t, F_actual,
                p0=[0.001, 0.1], maxfev=10000
            )
            p_fit, q_fit = popt
            M_fit = 1.0
            F_fitted = bass_cumulative(t, p_fit, q_fit)
        except Exception as e:
            st.error(f"❌ 模型拟合失败：{e}")
            st.info("请检查数据是否连续、是否包含足够年份（建议至少10年）。")
            st.stop()

    # 计算拟合优度 R²
    ss_res = np.sum((F_actual - F_fitted) ** 2)
    ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 计算拐点
    t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
    year_star = df["year"].min() + t_star

    # ============ 模型参数卡片 ============
    st.markdown("""
    <div class="custom-card">
        <h3>🔬 模型参数</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{p_fit:.6f}</div>
            <div class="label">创新系数 p</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{q_fit:.6f}</div>
            <div class="label">模仿系数 q</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{M_fit*100:.1f}%</div>
            <div class="label">饱和水平 M</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{int(round(year_star))}</div>
            <div class="label">拐点年份</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="value">{r2:.4f}</div>
            <div class="label">拟合优度 R²</div>
        </div>
        """, unsafe_allow_html=True)

    # ============ 诊断结果 ============
    st.markdown("""
    <div class="custom-card">
        <h3>🏥 诊断结果</h3>
    </div>
    """, unsafe_allow_html=True)
    
    F_current = df["F_actual"].iloc[-1]
    current_year = df["year"].max()
    dist_to_star = abs(current_year - year_star)

    recent_3_years = df.tail(3)
    growth_rate = recent_3_years["ai_count"].pct_change().mean()
    if pd.isna(growth_rate):
        growth_rate = 0

    # 综合判断阶段
    if F_current < 0.05:
        stage = "萌芽期"
        stage_class = "stage-萌芽期"
        desc = f"AI技术渗透率仅为{F_current*100:.1f}%，远低于10%的创新鸿沟阈值。目前仅有少数创新者在探索AI在该学科的应用，尚未形成规模效应。"
    elif F_current < 0.10 and current_year < year_star:
        stage = "萌芽后期"
        stage_class = "stage-萌芽后期"
        desc = f"渗透率{F_current*100:.1f}%，接近10%临界点。预计{int(year_star)}年前后将迎来快速增长拐点。"
    elif F_current >= 0.10 and dist_to_star <= 3:
        stage = "爆发期"
        stage_class = "stage-爆发期"
        desc = f"渗透率{F_current*100:.1f}%，已跨过10%临界点，且当前年份({current_year})处于理论拐点({int(year_star)})附近。AI扩散速率达到峰值，大量研究者涌入。"
    elif F_current >= 0.40 and dist_to_star > 3:
        stage = "成熟期"
        stage_class = "stage-成熟期"
        desc = f"渗透率{F_current*100:.1f}%，超过40%。AI已成为该学科主流研究方法之一，扩散速度明显放缓。"
    elif F_current >= 0.75:
        stage = "饱和期"
        stage_class = "stage-饱和期"
        desc = f"渗透率高达{F_current*100:.1f}%，接近饱和水平(M={M_fit*100:.1f}%)。AI在该学科的渗透空间有限。"
    else:
        stage = "发展期"
        stage_class = "stage-发展期"
        desc = f"渗透率{F_current*100:.1f}%，处于持续增长阶段。距离理论拐点还有{int(dist_to_star)}年。"

    # 扩散动力判定
    if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
        diffusion_type = "创新主导型"
        diffusion_desc = "学科内部自发的前沿探索是推动AI技术发展的主要动力。"
    elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
        diffusion_type = "模仿主导型"
        diffusion_desc = "学科对AI技术的采纳主要源于对同行或相关领域成功实践的模仿与学习。"
    else:
        diffusion_type = "均衡扩散型"
        diffusion_desc = "创新与模仿两种力量在扩散过程中作用相当。"

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem; background:#f8f9ff; border-radius:15px;'>
            <h4 style='color:#555; margin-bottom:1rem;'>当前发展阶段</h4>
            <div class="stage-badge {stage_class}">{stage}</div>
            <p style='margin-top:1rem; color:#666; font-size:0.95rem;'>{desc}</p>
            <p style='font-size:0.85rem; color:#999; margin-top:0.5rem;'>
                累计渗透率：{F_current*100:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem; background:#f8f9ff; border-radius:15px;'>
            <h4 style='color:#555; margin-bottom:1rem;'>扩散动力类型</h4>
            <div class="stage-badge" style="background:linear-gradient(135deg,#e3f2fd,#bbdefb);color:#1565c0;">
                {diffusion_type}
            </div>
            <p style='margin-top:1rem; color:#666; font-size:0.95rem;'>{diffusion_desc}</p>
            <p style='font-size:0.85rem; color:#999; margin-top:0.5rem;'>
                距拐点：{dist_to_star:.1f} 年 | 增长率：{growth_rate*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 诊断依据
    with st.expander("📋 查看详细诊断依据"):
        st.write(f"- 当前累计渗透率：{F_current*100:.2f}%")
        st.write(f"- 理论拐点年份：{int(year_star)}")
        st.write(f"- 距拐点距离：{dist_to_star:.1f} 年")
        st.write(f"- 近3年AI发文平均增长率：{growth_rate*100:.1f}%")
        st.write(f"- 估计饱和水平：{M_fit*100:.1f}%")
        st.write(f"- 拟合优度 R²：{r2:.4f}")

    # ============ 可视化分析 ============
    st.markdown("""
    <div class="custom-card">
        <h3>📈 可视化分析</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 发文量趋势", "📉 模型拟合", "🎯 渗透率变化"])
    
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["year"], y=df["ai_count"],
            mode="lines+markers", name="AI发文量",
            line=dict(color="#667eea", width=3),
            marker=dict(size=8, color="#667eea")
        ))
        fig1.update_layout(
            title="年度 AI 发文量趋势",
            xaxis_title="年份", yaxis_title="发文量",
            height=450, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_actual,
            mode="markers", name="实际值",
            marker=dict(color="#ff6b6b", size=10)
        ))
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_fitted,
            mode="lines", name="Bass拟合",
            line=dict(color="#667eea", width=3)
        ))
        fig2.update_layout(
            title="Bass 模型拟合效果",
            xaxis_title="年份", yaxis_title="累计渗透率 F(t)",
            height=450, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        df["annual_F"] = df["ai_count"] / df["total_count"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df["year"], y=df["annual_F"] * 100,
            name="年度渗透率",
            marker_color="rgba(102, 126, 234, 0.6)"
        ))
        fig3.add_trace(go.Scatter(
            x=df["year"], y=df["F_actual"] * 100,
            mode="lines+markers", name="累计渗透率",
            line=dict(color="#764ba2", width=3),
            marker=dict(size=8)
        ))
        fig3.update_layout(
            title="渗透率变化趋势",
            xaxis_title="年份", yaxis_title="渗透率 (%)",
            height=450, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ============ 未来预测 ============
    st.markdown("""
    <div class="custom-card">
        <h3>🔮 未来预测（2026—2050）</h3>
    </div>
    """, unsafe_allow_html=True)
    
    future_years = np.arange(df["year"].max() + 1, 2051)
    t_future = future_years - df["year"].min()
    F_future = bass_cumulative_with_M(t_future.astype(float), p_fit, q_fit, M_fit)

    pred_df = pd.DataFrame({
        "年份": future_years,
        "累计渗透率预测": [f"{v*100:.2f}%" for v in F_future]
    })
    
    st.dataframe(
        pred_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "年份": st.column_config.NumberColumn(format="%d"),
            "累计渗透率预测": "累计渗透率预测"
        }
    )

    # ============ 导出报告 ============
    st.markdown("""
    <div class="custom-card">
        <h3>📥 导出诊断报告</h3>
    </div>
    """, unsafe_allow_html=True)

    if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
        diffusion_type_str = "创新主导型"
    elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
        diffusion_type_str = "模仿主导型"
    else:
        diffusion_type_str = "均衡扩散型"

    report_lines = []
    report_lines.append("# 学科AI成熟度诊断报告")
    report_lines.append("")
    report_lines.append("## 基本信息")
    report_lines.append(f"- 数据范围：{df['year'].min()}—{df['year'].max()}（共{len(df)}年）")
    report_lines.append(f"- 总文献量：{df['total_count'].sum():,}")
    report_lines.append(f"- AI相关文献：{df['ai_count'].sum():,}")
    report_lines.append("")
    report_lines.append("## 模型参数")
    report_lines.append(f"- 创新系数 p：{p_fit:.6f}")
    report_lines.append(f"- 模仿系数 q：{q_fit:.6f}")
    report_lines.append(f"- 饱和水平 M：{M_fit*100:.2f}%")
    report_lines.append(f"- 拐点年份：{int(year_star)}")
    report_lines.append(f"- 拟合优度 R²：{r2:.4f}")
    report_lines.append("")
    report_lines.append("## 诊断结论")
    report_lines.append(f"- 当前阶段：{stage}")
    report_lines.append(f"- 当前累计渗透率：{F_current*100:.2f}%")
    report_lines.append(f"- 扩散动力类型：{diffusion_type_str}")
    report_lines.append("")
    report_lines.append("## 未来预测（2026—2035）")
    for yr, val in zip(future_years[:10], F_future[:10]):
        report_lines.append(f"- {int(yr)}年：{val*100:.2f}%")

    report_text = "\n".join(report_lines)
    report_bytes = report_text.encode("utf-8")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="📥 下载诊断报告",
            data=report_bytes,
            file_name=f"学科AI诊断报告_{df['year'].max()}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        st.info("报告为 Markdown 格式，可用记事本或 Typora 打开")

except Exception as e:
    st.error(f"❌ 文件读取失败：{e}")

else:
# 未上传文件时的引导界面
st.markdown("""
<div style='text-align:center; padding:4rem 2rem; background:white; border-radius:20px; box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
    <div style='font-size:5rem; margin-bottom:1rem;'>📤</div>
    <h2 style='color:#667eea; margin-bottom:1rem;'>欢迎使用学科AI成熟度诊断工具</h2>
    <p style='color:#666; font-size:1.1rem; max-width:600px; margin:0 auto 2rem auto;'>
        请从左侧上传包含学科年度发文量的 CSV 文件，系统将自动进行 Bass 模型拟合，
        诊断该学科 AI 研究的成熟度阶段。
    </p>
    <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;'>
        <div style='background:#f8f9ff; padding:1.5rem; border-radius:12px; min-width:150px;'>
            <div style='font-size:2rem;'>📊</div>
            <div style='color:#666; margin-top:0.5rem;'>Bass 模型拟合</div>
        </div>
        <div style='background:#f8f9ff; padding:1.5rem; border-radius:12px; min-width:150px;'>
            <div style='font-size:2rem;'>🏥</div>
            <div style='color:#666; margin-top:0.5rem;'>阶段诊断</div>
        </div>
        <div style='background:#f8f9ff; padding:1.5rem; border-radius:12px; min-width:150px;'>
            <div style='font-size:2rem;'>🔮</div>
            <div style='color:#666; margin-top:0.5rem;'>未来预测</div>
        </div>
        <div style='background:#f8f9ff; padding:1.5rem; border-radius:12px; min-width:150px;'>
            <div style='font-size:2rem;'>📥</div>
            <div style='color:#666; margin-top:0.5rem;'>报告导出</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#f8f9ff; padding:1.5rem; border-radius:15px; margin-top:2rem;'>
    <h4 style='color:#555; margin-bottom:1rem;'>📄 CSV 文件格式示例</h4>
</div>
""", unsafe_allow_html=True)
st.code("year,ai_count,total_count\n2000,331,50000\n2001,386,52000\n2002,380,54000", language="csv")

# ============ 页脚 ============
st.markdown("""
<div class="footer">
<p>🧠 学科AI成熟度诊断工具 | 基于 Bass 扩散模型</p>
<p>© 2026 科研数据分析工具</p>
</div>
""", unsafe_allow_html=True)