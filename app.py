import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import base64

# ============ 页面配置 ============
st.set_page_config(
    page_title="AI4S 学科熟化度诊断平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义 CSS 美化 ============
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 标题区域 */
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .title-container p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
    }
    
    /* 卡片样式 */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border: 1px solid #e8e8e8;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* 指标数值 */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* 阶段标签 */
    .stage-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .stage-萌芽期 { background: #e8f5e9; color: #2e7d32; }
    .stage-萌芽后期 { background: #fff3e0; color: #e65100; }
    .stage-爆发期 { background: #ffebee; color: #c62828; }
    .stage-发展期 { background: #e3f2fd; color: #1565c0; }
    .stage-成熟期 { background: #f3e5f5; color: #6a1b9a; }
    .stage-饱和期 { background: #eceff1; color: #37474f; }
    
    /* 上传区域 */
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin-bottom: 2rem;
    }
    
    /* 页脚 */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #999;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============ 侧边栏 ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## 🧠 AI4S 诊断平台")
    st.markdown("---")
    
    st.markdown("### 📤 数据上传")
    uploaded_file = st.file_uploader(
        "上传 CSV 文件",
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
st.markdown("### 💡 使用说明")
st.caption("""
1. 准备学科年度数据
2. 上传 CSV 文件
3. 自动生成诊断报告
4. 可下载分析结果
""")

# ============ 主内容区 ============
# 标题
st.markdown("""
<div class="title-container">
<h1>🧠 学科 AI 熟化度诊断平台</h1>
<p>基于 Bass 扩散模型的学科人工智能融合程度智能诊断系统</p>
</div>
""", unsafe_allow_html=True)

# 模型函数
def bass_cumulative(t, p, q):
return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def bass_cumulative_with_M(t, p, q, M):
return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

# ============ 数据处理 ============
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
    <div class="card">
        <h3>📊 数据概览</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("数据年份范围", f"{df['year'].min()}—{df['year'].max()}")
    col2.metric("总年数", f"{len(df)} 年")
    col3.metric("总文献量", f"{df['total_count'].sum():,}")
    col4.metric("AI文献量", f"{df['ai_count'].sum():,}")
    
    # 模型拟合
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
            st.stop()
    
    # 计算指标
    ss_res = np.sum((F_actual - F_fitted) ** 2)
    ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
    year_star = df["year"].min() + t_star
    
    # ============ 模型参数卡片 ============
    st.markdown("""
    <div class="card">
        <h3>🔬 模型参数</h3>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <div class='metric-label'>创新系数 p</div>
            <div class='metric-value'>{p_fit:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <div class='metric-label'>模仿系数 q</div>
            <div class='metric-value'>{q_fit:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <div class='metric-label'>饱和水平 M</div>
            <div class='metric-value'>{M_fit*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <div class='metric-label'>拐点年份</div>
            <div class='metric-value'>{int(round(year_star))}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[4]:
        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <div class='metric-label'>拟合优度 R²</div>
            <div class='metric-value'>{r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ 诊断结果 ============
    st.markdown("""
    <div class="card">
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
    
    # 阶段判断
    if F_current < 0.05:
        stage = "萌芽期"
        stage_class = "stage-萌芽期"
        desc = "AI技术渗透率极低，处于初始探索阶段"
    elif F_current < 0.10 and current_year < year_star:
        stage = "萌芽后期"
        stage_class = "stage-萌芽后期"
        desc = "接近创新鸿沟阈值，即将迎来快速增长"
    elif F_current >= 0.10 and dist_to_star <= 3:
        stage = "爆发期"
        stage_class = "stage-爆发期"
        desc = "已跨过临界点，扩散速率达到峰值"
    elif F_current >= 0.40 and dist_to_star > 3:
        stage = "成熟期"
        stage_class = "stage-成熟期"
        desc = "AI成为主流研究方法，增速放缓"
    elif F_current >= 0.75:
        stage = "饱和期"
        stage_class = "stage-饱和期"
        desc = "接近饱和水平，渗透空间有限"
    else:
        stage = "发展期"
        stage_class = "stage-发展期"
        desc = "持续增长阶段"
    
    # 扩散动力
    if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
        diffusion_type = "创新主导型"
        diffusion_desc = "学科内部前沿探索驱动"
    elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
        diffusion_type = "模仿主导型"
        diffusion_desc = "同行模仿与网络效应驱动"
    else:
        diffusion_type = "均衡扩散型"
        diffusion_desc = "创新与模仿共同作用"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='text-align:center; padding:2rem;'>
            <h4>当前发展阶段</h4>
            <div class='stage-badge {stage_class}'>{stage}</div>
            <p style='margin-top:1rem; color:#666;'>{desc}</p>
            <p style='font-size:0.9rem; color:#999;'>
                累计渗透率：{F_current*100:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align:center; padding:2rem;'>
            <h4>扩散动力类型</h4>
            <div class='stage-badge' style='background:#e3f2fd; color:#1565c0;'>
                {diffusion_type}
            </div>
            <p style='margin-top:1rem; color:#666;'>{diffusion_desc}</p>
            <p style='font-size:0.9rem; color:#999;'>
                距拐点：{dist_to_star:.1f} 年 | 增长率：{growth_rate*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ 图表区域 ============
    st.markdown("""
    <div class="card">
        <h3>📈 可视化分析</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 发文量趋势", "📉 模型拟合", "🎯 渗透率变化"])
    
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["year"], y=df["ai_count"],
            mode="lines+markers",
            name="AI发文量",
            line=dict(color="#667eea", width=3),
            marker=dict(size=8, color="#667eea")
        ))
        fig1.update_layout(
            title="年度 AI 发文量趋势",
            xaxis_title="年份",
            yaxis_title="发文量",
            height=450,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_actual,
            mode="markers",
            name="实际值",
            marker=dict(color="#ff6b6b", size=10)
        ))
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_fitted,
            mode="lines",
            name="Bass拟合",
            line=dict(color="#667eea", width=3)
        ))
        fig2.update_layout(
            title="Bass 模型拟合效果",
            xaxis_title="年份",
            yaxis_title="累计渗透率 F(t)",
            height=450,
            hovermode="x unified",
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
            mode="lines+markers",
            name="累计渗透率",
            line=dict(color="#764ba2", width=3),
            marker=dict(size=8)
        ))
        fig3.update_layout(
            title="渗透率变化趋势",
            xaxis_title="年份",
            yaxis_title="渗透率 (%)",
            height=450,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # ============ 未来预测 ============
    st.markdown("""
    <div class="card">
        <h3>🔮 未来预测</h3>
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
    <div class="card">
        <h3>📥 导出报告</h3>
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
    st.error(f"❌ 处理出错：{e}")

else:
# 未上传文件时的引导界面
st.markdown("""
<div style='text-align:center; padding:4rem 2rem;'>
    <div style='font-size:4rem; margin-bottom:1rem;'>📤</div>
    <h2 style='color:#667eea;'>上传您的学科数据</h2>
    <p style='color:#666; font-size:1.1rem; max-width:600px; margin:1rem auto;'>
        请从左侧上传包含学科年度发文量的 CSV 文件，系统将自动进行 Bass 模型拟合，
        诊断该学科 AI 研究的成熟度阶段。
    </p>
    <div style='background:#f8f9ff; border-radius:12px; padding:2rem; max-width:500px; margin:2rem auto;'>
        <h4 style='color:#333;'>支持的功能</h4>
        <ul style='text-align:left; color:#666; line-height:2;'>
            <li>✅ Bass 扩散模型拟合</li>
            <li>✅ 学科熟化度阶段诊断</li>
            <li>✅ 扩散动力类型识别</li>
            <li>✅ 未来渗透率预测</li>
            <li>✅ 诊断报告导出</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ 页脚 ============
st.markdown("""
<div class="footer">
<p>🧠 AI4S 学科熟化度诊断平台 | 基于 Bass 扩散模型</p>
<p>© 2026 科研数据分析工具</p>
</div>
""", unsafe_allow_html=True)