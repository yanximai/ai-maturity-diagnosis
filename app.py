import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(page_title="学科AI成熟度诊断工具", layout="wide", page_icon="🔬")

# ============ 全局样式 ============
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.05rem; }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-card p { color: #888; font-size: 0.85rem; margin: 0; }
    .metric-card h3 { color: #667eea; margin: 0.3rem 0 0 0; font-size: 1.4rem; }
    .stage-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .stage-mengya { background: #e3f2fd; color: #1976d2; }
    .stage-baofa { background: #fff3e0; color: #f57c00; }
    .stage-chengshu { background: #e8f5e9; color: #388e3c; }
    .stage-baohe { background: #fce4ec; color: #c2185b; }
    .section-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============ 头图 ============
st.markdown("""
<div class="main-header">
    <h1>🔬 学科AI成熟度诊断工具</h1>
    <p>基于 Bass 扩散模型的学科人工智能融合程度智能诊断系统</p>
</div>
""", unsafe_allow_html=True)

# ============ 侧边栏 ============
with st.sidebar:
    st.title("📖 使用说明")
    st.markdown("""
    **操作步骤：**
    1. 准备CSV文件（三列：year、ai_count、total_count）
    2. 上传文件
    3. 查看诊断结果
    4. 下载诊断报告
    
    **数据要求：**
    - `year`：年份（整数）
    - `ai_count`：当年AI相关论文数
    - `total_count`：当年学科全部论文数
    - 至少5年连续数据
    """)
    st.markdown("---")
    st.caption("北京师范大学珠海校区 · 大创项目")


# ============ Bass模型 ============
def bass_cumulative(t, p, q):
    return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def bass_cumulative_with_M(t, p, q, M):
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


# ============ 文件上传 ============
uploaded_file = st.file_uploader("📤 上传CSV文件（需包含 year、ai_count、total_count 三列）", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in ["year", "ai_count", "total_count"]):
            st.error("❌ CSV文件必须包含 'year'、'ai_count'、'total_count' 三列。")
            st.stop()

        # 剔除2026及以后（数据不完整）
        df = df[df["year"] <= 2025].sort_values("year").reset_index(drop=True)
        df["cum_ai"] = df["ai_count"].cumsum()
        df["cum_total"] = df["total_count"].cumsum()
        df["F_actual"] = df["cum_ai"] / df["cum_total"]

        if len(df) < 5:
            st.error("❌ 数据不足5年，Bass模型无法稳定拟合。")
            st.stop()

        st.success(f"✅ 数据加载成功：共 {len(df)} 年（{df['year'].min()}—{df['year'].max()}）")

        # ============ 数据概览 ============
        st.markdown('<div class="section-title">📊 数据概览</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <p>数据年份范围</p>
                <h3>{df['year'].min()}—{df['year'].max()}</h3>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <p>总年数</p>
                <h3>{len(df)} 年</h3>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <p>总文献量</p>
                <h3>{df['total_count'].sum():,}</h3>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <p>AI相关文献</p>
                <h3>{df['ai_count'].sum():,}</h3>
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
                st.stop()

        ss_res = np.sum((F_actual - F_fitted) ** 2)
        ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
        year_star = df["year"].min() + t_star

        # ============ 模型参数 ============
        st.markdown('<div class="section-title">🔢 模型参数</div>', unsafe_allow_html=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            st.markdown(f"""
            <div class="metric-card">
                <p>创新系数 p</p>
                <h3>{p_fit:.6f}</h3>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="metric-card">
                <p>模仿系数 q</p>
                <h3>{q_fit:.6f}</h3>
            </div>
            """, unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="metric-card">
                <p>饱和水平 M</p>
                <h3>{M_fit*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        with p4:
            st.markdown(f"""
            <div class="metric-card">
                <p>拐点年份</p>
                <h3>{int(round(year_star))}</h3>
            </div>
            """, unsafe_allow_html=True)
        with p5:
            st.markdown(f"""
            <div class="metric-card">
                <p>拟合优度 R²</p>
                <h3>{r2:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)

        # ============ 扩散动力判定 ============
        st.markdown('<div class="section-title">⚡ 扩散动力类型</div>', unsafe_allow_html=True)
        if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
            diffusion_type_str = "创新主导型"
            st.info("🧬 **创新主导型**：学科内部自发的前沿探索是推动AI技术发展的主要动力。")
        elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
            diffusion_type_str = "模仿主导型"
            st.warning("🔁 **模仿主导型**：学科对AI技术的采纳主要源于对同行或相关领域成功实践的模仿与学习。")
        else:
            diffusion_type_str = "均衡扩散型"
            st.info("⚖️ **均衡扩散型**：创新与模仿两种力量在扩散过程中作用相当。")

        # ============ 阶段判断 ============
        st.markdown('<div class="section-title">🎯 当前发展阶段</div>', unsafe_allow_html=True)
        F_current = df["F_actual"].iloc[-1]
        current_year = df["year"].max()
        dist_to_star = abs(current_year - year_star)

        recent_3_years = df.tail(3)
        growth_rate = recent_3_years["ai_count"].pct_change().mean()
        if pd.isna(growth_rate):
            growth_rate = 0

        if F_current < 0.05:
            stage = "萌芽期"
            stage_class = "stage-mengya"
            desc = f"AI技术渗透率仅为{F_current*100:.1f}%，远低于10%的创新鸿沟阈值。目前仅有少数创新者在探索AI在该学科的应用，尚未形成规模效应。"
        elif F_current < 0.10 and current_year < year_star:
            stage = "萌芽后期"
            stage_class = "stage-mengya"
            desc = f"渗透率{F_current*100:.1f}%，接近10%临界点。预计{int(year_star)}年前后将迎来快速增长拐点。"
        elif F_current >= 0.10 and dist_to_star <= 3:
            stage = "爆发期"
            stage_class = "stage-baofa"
            desc = f"渗透率{F_current*100:.1f}%，已跨过10%临界点，且当前年份（{current_year}）处于理论拐点（{int(year_star)}）附近。AI扩散速率达到峰值。"
        elif F_current >= 0.40 and dist_to_star > 3:
            stage = "成熟期"
            stage_class = "stage-chengshu"
            desc = f"渗透率{F_current*100:.1f}%，超过40%。AI已成为该学科主流研究方法之一，扩散速度明显放缓。"
        elif F_current >= 0.75:
            stage = "饱和期"
            stage_class = "stage-baohe"
            desc = f"渗透率高达{F_current*100:.1f}%，接近饱和水平（M={M_fit*100:.1f}%）。"
        else:
            stage = "发展期"
            stage_class = "stage-baofa"
            desc = f"渗透率{F_current*100:.1f}%，处于持续增长阶段。距离理论拐点还有{int(dist_to_star)}年。"

        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <span class="stage-badge {stage_class}">📊 {stage}</span>
            <p style="margin-top: 1.2rem; color: #555; font-size: 1.05rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # 诊断依据
        with st.expander("🔍 查看诊断依据"):
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("当前累计渗透率", f"{F_current*100:.2f}%")
            d2.metric("理论拐点年份", f"{int(year_star)}")
            d3.metric("距拐点距离", f"{dist_to_star:.1f} 年")
            d4.metric("近3年AI平均增长率", f"{growth_rate*100:.1f}%")
            d5.metric("估计饱和水平", f"{M_fit*100:.1f}%")

        # ============ 可视化图表 ============
        st.markdown('<div class="section-title">📈 可视化分析</div>', unsafe_allow_html=True)

        # 图表1：历史发文量趋势
        st.subheader("历史发文量趋势")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["year"], y=df["ai_count"],
            mode="lines+markers", name="AI发文量",
            line=dict(color="#667eea", width=2.5),
            marker=dict(size=6)
        ))
        fig1.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, Arial", size=12),
            xaxis_title="年份", yaxis_title="发文量",
            height=400,
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 图表2：Bass模型拟合效果
        st.subheader("Bass模型拟合效果")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_actual,
            mode="markers", name="实际累计渗透率",
            marker=dict(color="#e74c3c", size=8)
        ))
        fig2.add_trace(go.Scatter(
            x=df["year"], y=F_fitted,
            mode="lines", name="Bass拟合曲线",
            line=dict(color="#3498db", width=2.5)
        ))
        fig2.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, Arial", size=12),
            xaxis_title="年份", yaxis_title="累计渗透率 F(t)",
            height=400,
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 图表3：渗透率变化趋势
        st.subheader("渗透率变化趋势")
        df["annual_F"] = df["ai_count"] / df["total_count"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df["year"], y=df["annual_F"] * 100,
            name="年度渗透率",
            marker_color="#a29bfe"
        ))
        fig3.add_trace(go.Scatter(
            x=df["year"], y=df["F_actual"] * 100,
            mode="lines+markers", name="累计渗透率",
            line=dict(color="#6c5ce7", width=2.5),
            marker=dict(size=7)
        ))
        fig3.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, Arial", size=12),
            xaxis_title="年份", yaxis_title="渗透率 (%)",
            height=400,
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ============ 未来预测 ============
        st.markdown('<div class="section-title">🔮 未来25年渗透率预测（2026—2050）</div>', unsafe_allow_html=True)
        future_years = np.arange(df["year"].max() + 1, 2051)
        t_future = future_years - df["year"].min()
        F_future = bass_cumulative_with_M(t_future.astype(float), p_fit, q_fit, M_fit)

        pred_df = pd.DataFrame({
            "年份": future_years,
            "累计渗透率预测": [f"{v*100:.2f}%" for v in F_future]
        })
        st.dataframe(pred_df, use_container_width=True, height=300)

        # 预测曲线图
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["year"], y=df["F_actual"] * 100,
            mode="lines", name="历史实际值",
            line=dict(color="#667eea", width=2)
        ))
        fig4.add_trace(go.Scatter(
            x=future_years, y=F_future * 100,
            mode="lines", name="未来预测值",
            line=dict(color="#e74c3c", width=2, dash="dash")
        ))
        fig4.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, Arial", size=12),
            xaxis_title="年份", yaxis_title="累计渗透率 (%)",
            height=400,
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ============ 导出报告 ============
        st.markdown('<div class="section-title">📥 导出诊断报告</div>', unsafe_allow_html=True)

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
        st.download_button(
            label="⬇️ 下载完整诊断报告（.md）",
            data=report_bytes,
            file_name=f"学科AI诊断报告_{df['year'].max()}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )

    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
        <h2 style="color: #667eea;">👋 欢迎使用学科AI成熟度诊断工具</h2>
        <p style="color: #666; font-size: 1.1rem;">上传学科年度发文量CSV文件，自动诊断该学科AI研究的成熟度阶段</p>
        <div style="margin-top: 2rem; text-align: left; display: inline-block; background: white; padding: 1.5rem; border-radius: 8px;">
            <p style="font-weight: bold; color: #333;">📋 CSV格式要求：</p>
            <ul style="color: #666;">
                <li>必须包含三列：<code>year</code>、<code>ai_count</code>、<code>total_count</code></li>
                <li><code>year</code>：年份（整数）</li>
                <li><code>ai_count</code>：该年AI相关论文数（整数）</li>
                <li><code>total_count</code>：该年学科全部论文数（整数）</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.code("year,ai_count,total_count\n2000,331,50000\n2001,386,52000\n2002,380,54000", language="csv")
