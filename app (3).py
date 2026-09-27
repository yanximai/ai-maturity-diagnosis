import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# ============ 页面配置 ============
st.set_page_config(page_title="学科AI成熟度诊断工具", layout="wide")

# ============ 只加美化样式，不动你的代码 ============
st.markdown("""
<style>
    /* 全局背景 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 标题美化 */
    .stTitle {
        color: #333 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 卡片容器 - 包裹每个模块 */
    div[data-testid="stBlock"] {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    div[data-testid="stBlock"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* metric 数值美化 */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9ff, #eef0ff);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    div[data-testid="metric-container"] label {
        color: #888 !important;
        font-size: 0.85rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #667eea !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* subheader 美化 */
    .stSubheader {
        color: #444 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        display: inline-block;
        margin-bottom: 1rem !important;
    }
    
    /* 成功消息美化 */
    .stAlert {
        border-radius: 12px !important;
        border-left: 5px solid #667eea !important;
    }
    
    /* info 消息美化 */
    .stInfo {
        border-radius: 12px !important;
        background: #f8f9ff !important;
    }
    
    /* error 消息美化 */
    .stError {
        border-radius: 12px !important;
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
    
    /* 下载按钮美化 */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 文件上传器美化 */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8f9ff, #eef0ff);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* 表格美化 */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e8e8e8;
    }
    
    /* expander 美化 */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e8e8e8;
        background: white;
    }
    
    /* tabs 美化 */
    div[data-testid="stTabs"] button {
        border-radius: 10px 10px 0 0 !important;
        font-weight: 600;
    }
    
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* code 区域美化 */
    .stCode {
        border-radius: 12px !important;
        border: 1px solid #e8e8e8 !important;
    }
    
    /* 分隔线美化 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
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
</style>
""", unsafe_allow_html=True)


st.title("学科AI成熟度诊断工具")
st.markdown("上传学科年度发文量CSV文件，自动诊断该学科AI研究的成熟度阶段。")


def bass_cumulative(t, p, q):
    """标准 Bass 模型（饱和水平归一化为 1）"""
    return (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def bass_cumulative_with_M(t, p, q, M):
    """带饱和水平的 Bass 模型"""
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


uploaded_file = st.file_uploader("上传CSV文件（需包含 year、ai_count、total_count 三列）", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in ["year", "ai_count", "total_count"]):
            st.error("CSV文件必须包含 'year'、'ai_count'、'total_count' 三列。")
            st.stop()

        df = df.sort_values("year").reset_index(drop=True)
        df["cum_ai"] = df["ai_count"].cumsum()
        df["cum_total"] = df["total_count"].cumsum()
        df["F_actual"] = df["cum_ai"] / df["cum_total"]

        if len(df) < 5:
            st.error("数据不足5年，Bass模型无法稳定拟合。")
            st.stop()

        st.success(f"数据加载成功：共 {len(df)} 年（{df['year'].min()}—{df['year'].max()}）")

        # ============ 模型拟合 ============
        t = (df["year"] - df["year"].min()).values.astype(float)
        F_actual = df["F_actual"].values

        try:
            # 先尝试带饱和水平 M 的拟合
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
            # 带 M 拟合失败时，回退到标准 Bass 模型
            try:
                popt, pcov = curve_fit(
                    bass_cumulative, t, F_actual,
                    p0=[0.001, 0.1], maxfev=10000
                )
                p_fit, q_fit = popt
                M_fit = 1.0
                F_fitted = bass_cumulative(t, p_fit, q_fit)
            except Exception as e:
                st.error(f"模型拟合失败：{e}")
                st.info("请检查数据是否连续、是否包含足够年份（建议至少10年）。")
                st.stop()

        # 计算拟合优度 R²
        ss_res = np.sum((F_actual - F_fitted) ** 2)
        ss_tot = np.sum((F_actual - np.mean(F_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 计算拐点（M 不影响拐点位置）
        t_star = np.log(q_fit / p_fit) / (p_fit + q_fit)
        year_star = df["year"].min() + t_star

        # ============ 参数展示 ============
        st.subheader("模型参数")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("创新系数 p", f"{p_fit:.6f}")
        col2.metric("模仿系数 q", f"{q_fit:.6f}")
        col3.metric("饱和水平 M", f"{M_fit*100:.2f}%")
        col4.metric("拐点年份", f"{int(round(year_star))}")
        col5.metric("拟合优度 R²", f"{r2:.4f}")

        # ============ 扩散动力判定 ============
        st.subheader("扩散动力类型")
        if p_fit > q_fit and (p_fit - q_fit) >= 0.02:
            st.info("**创新主导型**：学科内部自发的前沿探索是推动AI技术发展的主要动力。")
        elif q_fit > p_fit and (q_fit - p_fit) >= 0.02:
            st.warning("**模仿主导型**：学科对AI技术的采纳主要源于对同行或相关领域成功实践的模仿与学习。")
        else:
            st.info("**均衡扩散型**：创新与模仿两种力量在扩散过程中作用相当。")

        # ============ 阶段判断 ============
        st.subheader("当前发展阶段")
        F_current = df["F_actual"].iloc[-1]
        current_year = df["year"].max()
        dist_to_star = abs(current_year - year_star)

        # 计算近3年平均增长率
        recent_3_years = df.tail(3)
        growth_rate = recent_3_years["ai_count"].pct_change().mean()
        if pd.isna(growth_rate):
            growth_rate = 0

        # 综合判断阶段
        if F_current < 0.05:
            stage = "萌芽期"
            desc = f"AI技术渗透率仅为{F_current*100:.1f}%，远低于10%的创新鸿沟阈值。目前仅有少数创新者在探索AI在该学科的应用，尚未形成规模效应。"
        elif F_current < 0.10 and current_year < year_star:
            stage = "萌芽后期"
            desc = f"渗透率{F_current*100:.1f}%，接近10%临界点。预计{int(year_star)}年前后将迎来快速增长拐点。"
        elif F_current >= 0.10 and dist_to_star <= 3:
            stage = "爆发期"
            desc = f"渗透率{F_current*100:.1f}%，已跨过10%临界点，且当前年份({current_year})处于理论拐点({int(year_star)})附近。AI扩散速率达到峰值，大量研究者涌入。"
        elif F_current >= 0.40 and dist_to_star > 3:
            stage = "成熟期"
            desc = f"渗透率{F_current*100:.1f}%，超过40%。AI已成为该学科主流研究方法之一，扩散速度明显放缓。研究重点可能从'是否采用AI'转向'如何更好地应用AI'。"
        elif F_current >= 0.75:
            stage = "饱和期"
            desc = f"渗透率高达{F_current*100:.1f}%，接近饱和水平(M={M_fit*100:.1f}%)。AI在该学科的渗透空间有限，新的增长点可能需要突破性技术创新。"
        else:
            stage = "发展期"
            desc = f"渗透率{F_current*100:.1f}%，处于持续增长阶段。距离理论拐点还有{int(dist_to_star)}年。"

        st.markdown(f"### 📊 当前阶段：**{stage}**")
        st.markdown(desc)

        # 诊断依据
        with st.expander("查看诊断依据"):
            st.write(f"- 当前累计渗透率：{F_current*100:.2f}%")
            st.write(f"- 理论拐点年份：{int(year_star)}")
            st.write(f"- 距拐点距离：{dist_to_star:.1f} 年")
            st.write(f"- 近3年AI发文平均增长率：{growth_rate*100:.1f}%")
            st.write(f"- 估计饱和水平：{M_fit*100:.1f}%")
            st.write(f"- 拟合优度 R²：{r2:.4f}")

        # ============ 历史趋势图 ============
        st.subheader("历史发文量趋势")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["year"], y=df["ai_count"],
                                   mode="lines+markers", name="AI发文量",
                                   line=dict(color="#1f77b4", width=2)))
        fig1.update_layout(xaxis_title="年份", yaxis_title="发文量", height=400)
        st.plotly_chart(fig1, use_container_width=True)

        # ============ 模型拟合图 ============
        st.subheader("Bass模型拟合效果")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["year"], y=F_actual,
                                   mode="markers", name="实际累计渗透率",
                                   marker=dict(color="red", size=8)))
        fig2.add_trace(go.Scatter(x=df["year"], y=F_fitted,
                                   mode="lines", name="Bass拟合曲线",
                                   line=dict(color="blue", width=2)))
        fig2.update_layout(xaxis_title="年份", yaxis_title="累计渗透率 F(t)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # ============ 渗透率变化趋势 ============
        st.subheader("渗透率变化趋势")
        df["annual_F"] = df["ai_count"] / df["total_count"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df["year"], y=df["annual_F"] * 100,
                               name="年度渗透率",
                               marker_color="lightblue"))
        fig3.add_trace(go.Scatter(x=df["year"], y=df["F_actual"] * 100,
                                   mode="lines+markers",
                                   name="累计渗透率",
                                   line=dict(color="darkblue", width=2),
                                   marker=dict(size=8)))
        fig3.update_layout(
            xaxis_title="年份",
            yaxis_title="渗透率 (%)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ============ 未来预测（25年） ============
        st.subheader("未来25年渗透率预测（2026—2050）")
        future_years = np.arange(df["year"].max() + 1, 2051)
        t_future = future_years - df["year"].min()
        F_future = bass_cumulative_with_M(t_future.astype(float), p_fit, q_fit, M_fit)

        pred_df = pd.DataFrame({
            "年份": future_years,
            "累计渗透率预测": [f"{v*100:.2f}%" for v in F_future]
        })
        st.dataframe(pred_df, use_container_width=True)

        # ============ 导出诊断报告 ============
        st.subheader("📥 导出诊断报告")

        # 判断扩散动力类型
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
        st.download_button(
            label="⬇️ 下载报告 (.md)",
            data=report_bytes,
            file_name=f"学科AI诊断报告_{df['year'].max()}.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"文件读取失败：{e}")

else:
    st.info("请上传CSV文件开始分析。文件格式示例：")
    st.code("year,ai_count,total_count\n2000,331,50000\n2001,386,52000\n2002,380,54000", language="csv")