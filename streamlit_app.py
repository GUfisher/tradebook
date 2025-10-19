import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import MinMaxScaler

# 页面配置
st.set_page_config(
    page_title="交易信息汇总看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 数据加载函数
@st.cache_data
def load_data():
    """加载并预处理数据"""
    df = pd.read_excel('交易信息汇总.xlsx')
    df['交易日期'] = pd.to_datetime(df['交易日期']).dt.strftime('%Y-%m-%d')
    df['观察日'] = pd.to_datetime(df['观察日'], errors='coerce')
    df['起始日期'] = pd.to_datetime(df['起始日期'], errors='coerce')
    df['到期日'] = pd.to_datetime(df['到期日'], errors='coerce')
    return df

def expand_observation_dates(df):
    """展开雪球敲出观察日序列"""
    snowball_data = df[df['雪球敲出观察日序列'].notna()].copy()

    unique_trades = snowball_data.groupby('Trade Id').agg({
        '标的名称': 'first',
        '交易日期': 'first',
        '起始日期': 'first',
        '到期日': 'first',
        'TRADE_KEYWORD.期权特殊类型': 'first',
        '雪球敲出观察日序列': 'first'
    }).reset_index()

    expanded_data = []

    for idx, row in unique_trades.iterrows():
        trade_id = row['Trade Id']
        underlying = row['标的名称']
        option_type = row['TRADE_KEYWORD.期权特殊类型']
        start_date = row['起始日期']
        maturity = row['到期日']

        obs_dates_str = str(row['雪球敲出观察日序列']).split(';')

        for i, date_str in enumerate(obs_dates_str, 1):
            date_obj = datetime.datetime.strptime(date_str.strip(), '%Y%m%d')

            expanded_data.append({
                'Trade Id': trade_id,
                '标的名称': underlying,
                '期权类型': option_type,
                '起始日期': start_date,
                '到期日': maturity,
                '观察日序号': i,
                '观察日': date_obj,
                '观察日期': date_obj.strftime('%Y-%m-%d'),
                '星期': date_obj.strftime('%A'),
                '总观察日数': len(obs_dates_str)
            })

    obs_df = pd.DataFrame(expanded_data)
    return obs_df

def create_enhanced_calendar_heatmap(obs_df):
    """创建带Trade ID标注的日历热力图"""
    # 按日期统计，收集所有Trade ID
    daily_stats = obs_df.groupby('观察日期').agg({
        'Trade Id': lambda x: list(x),
        '标的名称': lambda x: list(x),
        '观察日序号': lambda x: list(x)
    }).reset_index()

    daily_stats['观察次数'] = daily_stats['Trade Id'].apply(len)
    daily_stats['观察日'] = pd.to_datetime(daily_stats['观察日期'])

    # 获取日期范围
    min_date = obs_df['观察日'].min()
    max_date = obs_df['观察日'].max()

    # 生成所有日期
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

    # 创建完整的日期数据框
    full_df = pd.DataFrame({'观察日': all_dates})
    full_df['日期'] = full_df['观察日'].dt.strftime('%Y-%m-%d')
    full_df = full_df.merge(daily_stats[['观察日期', '观察次数', 'Trade Id', '标的名称', '观察日序号']],
                           left_on='日期', right_on='观察日期', how='left')
    full_df['观察次数'] = full_df['观察次数'].fillna(0)

    # 添加日历信息
    full_df['年'] = full_df['观察日'].dt.year
    full_df['月'] = full_df['观察日'].dt.month
    full_df['日'] = full_df['观察日'].dt.day
    full_df['星期'] = full_df['观察日'].dt.dayofweek
    full_df['星期名'] = full_df['观察日'].dt.strftime('%A')
    full_df['年月'] = full_df['观察日'].dt.strftime('%Y-%m')

    # 计算周数
    def get_week_of_month(date):
        first_day = date.replace(day=1)
        adjusted_dom = date.day + first_day.weekday()
        return int(np.ceil(adjusted_dom / 7.0)) - 1

    full_df['月内周数'] = full_df['观察日'].apply(get_week_of_month)

    # 创建悬停文本
    def create_hover_text(row):
        if row['观察次数'] > 0:
            text = f"<b>📅 {row['日期']} ({row['星期名']})</b><br>"
            text += f"<b>观察次数: {int(row['观察次数'])}</b><br><br>"

            trades = row['Trade Id']
            underlyings = row['标的名称']
            sequences = row['观察日序号']

            underlying_groups = {}
            for i, (trade, underlying, seq) in enumerate(zip(trades, underlyings, sequences)):
                if underlying not in underlying_groups:
                    underlying_groups[underlying] = []
                underlying_groups[underlying].append((trade, seq))

            for underlying, trade_list in sorted(underlying_groups.items()):
                text += f"<b>📊 {underlying}</b><br>"
                for trade, seq in trade_list:
                    text += f"  • Trade {trade} (观察#{seq})<br>"
                text += "<br>"

            return text.rstrip('<br>')
        else:
            return f"{row['日期']}<br>无观察"

    full_df['hover_text'] = full_df.apply(create_hover_text, axis=1)

    # 按年月分组创建子图
    year_months = sorted(full_df['年月'].unique())

    # 计算行列数
    n_months = len(year_months)
    cols = 3
    rows = int(np.ceil(n_months / cols))

    # 创建子图
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"<b>{ym}</b>" for ym in year_months],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)]
    )

    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for idx, ym in enumerate(year_months):
        row = idx // cols + 1
        col = idx % cols + 1

        month_df = full_df[full_df['年月'] == ym].copy()

        # 创建热力图矩阵
        heatmap_matrix = np.full((6, 7), np.nan)
        hover_matrix = [['' for _ in range(7)] for _ in range(6)]
        text_matrix = [['' for _ in range(7)] for _ in range(6)]

        for _, row_data in month_df.iterrows():
            week = row_data['月内周数']
            day = row_data['星期']
            if week < 6:
                heatmap_matrix[week][day] = row_data['观察次数']
                hover_matrix[week][day] = row_data['hover_text']

                day_num = row_data['日']
                obs_count = int(row_data['观察次数'])
                if obs_count > 0:
                    if obs_count == 1:
                        text_matrix[week][day] = f"<b>{day_num}</b>"
                    else:
                        text_matrix[week][day] = f"<b>{day_num}</b><br>({obs_count})"
                else:
                    text_matrix[week][day] = str(day_num)

        # 添加热力图
        heatmap = go.Heatmap(
            z=heatmap_matrix,
            x=weekday_labels,
            y=list(range(6)),
            text=text_matrix,
            hovertext=hover_matrix,
            hoverinfo='text',
            texttemplate='%{text}',
            textfont={"size": 9},
            colorscale=[
                [0, '#F8F9FA'],
                [0.2, '#E3F2FD'],
                [0.4, '#90CAF9'],
                [0.6, '#42A5F5'],
                [0.8, '#1976D2'],
                [1.0, '#0D47A1']
            ],
            showscale=(idx == 0),
            colorbar=dict(
                title=dict(text="<b>观察次数</b>"),
                tickmode="linear",
                tick0=0,
                dtick=1
            ) if idx == 0 else None,
            zmin=0,
            zmax=max(6, full_df['观察次数'].max())
        )

        fig.add_trace(heatmap, row=row, col=col)

    # 更新布局
    fig.update_layout(
        title={
            'text': '雪球敲出观察日日历',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=280 * rows,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig.update_yaxes(showticklabels=False)

    return fig

def print_trade_details(trade_id, df):
    """显示交易详情"""
    trade_data = df.loc[df['Trade Id'] == trade_id]

    if trade_data.empty:
        st.error(f"未找到交易 ID: {trade_id}")
        return

    latest_record = trade_data.sort_values('观察日').iloc[-1]

    st.markdown("### 📋 交易详情")

    # 基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("交易日期", latest_record['交易日期'])
        st.metric("交易方向", latest_record['交易方向'])
    with col2:
        st.metric("产品类型", latest_record['产品类型'])
        st.metric("期权类型", latest_record['TRADE_KEYWORD.期权特殊类型'])
    with col3:
        st.metric("标的资产", latest_record['标的名称'])
        st.metric("交割币种", latest_record['交割币种'])

    st.markdown("---")

    # 时间信息
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = latest_record['起始日期']
        if pd.notna(start_date):
            st.metric("起始日期", start_date.strftime('%Y-%m-%d'))
        else:
            st.metric("起始日期", "N/A")
    with col2:
        maturity_date = latest_record['到期日']
        if pd.notna(maturity_date):
            st.metric("到期日期", maturity_date.strftime('%Y-%m-%d'))
        else:
            st.metric("到期日期", "N/A")
    with col3:
        if pd.notna(latest_record['到期日']):
            days_remaining = (latest_record['到期日'] - datetime.datetime.now()).days
            st.metric("剩余天数", f"{days_remaining} 天")
        else:
            st.metric("剩余天数", "N/A")

    st.markdown("---")

    # 关键参数
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        strike = latest_record['行权价']
        st.metric("行权价", f"{strike:.2f}" if pd.notna(strike) else "N/A")
    with col2:
        barrier = latest_record['障碍向上敲出水平']
        st.metric("障碍水平", f"{barrier:.2f}" if pd.notna(barrier) else "N/A")
    with col3:
        notional = latest_record['名义本金']
        st.metric("名义本金", f"{notional:,.0f}" if pd.notna(notional) else "N/A")
    with col4:
        vol = latest_record['波动率']
        st.metric("波动率", f"{vol:.4f}%" if pd.notna(vol) else "N/A")

    st.markdown("---")

    # 财务指标
    st.markdown("### 💰 财务指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        premium = latest_record['期权费']
        st.metric("期权费", f"{premium:,.2f}" if pd.notna(premium) else "N/A")
    with col2:
        npv = latest_record['期权估值 NPV']
        st.metric("期权估值(NPV)", f"{npv:,.2f}" if pd.notna(npv) else "N/A")
    with col3:
        total_pnl = latest_record['总盈亏']
        st.metric("总盈亏", f"{total_pnl:,.2f}" if pd.notna(total_pnl) else "N/A")
    with col4:
        ytd_pnl = latest_record['本年总盈亏']
        st.metric("本年盈亏", f"{ytd_pnl:,.2f}" if pd.notna(ytd_pnl) else "N/A")
    with col5:
        overnight_pnl = latest_record['隔夜盈亏']
        st.metric("隔夜盈亏", f"{overnight_pnl:,.2f}" if pd.notna(overnight_pnl) else "N/A")

    st.markdown("---")

    # 希腊值
    st.markdown("### 📊 希腊值")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta = latest_record['DELTA(期权)']
        st.metric("Delta", f"{delta:,.2f}" if pd.notna(delta) else "N/A")
    with col2:
        gamma = latest_record['GAMMA']
        st.metric("Gamma", f"{gamma:,.2f}" if pd.notna(gamma) else "N/A")
    with col3:
        vega = latest_record['VEGA']
        st.metric("Vega", f"{vega:,.2f}" if pd.notna(vega) else "N/A")
    with col4:
        rho = latest_record['RHO']
        st.metric("Rho", f"{rho:,.2f}" if pd.notna(rho) else "N/A")
    with col5:
        theta = latest_record['THETA']
        st.metric("Theta", f"{theta:,.2f}" if pd.notna(theta) else "N/A")

    # 风险分析
    st.markdown("### ⚠️ 风险分析")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if pd.notna(latest_record['DELTA(期权)']) and pd.notna(latest_record['spot price']):
            delta_risk = latest_record['DELTA(期权)'] * latest_record['spot price']
            st.metric("Delta现金风险", f"{delta_risk:,.2f}")
        else:
            st.metric("Delta现金风险", "N/A")
    with col2:
        if pd.notna(latest_record['GAMMA']) and pd.notna(latest_record['spot price']):
            gamma_risk = latest_record['GAMMA'] * (latest_record['spot price'] ** 2) * 0.01
            st.metric("Gamma风险(1%变动)", f"{gamma_risk:,.2f}")
        else:
            st.metric("Gamma风险(1%变动)", "N/A")
    with col3:
        if pd.notna(latest_record['VEGA']):
            vega_exposure = latest_record['VEGA'] * 0.01
            st.metric("Vega风险(1%波动率)", f"{vega_exposure:,.2f}")
        else:
            st.metric("Vega风险(1%波动率)", "N/A")
    with col4:
        if pd.notna(latest_record['THETA']):
            theta_decay = latest_record['THETA']
            st.metric("每日Theta损益", f"{theta_decay:,.2f}")
        else:
            st.metric("每日Theta损益", "N/A")

    # 警告信息
    if pd.notna(latest_record['到期日']):
        days_remaining = (latest_record['到期日'] - datetime.datetime.now()).days
        if days_remaining < 30:
            st.warning(f"⚠️ 警告: 交易即将到期! 剩余天数: {days_remaining}天")

def min_max_scale(series):
    """Min-Max标准化"""
    positive_count = (series > 0).sum()
    negative_count = (series < 0).sum()

    if positive_count > negative_count:
        return (series - series.min()) / (series.max() - series.min())
    elif negative_count > positive_count:
        return -(series - series.min()) / (series.max() - series.min())
    else:
        return (series - series.min()) / (series.max() - series.min())

def create_greeks_chart(df_normalized):
    """创建希腊值分析图表"""
    x = df_normalized['观察日']

    fig = go.Figure()

    # 添加现货价格（如果存在）
    if 'spot price' in df_normalized.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df_normalized['spot price'],
            name='Spot Price',
            mode='lines',
            line=dict(color='black', width=2.5),
            marker=dict(size=6, symbol='circle')
        ))

    # 添加Delta
    fig.add_trace(go.Scatter(
        x=x, y=df_normalized['DELTA(期权)'],
        name='Delta',
        mode='lines',
        line=dict(color='royalblue', width=2, dash='solid'),
        marker=dict(size=6, symbol='circle')
    ))

    # 添加Gamma
    fig.add_trace(go.Scatter(
        x=x, y=df_normalized['GAMMA'],
        name='Gamma',
        mode='lines',
        line=dict(color='firebrick', width=2, dash='dashdot'),
    ))

    # 添加Vega
    fig.add_trace(go.Scatter(
        x=x, y=df_normalized['VEGA（1%）'],
        name='Vega',
        mode='lines',
        line=dict(color='green', width=2),
    ))

    # 添加总盈亏
    fig.add_trace(go.Scatter(
        x=x, y=df_normalized['总盈亏'],
        name='PNL',
        mode='lines',
        line=dict(color='orange', width=3),
    ))

    fig.update_layout(
        title='Min-Max Standardized Greeks Analysis',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            bordercolor='black',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Rockwell'
        ),
        height=500
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinecolor='Gray')

    return fig

def create_metric_chart(picked_data, metric_name, title):
    """创建单个指标的面积图"""
    if metric_name not in picked_data.columns:
        return None

    min_y = picked_data[metric_name].min()
    padding = abs(min_y) * 0.1 if min_y != 0 else 0.5
    baseline = min_y - padding

    fig = go.Figure()

    # 添加折线
    fig.add_trace(go.Scatter(
        x=picked_data['观察日'],
        y=picked_data[metric_name],
        mode='lines',
        name=metric_name,
        line=dict(color='rgb(0,100,80)', width=2)
    ))

    # 添加阴影区域
    fig.add_trace(go.Scatter(
        x=picked_data['观察日'],
        y=picked_data[metric_name],
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0,100,80,0.2)',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title='观察日',
        yaxis_title=metric_name,
        template='plotly_white',
        yaxis_range=[baseline, picked_data[metric_name].max() + padding],
        height=400
    )

    return fig

# 主应用
def main():
    st.markdown('<div class="main-header">📊 交易信息汇总看板</div>', unsafe_allow_html=True)

    # 加载数据
    try:
        otc_trade = load_data()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.info("请确保 '交易信息汇总.xlsx' 文件在当前目录下")
        return

    # 侧边栏
    st.sidebar.header("⚙️ 筛选选项")

    # 交易ID选择
    trade_ids = ['全部'] + list(otc_trade['Trade Id'].unique())
    selected_trade_id = st.sidebar.selectbox("选择 Trade ID", trade_ids, index=0)

    # 日期选择
    st.sidebar.subheader("📅 日期选择")
    available_dates = sorted(otc_trade['观察日'].unique())

    # 默认选择最新日期
    default_current_idx = len(available_dates) - 1 if len(available_dates) > 0 else 0

    current_date = st.sidebar.selectbox(
        "当前日期",
        available_dates,
        index=default_current_idx,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )

    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📋 交易概览", "📅 观察日日历", "📈 希腊值分析", "📊 指标详情"])

    # Tab 1: 交易概览
    with tab1:
        st.markdown(f"## 交易概览 - {current_date.strftime('%Y-%m-%d')}")

        # 筛选当前日期的数据
        current_date_data = otc_trade[otc_trade['观察日'] == current_date]

        # 显示汇总统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总交易数", len(current_date_data['Trade Id'].unique()))
        with col2:
            total_notional = current_date_data['名义本金'].sum()
            st.metric("总名义本金", f"{total_notional:,.0f}")
        with col3:
            total_pnl = current_date_data['总盈亏'].sum()
            st.metric("总盈亏", f"{total_pnl:,.2f}")
        with col4:
            avg_vol = current_date_data['波动率'].mean()
            st.metric("平均波动率", f"{avg_vol:.4f}%")

        st.markdown("---")

        # 如果选择了特定交易ID，显示详情
        if selected_trade_id != '全部':
            # 传入筛选后的数据
            trade_data_on_date = current_date_data[current_date_data['Trade Id'] == selected_trade_id]
            if not trade_data_on_date.empty:
                print_trade_details(selected_trade_id, otc_trade)
            else:
                st.warning(f"⚠️ 交易 {selected_trade_id} 在 {current_date.strftime('%Y-%m-%d')} 没有数据")
        else:
            st.info(f"📊 显示 {current_date.strftime('%Y-%m-%d')} 所有交易组合的汇总分析")

            # 组合财务指标汇总（使用当前选择日期的数据）
            st.markdown("### 💰 组合财务指标")
            # 使用当前日期的数据
            latest_data = current_date_data.copy()

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                total_premium = latest_data['期权费'].sum()
                st.metric("总期权费", f"{total_premium:,.2f}")
            with col2:
                total_npv = latest_data['期权估值 NPV'].sum()
                st.metric("组合估值(NPV)", f"{total_npv:,.2f}")
            with col3:
                portfolio_pnl = latest_data['总盈亏'].sum()
                st.metric("组合总盈亏", f"{portfolio_pnl:,.2f}")
            with col4:
                ytd_pnl = latest_data['本年总盈亏'].sum()
                st.metric("本年总盈亏", f"{ytd_pnl:,.2f}")
            with col5:
                overnight_pnl = latest_data['隔夜盈亏'].sum()
                st.metric("隔夜盈亏", f"{overnight_pnl:,.2f}")

            st.markdown("---")

            # 组合希腊值汇总
            st.markdown("### 📊 组合希腊值")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                total_delta = latest_data['DELTA(期权)'].sum()
                st.metric("组合 Delta", f"{total_delta:,.2f}")
            with col2:
                total_gamma = latest_data['GAMMA'].sum()
                st.metric("组合 Gamma", f"{total_gamma:,.2f}")
            with col3:
                total_vega = latest_data['VEGA'].sum()
                st.metric("组合 Vega", f"{total_vega:,.2f}")
            with col4:
                total_theta = latest_data['THETA'].sum()
                st.metric("组合 Theta", f"{total_theta:,.2f}")
            with col5:
                total_rho = latest_data['RHO'].sum()
                st.metric("组合 Rho", f"{total_rho:,.2f}")

            st.markdown("---")

            # 组合风险分析
            st.markdown("### ⚠️ 组合风险分析")

            # 计算每个交易的风险指标，然后求和（Delta现金风险剔除EquityLinkedSwap）
            delta_cash_risks = []
            gamma_risks = []
            vega_risks = []
            theta_risks = []
            excluded_count = 0  # 统计剔除的交易数量

            for idx, row in latest_data.iterrows():
                # Delta现金风险 = Delta × Spot Price (剔除EquityLinkedSwap和None类型)
                if pd.notna(row['DELTA(期权)']) and pd.notna(row.get('spot price')):
                    option_type = row['TRADE_KEYWORD.期权特殊类型']
                    # 剔除 EquityLinkedSwap 和 None（空值）
                    if option_type == 'EquityLinkedSwap' or pd.isna(option_type):
                        excluded_count += 1
                    else:
                        delta_cash_risks.append(row['DELTA(期权)'] * row['spot price'])

                # Gamma风险 = Gamma × Spot² × 1%
                if pd.notna(row['GAMMA']) and pd.notna(row.get('spot price')):
                    gamma_risks.append(row['GAMMA'] * (row['spot price'] ** 2) * 0.01)

                # Vega风险 = Vega × 1%
                if pd.notna(row['VEGA']):
                    vega_risks.append(row['VEGA'] * 0.01)

                # Theta = 每日时间价值变化
                if pd.notna(row['THETA']):
                    theta_risks.append(row['THETA'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if delta_cash_risks:
                    portfolio_delta_risk = sum(delta_cash_risks)
                    st.metric("组合Delta现金风险", f"{portfolio_delta_risk:,.2f}",
                             help=f"已剔除 {excluded_count} 笔交易（EquityLinkedSwap或期权类型为空）")
                else:
                    st.metric("组合Delta现金风险", "N/A",
                             help=f"已剔除 {excluded_count} 笔交易（EquityLinkedSwap或期权类型为空）")

            with col2:
                if gamma_risks:
                    portfolio_gamma_risk = sum(gamma_risks)
                    st.metric("组合Gamma风险(1%变动)", f"{portfolio_gamma_risk:,.2f}")
                else:
                    st.metric("组合Gamma风险(1%变动)", "N/A")

            with col3:
                if vega_risks:
                    portfolio_vega_exposure = sum(vega_risks)
                    st.metric("组合Vega风险(1%波动率)", f"{portfolio_vega_exposure:,.2f}")
                else:
                    st.metric("组合Vega风险(1%波动率)", "N/A")

            with col4:
                if theta_risks:
                    portfolio_theta_decay = sum(theta_risks)
                    st.metric("组合每日Theta损益", f"{portfolio_theta_decay:,.2f}")
                else:
                    st.metric("组合每日Theta损益", "N/A")

            st.markdown("---")

            # 显示当前日期所有交易汇总表
            st.markdown(f"### 📋 {current_date.strftime('%Y-%m-%d')} 所有交易汇总")
            summary_df = current_date_data[['Trade Id', '标的名称', 'TRADE_KEYWORD.期权特殊类型',
                                           '交易日期', '到期日', '名义本金', '总盈亏',
                                           '期权估值 NPV', 'DELTA(期权)', 'GAMMA', 'VEGA']].copy()

            # 格式化显示
            summary_df.columns = ['Trade ID', '标的', '期权类型', '交易日期', '到期日',
                                  '名义本金', '总盈亏', 'NPV', 'Delta', 'Gamma', 'Vega']
            st.dataframe(summary_df, use_container_width=True)

    # Tab 2: 观察日日历
    with tab2:
        st.markdown("## 雪球敲出观察日日历")

        try:
            obs_df = expand_observation_dates(otc_trade)

            # 显示统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("独立雪球期权交易", len(obs_df['Trade Id'].unique()))
            with col2:
                st.metric("总观察日记录", len(obs_df))
            with col3:
                date_range = f"{obs_df['观察日'].min().strftime('%Y-%m-%d')} 至 {obs_df['观察日'].max().strftime('%Y-%m-%d')}"
                st.info(f"日期范围: {date_range}")

            # 显示日历热力图
            calendar_fig = create_enhanced_calendar_heatmap(obs_df)
            st.plotly_chart(calendar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"日历生成失败: {e}")

    # Tab 3: 希腊值分析
    with tab3:
        st.markdown("## 希腊值分析")

        # 准备数据（包含所有状态，包括TERMINATED）
        if selected_trade_id == '全部':
            # 获取所有标的列表
            underlying_list = sorted(otc_trade['标的名称'].unique().tolist())

            # 添加可视化模式选择
            viz_mode = st.radio(
                "选择可视化模式",
                ["分资产对比图", "单一资产详情", "组合级指标(仅Vega/Theta)"],
                horizontal=True
            )

            if viz_mode == "组合级指标(仅Vega/Theta)":
                # 仅显示可跨标的加总的指标：Vega、Theta、Rho
                # 按观察日直接汇总
                time_series_data = otc_trade.groupby('观察日').agg({
                    '期权估值（报送）': 'sum',
                    '期权估值 NPV': 'sum',
                    '期权费估值': 'sum',
                    'VEGA（1%）': 'sum',
                    'THETA': 'sum',
                    'RHO': 'sum',
                    '总盈亏': 'sum'
                }).reset_index()

                picked_data = time_series_data.set_index('观察日')
                cols_to_normalize = ['期权估值（报送）','期权估值 NPV','期权费估值','VEGA（1%）','THETA','RHO','总盈亏']

                # 标准化
                df_normalized = picked_data.copy()
                df_normalized[cols_to_normalize] = df_normalized[cols_to_normalize].apply(min_max_scale)
                df_normalized = round(df_normalized, 4)
                df_normalized.reset_index(inplace=True)

                # 显示希腊值综合图表
                greeks_fig = create_greeks_chart(df_normalized)
                st.plotly_chart(greeks_fig, use_container_width=True)

            elif viz_mode == "单一资产详情":
                # 选择单一标的查看详细信息
                selected_underlying = st.selectbox("选择标的资产", underlying_list, index=0)

                # 按选定标的筛选数据
                underlying_data = otc_trade[otc_trade['标的名称'] == selected_underlying].copy()

                # 按观察日直接汇总该标的的所有记录
                time_series_data = underlying_data.groupby('观察日').agg({
                    'spot price': 'first',  # spot price对同一标的同一天应该是相同的
                    '期权估值（报送）': 'sum',
                    '期权估值 NPV': 'sum',
                    '期权费估值': 'sum',
                    'DELTA(期权)': 'sum',
                    'GAMMA': 'sum',
                    'VEGA（1%）': 'sum',
                    'THETA': 'sum',
                    'RHO': 'sum',
                    '总盈亏': 'sum'
                }).reset_index()

                picked_data = time_series_data.set_index('观察日')
                cols_to_normalize = ['spot price','期权估值（报送）','期权估值 NPV','期权费估值','DELTA(期权)','GAMMA','VEGA（1%）','THETA','RHO','总盈亏']

                # 标准化
                df_normalized = picked_data.copy()
                df_normalized[cols_to_normalize] = df_normalized[cols_to_normalize].apply(min_max_scale)
                df_normalized = round(df_normalized, 4)
                df_normalized.reset_index(inplace=True)

                # 显示希腊值综合图表
                greeks_fig = create_greeks_chart(df_normalized)
                st.plotly_chart(greeks_fig, use_container_width=True)

            else:  # 分资产对比图模式
                # 准备按标的分组的数据
                # 按观察日和标的名称直接汇总
                underlying_summary = otc_trade.groupby(['观察日', '标的名称']).agg({
                    'spot price': 'first',
                    'DELTA(期权)': 'sum',
                    'GAMMA': 'sum',
                    'VEGA（1%）': 'sum',
                    'THETA': 'sum',
                    '总盈亏': 'sum',
                    '隔夜盈亏': 'sum'
                }).reset_index()

                # 选择要展示的希腊值
                greek_to_plot = st.selectbox(
                    "选择希腊值指标",
                    ['DELTA(期权)', 'GAMMA', 'VEGA（1%）', 'THETA', '总盈亏', '隔夜盈亏'],
                    index=0
                )

                # 添加说明
                if greek_to_plot == '总盈亏':
                    st.info("💡 总盈亏是累积值，显示从交易开始至该日的累计盈亏")
                elif greek_to_plot == '隔夜盈亏':
                    st.info("💡 隔夜盈亏是单日增量值，显示该日的盈亏变化")

                # 创建分资产对比图
                fig = go.Figure()

                colors = ['royalblue', 'firebrick', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']

                for idx, underlying in enumerate(underlying_list):
                    underlying_df = underlying_summary[underlying_summary['标的名称'] == underlying]

                    if not underlying_df.empty:
                        fig.add_trace(go.Scatter(
                            x=underlying_df['观察日'],
                            y=underlying_df[greek_to_plot],
                            name=underlying,
                            mode='lines+markers',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            marker=dict(size=5)
                        ))

                fig.update_layout(
                    title=f'{greek_to_plot} - 按标的资产对比',
                    xaxis_title='观察日',
                    yaxis_title=greek_to_plot,
                    template='plotly_white',
                    hovermode='x unified',
                    legend=dict(
                        bordercolor='black',
                        borderwidth=1
                    ),
                    height=600
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

                st.plotly_chart(fig, use_container_width=True)

        else:
            # 单个交易的希腊值分析
            picked_data = otc_trade.loc[otc_trade['Trade Id'] == selected_trade_id]
            picked_data = picked_data[['观察日','spot price','期权估值（报送）','期权估值 NPV','期权费估值','DELTA(期权)','GAMMA','VEGA（1%）','THETA','RHO','总盈亏','交易状态']]
            # 不再筛选交易状态，包含所有状态
            cols_to_normalize = ['spot price','期权估值（报送）','期权估值 NPV','期权费估值','DELTA(期权)','GAMMA','VEGA（1%）','THETA','RHO','总盈亏']

            # 标准化
            df_normalized = picked_data.copy()
            df_normalized[cols_to_normalize] = df_normalized[cols_to_normalize].apply(min_max_scale)
            df_normalized = round(df_normalized, 4)
            df_normalized.reset_index(inplace=True)

            # 显示希腊值综合图表
            greeks_fig = create_greeks_chart(df_normalized)
            st.plotly_chart(greeks_fig, use_container_width=True)

    # Tab 4: 指标详情
    with tab4:
        st.markdown("## 指标详情")

        if selected_trade_id == '全部':
            st.warning("请选择特定的 Trade ID 查看指标详情")
        else:
            picked_data = otc_trade.loc[otc_trade['Trade Id'] == selected_trade_id]
            picked_data = picked_data.loc[picked_data['交易状态']=='VERIFIED']

            # 创建各个指标图表
            metrics = [
                ('spot price', 'Spot Price'),
                ('DELTA(期权)', 'DELTA(期权)'),
                ('GAMMA', 'GAMMA'),
                ('VEGA（1%）', 'VEGA（1%）')
            ]

            for metric_name, title in metrics:
                fig = create_metric_chart(picked_data, metric_name, title)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
