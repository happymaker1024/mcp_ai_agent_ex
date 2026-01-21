from crewai.tools import tool
import yfinance as yf
import pandas as pd


@tool("Updated Comprehensive Stock Analysis")
def comprehensive_stock_analysis(ticker: str) -> dict:
    """
    주어진 주식 티커에 대한 종합 재무 분석
    """

    # -----------------------------
    # 1. 공통 유틸 함수
    # -----------------------------
    def format_number(x):
        if x is None or pd.isna(x):
            return "N/A"
        return f"{x:,.0f}"

    def format_percent(x):
        if x is None or pd.isna(x):
            return "N/A"
        return f"{x:.2f}%"

    def growth_rate(current, previous):
        if previous and current and previous != 0:
            return (current - previous) / abs(previous) * 100
        return None

    # -----------------------------
    # 2. 데이터 수집
    # -----------------------------
    stock = yf.Ticker(ticker)

    price_df = stock.history(period="1d")
    latest_price = price_df["Close"].iloc[-1]
    latest_time = price_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")

    annual_fin = stock.get_financials()
    quarterly_fin = stock.get_financials(freq="quarterly")
    balance = stock.get_balance_sheet()

    # 가장 최근 컬럼
    cur = annual_fin.columns[0]
    prev = annual_fin.columns[1]

    q_cur = quarterly_fin.columns[0]
    q_prev = quarterly_fin.columns[1]

    # -----------------------------
    # 3. 연간 주요 수치
    # -----------------------------
    revenue = annual_fin.loc["TotalRevenue", cur]
    cost = annual_fin.loc["CostOfRevenue", cur]
    gross_profit = annual_fin.loc["GrossProfit", cur]
    operating_income = annual_fin.loc["OperatingIncome", cur]
    net_income = annual_fin.loc["NetIncome", cur]
    ebitda = annual_fin.loc["EBITDA", cur]
    eps = annual_fin.loc["DilutedEPS", cur]

    # 성장률
    revenue_growth = growth_rate(revenue, annual_fin.loc["TotalRevenue", prev])
    net_income_growth = growth_rate(net_income, annual_fin.loc["NetIncome", prev])

    # 수익성
    gross_margin = (gross_profit / revenue) * 100 if revenue else None
    operating_margin = (operating_income / revenue) * 100 if revenue else None
    net_margin = (net_income / revenue) * 100 if revenue else None

    # 부채비율
    total_assets = balance.loc["TotalAssets", balance.columns[0]]
    total_liabilities = balance.loc["TotalLiabilitiesNetMinorityInterest", balance.columns[0]]
    debt_ratio = (total_liabilities / total_assets) * 100 if total_assets else None

    # -----------------------------
    # 4. 분기 주요 수치
    # -----------------------------
    q_revenue = quarterly_fin.loc["TotalRevenue", q_cur]
    q_net_income = quarterly_fin.loc["NetIncome", q_cur]

    q_revenue_growth = growth_rate(q_revenue, quarterly_fin.loc["TotalRevenue", q_prev])
    q_net_income_growth = growth_rate(q_net_income, quarterly_fin.loc["NetIncome", q_prev])

    # -----------------------------
    # 5. 재무제표 요약 함수
    # -----------------------------
    def financial_summary(fin_df):
        result = {}
        for col in fin_df.columns:
            date = col.strftime("%Y-%m-%d")
            result[date] = {
                "총수익": format_number(fin_df.loc["TotalRevenue", col]),
                "영업이익": format_number(fin_df.loc["OperatingIncome", col]),
                "순이익": format_number(fin_df.loc["NetIncome", col]),
                "EBITDA": format_number(fin_df.loc["EBITDA", col]),
                "EPS": f"${fin_df.loc['DilutedEPS', col]:.2f}" if pd.notna(fin_df.loc["DilutedEPS", col]) else "N/A"
            }
        return result

    # -----------------------------
    # 6. 최종 결과 구성
    # -----------------------------
    result = {
        "현재 주가": {
            "가격": latest_price,
            "기준 시간": latest_time
        },

        "연간 데이터": {
            "매출": format_number(revenue),
            "매출원가": format_number(cost),
            "매출총이익": format_number(gross_profit),
            "영업이익": format_number(operating_income),
            "순이익": format_number(net_income),
            "EBITDA": format_number(ebitda),
            "EPS": f"${eps:.2f}" if pd.notna(eps) else "N/A",
            "매출총이익률": format_percent(gross_margin),
            "영업이익률": format_percent(operating_margin),
            "순이익률": format_percent(net_margin),
            "매출 성장률": format_percent(revenue_growth),
            "순이익 성장률": format_percent(net_income_growth),
            "부채비율": format_percent(debt_ratio),
        },

        "분기 데이터": {
            "매출": format_number(q_revenue),
            "순이익": format_number(q_net_income),
            "매출 성장률(QoQ)": format_percent(q_revenue_growth),
            "순이익 성장률(QoQ)": format_percent(q_net_income_growth),
        },

        "연간 재무제표 요약": financial_summary(annual_fin),
        "분기 재무제표 요약": financial_summary(quarterly_fin),
    }

    return result

if __name__ == "__main__":
    analysis = comprehensive_stock_analysis.run("AAPL")
    import pprint
    pprint.pprint(analysis)