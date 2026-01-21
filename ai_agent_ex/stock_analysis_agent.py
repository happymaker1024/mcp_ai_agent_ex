from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from datetime import datetime

import os
from dotenv import load_dotenv
from marshmallow import pprint

# 사용자 정의 재무분석 툴 로딩
from stock_analysis_tool import comprehensive_stock_analysis

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def scock_analysis_agent(ticker: str) -> str:
    # 웹 겸색 툴 설정
    search_tool = SerperDevTool()

    current_time = datetime.now()
    llm = LLM(model = "openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    invest_llm = LLM(model = "openai/o3-mini-2025-01-31")
    # invest_llm = LLM(model="anthropic/claude-3-5-sonnet-20240620")

    # 재무 분석가
    financial_analyst = Agent(
        role="Financial Analyst",
        goal="회사의 재무 상태 및 성과 분석",
        backstory="당신은 재무 제표와 비율을 해석하는 데 전문성을 갖춘 노련한 재무 분석가입니다.날짜: {current_time:%Y년 %m월 %d일}",
        tools=[comprehensive_stock_analysis],
        llm=llm,
        max_iter=3,
        allow_delegation=False,
        verbose=True
    )

    # 시장 분석가
    market_analyst = Agent(
        role="Market Analyst",
        goal="회사의 시장 지위 및 업계 동향 분석",
        backstory="당신은 기업/산업 현황 및 경쟁 환경을 전문적으로 분석할 수 있는 숙련된 시장 분석가입니다.날짜: {current_time:%Y년 %m월 %d일}",
        tools=[search_tool],
        llm=llm,
        max_iter=3,
        allow_delegation=False,
        # verbose=True
    )

    # 위험 분석가
    risk_analyst = Agent(
        role="Risk Analyst",
        goal="주식과 관련된 잠재적 위험 식별 및 평가",
        backstory="당신은 투자에서 명백한 위험과 숨겨진 위험을 모두 식별하는 예리한 안목을 갖춘 신중한 위험 분석가입니다.날짜: {current_time:%Y년 %m월 %d일}",
        tools=[comprehensive_stock_analysis],
        llm=llm,
        allow_delegation=False,
        # verbose=True
    )

    # 투자 전문가
    investment_advisor = Agent(
        role="Investment Advisor",
        goal="전체 분석을 기반으로 한 투자 추천 제공",
        backstory="다양한 분석을 종합하여 전략적 투자 조언을 제공하는 신뢰할 수 있는 투자 자문가입니다.날짜: {current_time:%Y년 %m월 %d일}",
        llm=invest_llm,
        # allow_delegation=False,
        allow_delegation=True,
        # verbose=True
    )

    def get_user_input():
        ticker = input("투자 자문을 구하고 싶은 기업명을 입력해주세요: ")
        return ticker

    def create_dynamic_tasks(ticker):
        financial_analysis = Task(
            name="financial_analysis",
            description=f"""{ticker}에 대한 철저한 재무 분석을 수행합니다. 
            주요 재무 지표에 집중하세요. 
            회사의 재무 건전성 및 성과 추세에 대한 인사이트를 제공합니다.날짜: {current_time:%Y년 %m월 %d일}""",
            agent=financial_analyst,
            expected_output=f"""{ticker}의 재무 상태에 대한 종합적인 분석 보고서. 
            주요 재무 지표, 수익성, 부채 비율 등을 포함하며, 
            회사의 재무 건전성과 성과 동향에 대한 인사이트를 제공해야 합니다."""
        )

        market_analysis = Task(
            name="market_analysis",
            description=f"""{ticker}의 시장 위치를 분석합니다. 
            경쟁 우위, 시장 점유율, 업계 동향을 평가하세요. 
            회사의 성장 잠재력과 시장 과제에 대한 인사이트를 제공하세요.날짜: {current_time:%Y년 %m월 %d일}""",
            agent=market_analyst,
            expected_output=f"""{ticker}의 시장 위치에 대한 상세한 분석 보고서. 
            경쟁 우위, 시장 점유율, 산업 동향을 평가하고, 
            회사의 성장 잠재력과 시장 과제에 대한 인사이트를 포함해야 합니다."""
        )

        risk_assessment = Task(
            name="risk_assessment",
            description=f"""{ticker}에 대한 투자와 관련된 주요 위험을 파악하고 평가합니다. 
            시장 위험, 운영 위험, 재무 위험 및 회사별 위험을 고려하세요. 
            종합적인 위험 프로필을 제공합니다.날짜: {current_time:%Y년 %m월 %d일}""",
            agent=risk_analyst,
            expected_output=f"""{ticker} 투자와 관련된 주요 리스크에 대한 포괄적인 평가 보고서. 
            시장 리스크, 운영 리스크, 재무 리스크, 회사 특정 리스크를 고려하여 
            종합적인 리스크 분석 결과를 제시해야 합니다."""
        )

        investment_recommendation = Task(
            name="investment_recommendation",
            description=f"""{ticker}의 재무 분석, 시장 분석, 위험 평가를 바탕으로 종합적인 투자 추천을 제공합니다. 
            주식의 잠재 수익률, 위험 및 다양한 유형의 투자자에 대한 적합성을 고려하여 마크다운으로 작성해 주세요.
            한글로 작성하세요.날짜: {current_time:%Y년 %m월 %d일}""",
            agent=investment_advisor,
            expected_output=f"""
            1. 제목 및 기본 정보
            - 회사명, 티커, 현재 주가, 목표주가, 투자의견 등
            
            2. 요약(Executive Summary)
            - 핵심 투자 포인트와 주요 재무 지표를 간단히 정리
            
            3. 기업 개요
            - 회사의 주요 사업 영역, 연혁, 시장 점유율 등
            
            4. 산업 및 시장 분석
            - 해당 기업이 속한 산업의 트렌드와 전망
            
            5. 재무 분석
            - 매출, 영업이익, 순이익 등 주요 재무지표 분석
            - 수익성, 성장성, 안정성 지표 분석
            
            6. 밸류에이션
            - P/E, P/B, ROE 등 주요 밸류에이션 지표 분석
            - 경쟁사 대비 상대 밸류에이션
            
            7. 투자 의견 및 목표주가
            - 투자의견 제시 및 근거 설명
            - 목표주가 산정 방법과 근거
            
            8. 투자 위험 요인
            - 잠재적인 리스크 요인들을 나열
            
            9. 재무제표 요약
            - 최근 몇 년간의 요약 손익계산서, 재무상태표, 현금흐름표

            """,
            output_file=f"./outputs/investment_advisor_{ticker}.md" 
        )

        return [financial_analysis, market_analysis, risk_assessment, investment_recommendation]


    # Crew 객체 생성
    tasks = create_dynamic_tasks(ticker)

    crew = Crew(
        agents=[financial_analyst, market_analyst, risk_analyst, investment_advisor],
        tasks=tasks,
        process=Process.sequential,
        output_log_file="logs/investment_advisor_run.json", 
        # verbose=True
    )

    result = crew.kickoff()

    return result

from rich.console import Console
from rich.markdown import Markdown

if __name__ == "__main__":
    ticker = "IREN"
    result = scock_analysis_agent(ticker)

    console = Console()
    console.print(Markdown(result.raw))