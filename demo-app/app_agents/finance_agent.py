import os
from datetime import datetime

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import EXASearchTool
from dotenv import load_dotenv

load_dotenv("../.env")

os.makedirs("task_outputs/finance", exist_ok=True)

llm = LLM(
    model="openai/gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url="https://api.openai.com/v1",
)

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")
search_tool = EXASearchTool()


def _build_researcher():
    return Agent(
        role="Market Researcher",
        goal="Research recent financial news, earnings, and market sentiment for {company}",
        backstory="You are a senior equity researcher who finds the most relevant market data and news.",
        llm=llm,
        tools=[search_tool],
        verbose=False,
    )


def _build_risk_analyst():
    return Agent(
        role="Risk Analyst",
        goal="Assess investment risks for {company} based on market research",
        backstory="You are a quantitative risk analyst who identifies financial, regulatory, and market risks.",
        llm=llm,
        verbose=False,
    )


def _build_strategist():
    return Agent(
        role="Investment Strategist",
        goal="Create a portfolio strategy recommendation for {company}",
        backstory="You synthesize research and risk data into actionable BUY/HOLD/SELL recommendations.",
        llm=llm,
        verbose=False,
    )


def _build_advisor():
    return Agent(
        role="Client Advisor",
        goal="Write a clear, client-facing investment advisory report for {company}",
        backstory="You translate technical analysis into plain-language reports that clients can act on.",
        llm=llm,
        verbose=False,
    )


def run_analysis(company: str, callback=None):
    """Run the 4-step finance pipeline synchronously. Returns dict with results."""
    date_str = datetime.now().strftime("%B %d, %Y")
    inputs = {"company": company, "date": date_str}

    researcher = _build_researcher()
    risk_analyst = _build_risk_analyst()
    strategist = _build_strategist()
    advisor = _build_advisor()

    # Step 1: Market Research
    if callback:
        callback("market_research", "running")
    research_task = Task(
        description=(
            "Research {company} — find recent earnings, news, analyst opinions, and market trends. "
            "Use the current date: {date}."
        ),
        expected_output="A comprehensive market research brief covering financials, news, and sentiment.",
        agent=researcher,
        output_file="task_outputs/finance/market_research.md",
    )
    research_crew = Crew(
        agents=[researcher], tasks=[research_task], process=Process.sequential, verbose=True
    )
    research_result = research_crew.kickoff(inputs=inputs)
    if callback:
        callback("market_research", "done")

    # Step 2: Risk Assessment
    if callback:
        callback("risk_assessment", "running")
    risk_task = Task(
        description=(
            "Assess investment risks for {company} based on the market research provided. "
            "Classify overall risk as: LOW, MODERATE, or HIGH. "
            "Start your response with 'Risk Level: <level>' on the first line."
        ),
        expected_output="Risk assessment with risk level classification, key risks, and mitigating factors.",
        agent=risk_analyst,
        context=[research_task],
        output_file="task_outputs/finance/risk_assessment.md",
    )
    risk_crew = Crew(
        agents=[risk_analyst], tasks=[risk_task], process=Process.sequential, verbose=True
    )
    risk_result = risk_crew.kickoff(inputs=inputs)
    if callback:
        callback("risk_assessment", "done")

    # Step 3: Routing — check risk level
    risk_text = risk_result.raw.lower()
    is_high_risk = "high" in risk_text.split("\n")[0] if risk_text else False

    if is_high_risk:
        if callback:
            callback("compliance", "running")
        compliance_note = (
            "COMPLIANCE FLAG: High risk detected. Strategy must include position size limits, "
            "stop-loss recommendations, and explicit risk disclosure for the client."
        )
        if callback:
            callback("compliance", "done")
    else:
        compliance_note = "No compliance flags. Standard recommendations apply."
        if callback:
            callback("compliance", "skipped")

    # Step 4: Advisory Report
    if callback:
        callback("advisory", "running")
    strategy_task = Task(
        description=(
            "Create an investment strategy for {company} considering the research and risk assessment. "
            f"Compliance note: {compliance_note}"
        ),
        expected_output="BUY/HOLD/SELL recommendation with rationale, price targets, and time horizon.",
        agent=strategist,
        context=[research_task, risk_task],
        output_file="task_outputs/finance/strategy.md",
    )
    advisory_task = Task(
        description=(
            "Write a client-facing advisory report for {company} that summarizes the research, "
            "risk assessment, and strategy recommendation in clear language."
        ),
        expected_output="A professional advisory report suitable for sending to investment clients.",
        agent=advisor,
        context=[research_task, risk_task, strategy_task],
        output_file="task_outputs/finance/advisory_report.md",
    )
    advisory_crew = Crew(
        agents=[strategist, advisor],
        tasks=[strategy_task, advisory_task],
        process=Process.sequential,
        verbose=True,
    )
    advisory_result = advisory_crew.kickoff(inputs=inputs)
    if callback:
        callback("advisory", "done")

    return {
        "company": company,
        "risk_level": "HIGH" if is_high_risk else "NORMAL",
        "advisory": advisory_result.raw,
    }
