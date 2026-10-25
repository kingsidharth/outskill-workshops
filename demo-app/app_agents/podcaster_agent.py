import os
from datetime import datetime

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import EXASearchTool
from dotenv import load_dotenv

load_dotenv("../.env")

os.makedirs("task_outputs/podcaster", exist_ok=True)

llm = LLM(
    model="openai/gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url="https://api.openai.com/v1",
)

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "")
search_tool = EXASearchTool()


def _build_researcher():
    return Agent(
        role="AI Podcast Researcher",
        goal="Find recent releases, articles, topics, and progress about {topic}",
        backstory="You are a seasoned researcher who uncovers the latest developments and presents them clearly.",
        llm=llm,
        tools=[search_tool],
        verbose=False,
    )


def _build_reporting_analyst():
    return Agent(
        role="AI Podcast Reporting Analyst",
        goal="Create detailed reports based on {topic} research findings",
        backstory="You are a meticulous analyst who turns complex research into clear podcast briefing documents.",
        llm=llm,
        verbose=False,
    )


def _build_scriptwriter():
    return Agent(
        role="Podcast Scriptwriter",
        goal="Create a fun, natural, interesting podcast script from a research report on {topic}",
        backstory=(
            "You are a talented scriptwriter who turns technical reports into engaging "
            "podcast conversations between two hosts."
        ),
        llm=llm,
        verbose=False,
    )


def run_podcast(topic: str, callback=None):
    """Run the 3-step podcast pipeline synchronously. Returns dict with results."""
    date_str = datetime.now().strftime("%B %d, %Y")
    inputs = {"topic": topic, "date": date_str}

    researcher = _build_researcher()
    reporting_analyst = _build_reporting_analyst()
    scriptwriter = _build_scriptwriter()

    # Step 1: Research
    if callback:
        callback("research", "running")
    research_task = Task(
        description=(
            "Conduct podcast research about {topic}. Find recent releases, articles, "
            "topics, and progress. Use the current date: {date}."
        ),
        expected_output="A list of 8-10 relevant findings with title, summary, and listener angle.",
        agent=researcher,
        output_file="task_outputs/podcaster/research.md",
    )
    research_crew = Crew(
        agents=[researcher], tasks=[research_task], process=Process.sequential, verbose=True
    )
    research_crew.kickoff(inputs=inputs)
    if callback:
        callback("research", "done")

    # Step 2: Report (with no-code guardrail)
    if callback:
        callback("report", "running")
    reporting_task = Task(
        description="Review the research findings and write a detailed podcast briefing for {topic}.",
        expected_output=(
            "A markdown report with at least 4 clearly labeled sections, "
            "each containing at least 3 bullet points of facts."
        ),
        guardrail=(
            "Reject if the report has fewer than 4 sections with ## headers. "
            "Reject if any section has fewer than 3 bullet points."
        ),
        agent=reporting_analyst,
        context=[research_task],
        output_file="task_outputs/podcaster/briefing.md",
    )
    report_crew = Crew(
        agents=[reporting_analyst], tasks=[reporting_task], process=Process.sequential, verbose=True
    )
    report_crew.kickoff(inputs=inputs)
    if callback:
        callback("report", "done")

    # Step 3: Script (with no-code guardrail)
    if callback:
        callback("script", "running")
    scripting_task = Task(
        description=(
            "Write a podcast script between two hosts Joe and Jane based on the briefing. "
            "Include a cold open, topic transitions, and a closing summary. "
            "Keep it lively and under 80 exchanges."
        ),
        expected_output="A complete podcast script with speaker labels for Joe and Jane.",
        guardrail=(
            "The script must use only Joe and Jane as speaker labels. "
            "It must include at least 12 lines of dialogue."
        ),
        agent=scriptwriter,
        context=[reporting_task],
        output_file="task_outputs/podcaster/script.md",
    )
    script_crew = Crew(
        agents=[scriptwriter], tasks=[scripting_task], process=Process.sequential, verbose=True
    )
    script_result = script_crew.kickoff(inputs=inputs)
    if callback:
        callback("script", "done")

    return {
        "topic": topic,
        "script": script_result.raw,
    }
