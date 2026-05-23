import asyncio
from openai import OpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    FileSearchTool,
)

_vector_store_id: str | None = None


class BookingContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    seat_update_requested: bool = False


@function_tool
def faq_lookup_tool(question: str) -> str:
    """Look up answers to frequently asked questions about the airline."""
    q = question.lower()
    if any(w in q for w in ["bag", "baggage", "luggage", "carry-on"]):
        return (
            "You are allowed one carry-on bag. "
            "It must be under 7 kgs and fit in the overhead bin (22x14x9 inches)."
        )
    if any(w in q for w in ["seat", "seats", "seating"]):
        return (
            "The plane has 120 seats: 22 business class and 98 economy. "
            "Rows 5-8 are Economy Plus with extra legroom."
        )
    if any(w in q for w in ["wifi", "internet", "wireless"]):
        return "Free wifi is available on all flights. Connect to 'IndigoDomesticWifi'."
    return "I'm sorry, I don't have an answer for that. Please contact support."


@function_tool
def update_seat(
    context: RunContextWrapper[BookingContext],
    confirmation_number: str,
    new_seat: str,
) -> str:
    """Update the seat for a given booking confirmation number."""
    if not context.context.seat_update_requested:
        return (
            "I can only update a seat when the customer explicitly asks to "
            "change, select, or move to a new seat."
        )

    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    context.context.seat_update_requested = False
    return f"Seat updated to {new_seat} for confirmation {confirmation_number}."


OFF_TOPIC_KEYWORDS = [
    "cricket", "recipe", "stock market", "movie", "politics",
    "weather", "bollywood", "ipl", "football",
]

SEAT_UPDATE_KEYWORDS = [
    "change my seat", "change seat", "update my seat", "update seat",
    "move my seat", "move seat", "select seat", "choose seat",
    "book seat", "assign seat", "seat to",
]


def _latest_user_text(input: str | list) -> str:
    if isinstance(input, str):
        return input

    for item in reversed(input):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue

        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return " ".join(parts)

    return str(input)


def _is_seat_update_request(message: str) -> bool:
    text = message.lower()
    return any(keyword in text for keyword in SEAT_UPDATE_KEYWORDS)


@input_guardrail
async def airline_topic_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list,
) -> GuardrailFunctionOutput:
    """Block messages that are clearly not related to airline travel."""
    text = _latest_user_text(input).lower()
    is_off_topic = any(kw in text for kw in OFF_TOPIC_KEYWORDS)
    return GuardrailFunctionOutput(
        output_info=f"Off-topic query blocked: {text[:60]}",
        tripwire_triggered=is_off_topic,
    )


@output_guardrail
async def response_quality_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output,
) -> GuardrailFunctionOutput:
    """Flag responses that are too short to be useful."""
    text = output.final_output if hasattr(output, "final_output") else str(output)
    is_too_short = len(text.strip()) < 20
    return GuardrailFunctionOutput(
        output_info=f"Response length: {len(text)} chars",
        tripwire_triggered=is_too_short,
    )


def _build_agents(vector_store_id: str | None) -> Agent:
    faq_tools = [FileSearchTool(vector_store_ids=[vector_store_id])] if vector_store_id else [faq_lookup_tool]

    faq_specialist = Agent[BookingContext](
        name="FAQ Specialist",
        handoff_description=(
            "Answers general questions about baggage, seat counts, seating "
            "information, wifi, check-in, and policies."
        ),
        instructions=(
            "You are an IndiGo Airlines customer service agent. "
            "Always use the available tools to answer questions — never rely on your own knowledge."
        ),
        model="gpt-4o-mini",
        tools=faq_tools,
        input_guardrails=[airline_topic_guardrail],
        output_guardrails=[response_quality_guardrail],
    )

    seat_specialist = Agent[BookingContext](
        name="Seat Booking Specialist",
        handoff_description=(
            "Handles explicit requests to change, select, move, or update a "
            "customer's seat for an existing booking."
        ),
        instructions=(
            "You are a seat booking agent for Indigo Airlines. "
            "Only call update_seat when the latest customer message explicitly asks "
            "to change, select, move, or update a seat. "
            "Do not update seats for informational questions like how many seats "
            "are on a flight. If confirmation number or new seat is missing, ask "
            "for the missing information."
        ),
        model="gpt-4o-mini",
        tools=[update_seat],
    )

    triage_agent = Agent[BookingContext](
        name="Triage Agent",
        instructions=(
            "You are the first point of contact for customer service for Indigo Airlines. "
            "Route based only on the latest customer request. "
            "Use FAQ Specialist for informational questions, including questions "
            "about how many seats there are or seating policy. "
            "Use Seat Booking Specialist only when the customer explicitly wants "
            "to change, select, move, or update their seat."
        ),
        model="gpt-4o-mini",
        handoffs=[
            handoff(faq_specialist),
            handoff(seat_specialist),
        ],
        input_guardrails=[airline_topic_guardrail],
    )

    return triage_agent


def get_triage_agent() -> Agent:
    return _build_agents(_vector_store_id)


def _init_vector_store_sync(pdf_path: str) -> str:
    client = OpenAI()

    with open(pdf_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="assistants")

    vector_store = client.vector_stores.create(name="IndiGo FAQ")
    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id,
    )

    import time
    while True:
        files = client.vector_stores.files.list(vector_store_id=vector_store.id).data
        if files and files[0].status == "completed":
            break
        time.sleep(1)

    return vector_store.id


async def init_vector_store(pdf_path: str) -> None:
    global _vector_store_id
    store_id = await asyncio.to_thread(_init_vector_store_sync, pdf_path)
    _vector_store_id = store_id
    print(f"IndiGo FAQ vector store ready: {_vector_store_id}")


async def run_chat(
    user_message: str,
    history: list[dict],
    context: BookingContext,
) -> dict:
    triage = get_triage_agent()
    context.seat_update_requested = (
        context.seat_update_requested or _is_seat_update_request(user_message)
    )
    input_messages = [{"role": "user", "content": user_message}]

    handoffs_seen: list[str] = []
    final_reply = ""
    last_agent = "Triage Agent"
    blocked = False

    try:
        result = await Runner.run(triage, input_messages, context=context)

        for item in result.new_items:
            if isinstance(item, HandoffOutputItem):
                handoffs_seen.append(
                    f"{item.source_agent.name} → {item.target_agent.name}"
                )
                last_agent = item.target_agent.name
            elif isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    final_reply = text
                    last_agent = item.agent.name

        new_history = result.to_input_list()

    except InputGuardrailTripwireTriggered:
        final_reply = "I'm sorry, I can only help with airline-related questions. Please ask about flights, baggage, check-in, or bookings."
        blocked = True
        new_history = history
    except OutputGuardrailTripwireTriggered:
        final_reply = "I wasn't able to generate a complete answer. Please try rephrasing your question."
        blocked = True
        new_history = history

    return {
        "reply": final_reply,
        "agent": last_agent,
        "handoffs": handoffs_seen,
        "blocked": blocked,
        "history": new_history,
        "context": context,
    }
