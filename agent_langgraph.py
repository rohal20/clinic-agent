"""Clinic Appointment Agent, built with LangChain + LangGraph.

Same features as agent_plain.py. What changes:
  LangChain: ChatOllama (the model), @tool (tools and their schemas), a text splitter,
             embeddings, a vector store and a retriever (RAG)
  LangGraph: a graph (agent node -> tools node -> agent), interrupt() for the patient's
             confirmation, and a checkpointer for memory

Run:  python agent_langgraph.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
MIN_HOURS_BEFORE_CHANGE = 24  # appointments can be changed only this many hours before

# ------------------------------------------------------------------
# 1. Clinic data (fake, kept in memory) - same as agent_plain.py
# ------------------------------------------------------------------
DOCTORS = {
    "D1": {"name": "Dr. Sarah Khan", "specialty": "general", "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]},
    "D2": {"name": "Dr. Omar Malik", "specialty": "dermatology", "days": ["Mon", "Wed", "Thu"]},
    "D3": {"name": "Dr. Ayesha Raza", "specialty": "pediatrics", "days": ["Tue", "Thu", "Sat"]},
}
TIMES = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BOOKINGS = []
SHOWN_SLOTS = set()  # (doctor_id, date, time) slots the patient has been shown


def to_date(day: str) -> date:
    day = day.strip().lower()
    if day == "today":
        return date.today()
    if day == "tomorrow":
        return date.today() + timedelta(days=1)
    return date.fromisoformat(day)


def free_times(doctor_id: str, d: date) -> list[str]:
    """Times this doctor is free on this date: not booked, and not already past."""
    doctor = DOCTORS[doctor_id]
    if WEEKDAYS[d.weekday()] not in doctor["days"]:
        return []
    taken = [b["time"] for b in BOOKINGS if b["doctor_id"] == doctor_id and b["date"] == d.isoformat()]
    now = datetime.now()
    free = []
    for t in TIMES:
        if t in taken:
            continue  # already booked
        if d == now.date() and t <= now.strftime("%H:%M"):
            continue  # already past today
        free.append(t)
    return free


# ------------------------------------------------------------------
# 2. RAG with LangChain: split -> embed -> store -> retrieve
# ------------------------------------------------------------------
faq_text = Path(__file__).with_name("faq.md").read_text(encoding="utf-8")
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")], strip_headers=False)
chunks = splitter.split_text(faq_text)                              # one Document per "## " section
vector_store = InMemoryVectorStore(OllamaEmbeddings(model=EMBED_MODEL))
vector_store.add_documents(chunks)                                  # embeds every chunk once
retriever = vector_store.as_retriever(search_kwargs={"k": 2})       # returns the top-2 chunks


# ------------------------------------------------------------------
# 3. Tools: @tool turns each function + docstring into a schema for the model
# ------------------------------------------------------------------
def ask_patient_to_confirm(question: str) -> bool:
    """Human-in-the-loop with LangGraph: interrupt() pauses the whole graph here.
    The app shows the question, and the patient's answer comes back as the return value."""
    answer = interrupt(question)
    return str(answer).strip().lower() in {"yes", "y"}


@tool
def search_clinic_info(question: str) -> str:
    """Search clinic information: fees, opening hours, location, payment and cancellation policy."""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def list_doctors() -> str:
    """List all doctors with their specialty (category) and working days."""
    lines = []
    for doctor_id, doctor in DOCTORS.items():
        lines.append(f"{doctor['specialty'].title()}: {doctor['name']} (id {doctor_id}), "
                     f"works {', '.join(doctor['days'])}")
    return "\n".join(lines)


@tool
def find_available_slots(specialty: Literal["general", "dermatology", "pediatrics"], day: str) -> str:
    """Find free appointment times for a specialty. day is 'today', 'tomorrow' or YYYY-MM-DD."""
    d = to_date(day)
    lines = [f"Date: {d:%A} {d.isoformat()}"]
    for doctor_id, doctor in DOCTORS.items():
        if doctor["specialty"] == specialty:
            times = free_times(doctor_id, d)
            lines.append(f"{doctor['name']} (id {doctor_id}): {', '.join(times) or 'not available'}")
            SHOWN_SLOTS.update((doctor_id, d.isoformat(), t) for t in times)  # remember what was shown
    return "\n".join(lines)


@tool
def book_appointment(doctor_id: str, day: str, time: str, patient_name: str) -> str:
    """Book an appointment. Use only after the patient has picked a doctor and a time from
    find_available_slots and given their name. day is YYYY-MM-DD, time is HH:MM."""
    if doctor_id not in DOCTORS:
        return f"Error: unknown doctor id {doctor_id}. Use list_doctors."
    d = to_date(day)
    if (doctor_id, d.isoformat(), time) not in SHOWN_SLOTS:
        return "First show the patient the available slots with find_available_slots, then let them pick one."
    if time not in free_times(doctor_id, d):
        return f"Sorry, {time} is not free. Please choose another time."
    doctor = DOCTORS[doctor_id]["name"]
    if not ask_patient_to_confirm(f"Confirm: {patient_name} with {doctor} on {d:%A %Y-%m-%d} at {time}?"):
        return "The patient did not confirm. Nothing was booked."
    booking = {"id": len(BOOKINGS) + 1, "doctor_id": doctor_id, "date": d.isoformat(),
               "time": time, "patient": patient_name}
    BOOKINGS.append(booking)
    return f"Booked {doctor} on {d:%A %Y-%m-%d} at {time} for {patient_name} (booking id {booking['id']})."


@tool
def my_appointments(patient_name: str) -> str:
    """List all appointments for a patient, with their booking ids."""
    found = [b for b in BOOKINGS if b["patient"].lower() == patient_name.strip().lower()]
    if not found:
        return f"No appointments found for {patient_name}."
    return "\n".join(f"Booking id {b['id']}: {DOCTORS[b['doctor_id']]['name']} on {b['date']} at {b['time']}"
                     for b in found)


@tool
def reschedule_appointment(booking_id: int, new_day: str, new_time: str) -> str:
    """Change an appointment to a new day and time with the same doctor. Use only after showing
    free times with find_available_slots. new_day is YYYY-MM-DD, new_time is HH:MM."""
    booking = next((b for b in BOOKINGS if b["id"] == int(booking_id)), None)
    if booking is None:
        return f"Error: no booking found with id {booking_id}."
    current = datetime.strptime(f"{booking['date']} {booking['time']}", "%Y-%m-%d %H:%M")
    if current - datetime.now() < timedelta(hours=MIN_HOURS_BEFORE_CHANGE):
        return f"Sorry, appointments can only be changed up to {MIN_HOURS_BEFORE_CHANGE} hours before."
    doctor_id = booking["doctor_id"]
    d = to_date(new_day)
    if (doctor_id, d.isoformat(), new_time) not in SHOWN_SLOTS:
        return "First show the patient the available slots with find_available_slots, then let them pick one."
    if new_time not in free_times(doctor_id, d):
        return f"Sorry, {new_time} is not free. Please choose another time."
    doctor = DOCTORS[doctor_id]["name"]
    if not ask_patient_to_confirm(f"Move booking {booking_id} with {doctor} from {booking['date']} "
                                  f"{booking['time']} to {d.isoformat()} {new_time}?"):
        return "The patient did not confirm. Nothing was changed."
    booking["date"], booking["time"] = d.isoformat(), new_time
    return f"Rescheduled booking {booking_id} to {d:%A %Y-%m-%d} at {new_time} with {doctor}."


TOOLS = [search_clinic_info, list_doctors, find_available_slots, book_appointment,
         my_appointments, reschedule_appointment]

# ------------------------------------------------------------------
# 4. The graph:  START -> agent -> (tool calls?) -> tools -> agent ... -> END
# ------------------------------------------------------------------
llm = ChatOllama(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)  # the model now knows the tools

DOCTOR_LIST = "\n".join(
    f"- {doc['specialty'].title()}: {doc['name']} (id {doc_id}), works {', '.join(doc['days'])}"
    for doc_id, doc in DOCTORS.items()
)

SYSTEM_PROMPT = f"""You are the appointment assistant for City Care Clinic (a demo clinic).
Today is {date.today():%A %Y-%m-%d}.

The clinic has ONLY these doctors:
{DOCTOR_LIST}
Never mention any other doctor or specialty.

To book an appointment, follow these steps in order:
1. If the patient hasn't chosen a category, call list_doctors and show the categories with
   the doctor names. Ask which one they want.
2. Ask which day, if they haven't said.
3. Call find_available_slots and show the free times. Ask which time they want.
4. Ask for the patient's name, if you don't know it.
5. Call book_appointment. The patient will be asked to confirm.

To change an appointment:
1. Ask for the patient's name and call my_appointments to find the booking id.
2. Ask for the new day, call find_available_slots for that doctor's category, show the times.
3. Call reschedule_appointment. Changes are allowed only up to {MIN_HOURS_BEFORE_CHANGE} hours before.

Other rules:
- For fees, hours or policies, call search_clinic_info. Never make them up.
- Do not give medical advice.
- Keep answers short and friendly."""


def agent(state: MessagesState):
    """Agent node: the LLM reads the conversation and either replies or asks for tools."""
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)                        # state = the list of messages
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))  # runs the requested tools
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)    # tool calls -> "tools", otherwise -> END
builder.add_edge("tools", "agent")                         # after tools, back to the agent (the loop)
graph = builder.compile(checkpointer=InMemorySaver())      # checkpointer = memory per thread_id


# ------------------------------------------------------------------
# 5. Chat in the terminal
# ------------------------------------------------------------------
def run(payload, config) -> str | None:
    """Run the graph, printing each tool call. Returns the confirmation question if it paused."""
    for update in graph.stream(payload, config, stream_mode="updates"):  # one update per node
        if "__interrupt__" in update:                 # a tool called interrupt(): the graph paused
            return update["__interrupt__"][0].value
        if "agent" in update:                         # show which tools the LLM asked for
            for call in update["agent"]["messages"][-1].tool_calls:
                print(f"   [tool] {call['name']} {call['args']}")
    return None


def main():
    config = {"configurable": {"thread_id": "patient-1"}}
    print("City Care Clinic assistant. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        question = run({"messages": [{"role": "user", "content": text}]}, config)
        while question:  # the graph is paused, waiting for the patient's yes/no
            answer = input(f"   {question} (yes/no): ")
            question = run(Command(resume=answer), config)
        print("Assistant:", graph.get_state(config).values["messages"][-1].content)


if __name__ == "__main__":
    main()
