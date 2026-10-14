"""Clinic Appointment Agent, built with LangChain + LangGraph.

  LangChain: ChatOllama (the model), @tool (tools and their schemas), a text splitter,
             embeddings, a vector store and a retriever (RAG)
  LangGraph: a graph (agent node -> tools node -> agent or END), interrupt() for the
             patient's confirmation, and a checkpointer for memory

Run:  python agent_langgraph.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

CLINIC_NAME = "City Care Clinic"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
MIN_HOURS_BEFORE_CHANGE = 24  # appointments can be changed only this many hours before
SHOW_STEPS = True             # print each tool call (handy for demos and debugging)

# ------------------------------------------------------------------
# 1. Clinic data (fake, kept in memory)
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


WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_FORMATS = ["%Y-%m-%d", "%A, %d %B %Y", "%A %d %B %Y", "%d %B %Y", "%d %B"]


def to_date(day: str) -> date:
    """Turn the patient's words into a real date. The code does date maths, not the LLM.
    Accepts: today, tomorrow, day after tomorrow, a weekday ('friday', 'next fri'),
    or a date ('2026-10-16', '16 October 2026')."""
    text = day.strip().lower().removeprefix("next ").removeprefix("this ").strip()
    today = date.today()
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    if text == "day after tomorrow":
        return today + timedelta(days=2)
    for index, name in enumerate(WEEKDAY_NAMES):
        if text in (name, name[:3]):
            return today + timedelta(days=(index - today.weekday()) % 7)  # the coming one (today counts)
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(day.strip(), fmt).date()
            return parsed.replace(year=today.year) if fmt == "%d %B" else parsed
        except ValueError:
            continue
    raise ValueError(f"'{day}' is not a valid day. Use today, tomorrow, a weekday like friday, "
                     "or a date like 2026-10-16.")


def find_doctor(doctor: str) -> str | None:
    """Accept a doctor id ('D2') or a name ('Dr. Omar Malik', 'omar malik'). Returns the id."""
    key = doctor.strip().lower().removeprefix("dr.").strip()
    for doctor_id, info in DOCTORS.items():
        if key == doctor_id.lower() or key in info["name"].lower():
            return doctor_id
    return None


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
def build_retriever():
    faq_text = Path(__file__).with_name("faq.md").read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")], strip_headers=False)
    chunks = splitter.split_text(faq_text)                          # one Document per "## " section
    store = InMemoryVectorStore(OllamaEmbeddings(model=EMBED_MODEL))
    store.add_documents(chunks)                                     # embeds every chunk once
    return store.as_retriever(search_kwargs={"k": 2})               # returns the top-2 chunks


try:
    retriever = build_retriever()
except Exception as error:  # usually: Ollama isn't running, or the embedding model isn't pulled
    sys.exit(f"Could not start the assistant: {error}\n"
             f"Check that Ollama is running, then run:  ollama pull {EMBED_MODEL}")


# ------------------------------------------------------------------
# 3. Tools
#    return_direct=True means the tool's text goes straight to the patient,
#    so important lists can't be dropped or changed by the model.
# ------------------------------------------------------------------
def ask_patient_to_confirm(question: str) -> bool:
    """Human-in-the-loop: interrupt() pauses the graph until the patient answers."""
    answer = interrupt(question)
    return str(answer).strip().lower() in {"yes", "y"}


@tool
def search_clinic_info(question: str) -> str:
    """Search clinic information: fees, opening hours, location, payment and cancellation policy."""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


@tool(return_direct=True)
def list_doctors() -> str:
    """Show the patient all categories (specialties) with their doctors and working days."""
    lines = ["We have the following categories:"]
    for number, doctor in enumerate(DOCTORS.values(), start=1):
        lines.append(f"  {number}. {doctor['specialty'].title()} - {doctor['name']} "
                     f"({', '.join(doctor['days'])})")
    lines.append("Which category would you like to book?")
    return "\n".join(lines)


@tool(return_direct=True)
def find_available_slots(specialty: Literal["general", "dermatology", "pediatrics"], day: str) -> str:
    """Show the patient the free appointment times for a specialty on a day.
    day: pass the patient's own words, e.g. 'friday', 'tomorrow', 'today' or a date.
    Do not convert weekdays to dates yourself."""
    d = to_date(day)
    lines = [f"Available times on {d:%A, %d %B %Y} ({d.isoformat()}):"]
    any_free = False
    for doctor_id, doctor in DOCTORS.items():
        if doctor["specialty"] != specialty:
            continue
        times = free_times(doctor_id, d)
        if times:
            any_free = True
            lines.append(f"  {doctor['name']}: {', '.join(times)}")
            SHOWN_SLOTS.update((doctor_id, d.isoformat(), t) for t in times)  # remember what was shown
        else:
            lines.append(f"  {doctor['name']}: not available (works {', '.join(doctor['days'])})")
    lines.append("Which time would you like?" if any_free else "Which other day would suit you?")
    return "\n".join(lines)


@tool
def book_appointment(doctor: str, day: str, time: str, patient_name: str) -> str:
    """Book an appointment after the patient has picked a time from find_available_slots and
    given their name. doctor is the doctor's id or name, day is YYYY-MM-DD, time is HH:MM."""
    doctor_id = find_doctor(doctor)
    if doctor_id is None:
        return f"Error: unknown doctor '{doctor}'."
    d = to_date(day)
    if (doctor_id, d.isoformat(), time) not in SHOWN_SLOTS:
        return "First show the patient the available slots with find_available_slots, then let them pick one."
    if time not in free_times(doctor_id, d):
        return f"Sorry, {time} is no longer free. Please choose another time."
    name = DOCTORS[doctor_id]["name"]
    if not ask_patient_to_confirm(f"Confirm: {patient_name} with {name} on {d:%A, %d %B} at {time}?"):
        return "The patient did not confirm. Nothing was booked."
    booking = {"id": len(BOOKINGS) + 1, "doctor_id": doctor_id, "date": d.isoformat(),
               "time": time, "patient": patient_name}
    BOOKINGS.append(booking)
    return f"Booked: {name} on {d:%A, %d %B %Y} at {time} for {patient_name}. Booking id {booking['id']}."


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
    """Move an appointment to a new day and time with the same doctor, after showing free times
    with find_available_slots. new_day is the patient's words (e.g. 'friday') or YYYY-MM-DD,
    new_time is HH:MM."""
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
    name = DOCTORS[doctor_id]["name"]
    if not ask_patient_to_confirm(f"Move booking {booking_id} with {name} from {booking['date']} "
                                  f"{booking['time']} to {d.isoformat()} {new_time}?"):
        return "The patient did not confirm. Nothing was changed."
    booking["date"], booking["time"] = d.isoformat(), new_time
    return f"Rescheduled booking {booking_id}: {name} on {d:%A, %d %B %Y} at {new_time}."


TOOLS = [search_clinic_info, list_doctors, find_available_slots, book_appointment,
         my_appointments, reschedule_appointment]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# ------------------------------------------------------------------
# 4. The graph:  START -> agent -> tools -> (agent | END)
# ------------------------------------------------------------------
llm = ChatOllama(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)

# Built from DOCTORS, so the prompt always matches the real data (grounding, no invented doctors)
DOCTOR_LIST = "\n".join(f"- {d['specialty']}: {d['name']} (id {i}), works {', '.join(d['days'])}"
                        for i, d in DOCTORS.items())

# The next 7 days, worked out by code, so the LLM never has to calculate a date itself
NEXT_7_DAYS = "\n".join(f"- {date.today() + timedelta(days=n):%A} = {date.today() + timedelta(days=n)}"
                        for n in range(7))

SYSTEM_PROMPT = f"""You are the appointment assistant for {CLINIC_NAME}.
Today is {date.today():%A %Y-%m-%d}.

The clinic has ONLY these doctors. Never mention any other doctor or specialty:
{DOCTOR_LIST}

Calendar for the next 7 days (use it, never calculate dates yourself):
{NEXT_7_DAYS}
When calling find_available_slots, pass the day in the patient's own words (e.g. "friday").

Booking steps, in order:
1. If the patient hasn't chosen a category, call list_doctors.
2. If they haven't said which day, ask.
3. Call find_available_slots to show the free times.
4. When they pick a time, ask for their name if you don't know it.
5. Call book_appointment. The patient will be asked to confirm.

Changing an appointment:
1. Ask for their name and call my_appointments to find the booking id.
2. Ask for the new day and call find_available_slots for that doctor's specialty.
3. Call reschedule_appointment. Changes are allowed only up to {MIN_HOURS_BEFORE_CHANGE} hours before.

Rules:
- For fees, hours or policies, call search_clinic_info. Never make them up.
- Do not give medical advice.
- Be polite, professional and brief."""


def agent(state: MessagesState):
    """Agent node: the LLM reads the conversation and either replies or asks for tools."""
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def after_tools(state: MessagesState):
    """Conditional edge after the tools node: if every tool that just ran is return_direct
    (and succeeded), its text is the reply, so we end. Otherwise the agent continues."""
    recent = []
    for message in reversed(state["messages"]):
        if message.type != "tool":
            break
        recent.append(message)
    if recent and all(TOOLS_BY_NAME[m.name].return_direct and m.status != "error" for m in recent):
        return END
    return "agent"


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)          # tool calls -> tools, else -> END
builder.add_conditional_edges("tools", after_tools, ["agent", END])
graph = builder.compile(checkpointer=InMemorySaver())             # memory per thread_id


# ------------------------------------------------------------------
# 5. Chat in the terminal
# ------------------------------------------------------------------
def run(payload, config) -> str | None:
    """Run the graph. Returns the confirmation question if a tool paused it, else None."""
    for update in graph.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in update:
            return update["__interrupt__"][0].value
        if SHOW_STEPS and "agent" in update:
            for call in update["agent"]["messages"][-1].tool_calls:
                print(f"   [tool] {call['name']} {call['args']}")
    return None


def main():
    config = {"configurable": {"thread_id": "patient-1"}}
    print("=" * 60)
    print(f"  {CLINIC_NAME} - Appointment Assistant")
    print("=" * 60)
    print("Hello! I can book or change an appointment, or answer questions")
    print("about fees, timings and clinic policies. Type 'quit' to exit.\n")
    print(list_doctors.invoke({}).replace("Which category would you like to book?", "").strip())

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit", "bye"}:
            print(f"Assistant: Thank you for contacting {CLINIC_NAME}. Take care!")
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