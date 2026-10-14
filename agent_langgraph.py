"""Clinic Appointment Assistant, built with LangChain + LangGraph.

Design: the LLM understands the patient; the graph runs the steps.
  LangChain: ChatOllama, structured output (Pydantic), RAG (splitter, embeddings,
             vector store, retriever)
  LangGraph: custom state, nodes, conditional edges, interrupt() for confirmation,
             and a checkpointer for memory

Why not a free tool-calling loop? With a small local model (llama3.1:8b) the model kept
calling the wrong tool, ignoring tool results and getting dates wrong. Booking is a known
step-by-step process, so the graph runs the steps, and the LLM does what it is good at:
understanding what the patient wants and answering questions from the clinic's documents.

Run:  python agent_langgraph.py
"""

import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

CLINIC_NAME = "City Care Clinic"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
MIN_HOURS_BEFORE_CHANGE = 24  # appointments can be changed only this many hours before
SHOW_STEPS = True             # print what the assistant understood (handy for demos)

# ------------------------------------------------------------------
# 1. Clinic data (fake, kept in memory). One doctor per category.
# ------------------------------------------------------------------
DOCTORS = {
    "D1": {"name": "Dr. Sarah Khan", "specialty": "general", "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]},
    "D2": {"name": "Dr. Omar Malik", "specialty": "dermatology", "days": ["Mon", "Wed", "Thu"]},
    "D3": {"name": "Dr. Ayesha Raza", "specialty": "pediatrics", "days": ["Tue", "Thu", "Sat"]},
}
TIMES = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BOOKINGS = []


def doctor_for(specialty: str) -> str:
    return next(doctor_id for doctor_id, d in DOCTORS.items() if d["specialty"] == specialty)


def free_times(doctor_id: str, d: date) -> list[str]:
    """Times this doctor is free on this date: a working day, not booked, not already past."""
    doctor = DOCTORS[doctor_id]
    if WEEKDAYS[d.weekday()] not in doctor["days"]:
        return []
    taken = [b["time"] for b in BOOKINGS if b["doctor_id"] == doctor_id and b["date"] == d.isoformat()]
    now = datetime.now()
    return [t for t in TIMES
            if t not in taken and not (d == now.date() and t <= now.strftime("%H:%M"))]


# ------------------------------------------------------------------
# 2. Reading the patient's words in code (exact, so the LLM never has to)
# ------------------------------------------------------------------
WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
SPECIALTY_WORDS = {
    "general": ["general", "physician", "gp"],
    "dermatology": ["dermatolog", "skin"],
    "pediatrics": ["pediatric", "paediatric", "child", "kid", "baby"],
}


def make_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:  # e.g. 31-02
        return None


def date_without_year(month: int, day: int) -> Optional[date]:
    """'21-10' or '21 october': this year, or next year if that date has already passed."""
    d = make_date(date.today().year, month, day)
    if d and d < date.today():
        d = make_date(date.today().year + 1, month, day)
    return d


def parse_day(text: str) -> Optional[str]:
    """Find the day the patient means and return it as YYYY-MM-DD (None if there isn't one).
    Checked in order, so an exact date wins over a weekday ("wednesday on 21-10-2026" -> 21 Oct):
      1. numeric dates: 2026-10-21, 21-10-2026, 21/10/2026, 21-10, 21/10
      2. month names:   21 october, 21st oct 2026, october 21
      3. relative:      today, tomorrow, day after tomorrow
      4. weekdays:      wednesday, next wed, wednesday next week"""
    t = text.lower()
    today = date.today()

    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        if d := make_date(int(m[1]), int(m[2]), int(m[3])):
            return d.isoformat()
    if m := re.search(r"\b(\d{1,2})[-/](\d{1,2})(?:[-/](\d{2,4}))?\b", t):
        day, month = int(m[1]), int(m[2])
        if m[3]:
            year = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
            d = make_date(year, month, day)
        else:
            d = date_without_year(month, day)
        if d:
            return d.isoformat()

    for pattern, day_group, month_group, year_group in (
            (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}(?:,?\s+(\d{{4}}))?", 1, 2, 3),
            (rf"\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", 2, 1, 3)):
        if m := re.search(pattern, t):
            month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(m[month_group])
            day = int(m[day_group])
            d = make_date(int(m[year_group]), month + 1, day) if m[year_group] else date_without_year(month + 1, day)
            if d:
                return d.isoformat()

    if "day after tomorrow" in t:
        return (today + timedelta(days=2)).isoformat()
    if re.search(r"\btoday\b", t):
        return today.isoformat()
    if re.search(r"\btomorrow\b", t):
        return (today + timedelta(days=1)).isoformat()

    for m in re.finditer(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", t):
        index = next((i for i, name in enumerate(WEEKDAY_NAMES) if name.startswith(m[0])), None)
        if index is None:
            continue  # e.g. "month" starts like "mon" but isn't a day
        if "next week" in t:  # that weekday in the following Monday-Sunday week
            next_monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
            return (next_monday + timedelta(days=index)).isoformat()
        days_ahead = (index - today.weekday()) % 7
        if days_ahead == 0 and not re.search(rf"\bthis {m[0]}", t):
            days_ahead = 7  # "wednesday" said on a Wednesday means next Wednesday
        return (today + timedelta(days=days_ahead)).isoformat()
    return None


def parse_time(text: str, awaiting: Optional[str]) -> Optional[str]:
    t = text.lower()
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", t)
    if m:
        hour, minute, ampm = int(m.group(1)), m.group(2), m.group(3)
    elif m := re.search(r"\b(\d{1,2})\s*(am|pm)\b", t):
        hour, minute, ampm = int(m.group(1)), "00", m.group(2)
    elif awaiting == "time" and (m := re.fullmatch(r"\s*(\d{1,2})\s*", t)):
        hour, minute, ampm = int(m.group(1)), "00", None
    else:
        return None
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}"


def parse_specialty(text: str, awaiting: Optional[str]) -> Optional[str]:
    t = text.lower()
    for specialty, words in SPECIALTY_WORDS.items():
        if any(word in t for word in words):
            return specialty
    if awaiting == "specialty" and (m := re.fullmatch(r"\s*(\d)\s*\.?\s*", t)):  # "2" = 2nd category
        number = int(m.group(1))
        if 1 <= number <= len(DOCTORS):
            return list(DOCTORS.values())[number - 1]["specialty"]
    return None


NOT_NAMES = {"appointment", "appointments", "book", "booking", "change", "show", "what", "when", "how"}


def parse_name(text: str) -> Optional[str]:
    t = re.sub(r"^(my name is|my name's|name is|i am|i'm|it's|it is|this is)\s+", "", text.strip(), flags=re.I)
    words = re.sub(r"[^A-Za-z .'-]", "", t).split()
    if not words or any(w.lower() in NOT_NAMES for w in words):
        return None
    return " ".join(w.capitalize() for w in words[:4])


def parse_booking_id(text: str) -> Optional[int]:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def clean_llm_name(name: Optional[str], text: str) -> Optional[str]:
    """Accept the LLM's name only if it's a real name the patient actually typed.
    (Small models sometimes return the text "None" or invent a name.)"""
    if not name or name.strip().lower() in {"none", "null", "unknown", "n/a", "patient", "the patient"}:
        return None
    if name.strip().lower() not in text.lower():
        return None
    return " ".join(w.capitalize() for w in name.split())


def keyword_intent(text: str) -> str:
    """Backup if the LLM call fails: simple keyword rules."""
    t = text.lower()
    if re.search(r"\b(reschedule|change|move|shift)\b", t):
        return "reschedule"
    if re.search(r"\b(my (appointment|booking)s?|already booked|booking id|is it confirmed)\b", t):
        return "my_appointments"  # asking about an existing appointment, not booking a new one
    if re.search(r"\b(book|appointment)\b", t):
        return "book"
    if "?" in t or re.search(r"\b(fees?|charges?|cost|price|hours?|timings?|open|where|address|payment|pay|policy)\b", t):
        return "question"
    if re.search(r"\b(hi|hello|hey|salam|thanks|thank you)\b", t):
        return "greeting"
    return "other"


# ------------------------------------------------------------------
# 3. RAG with LangChain: split -> embed -> store -> retrieve
# ------------------------------------------------------------------
def build_retriever():
    faq_text = Path(__file__).with_name("faq.md").read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")], strip_headers=False)
    chunks = splitter.split_text(faq_text)                          # one Document per "## " section
    store = InMemoryVectorStore(OllamaEmbeddings(model=EMBED_MODEL))
    store.add_documents(chunks)                                     # embeds every chunk once
    return store.as_retriever(search_kwargs={"k": 3})               # returns the top-3 chunks


try:
    retriever = build_retriever()
except Exception as error:  # usually: Ollama isn't running, or the embedding model isn't pulled
    sys.exit(f"Could not start the assistant: {error}\n"
             f"Check that Ollama is running, then run:  ollama pull {EMBED_MODEL}")


# ------------------------------------------------------------------
# 4. The LLM's two jobs: understand the message, and answer questions
# ------------------------------------------------------------------
class Understanding(BaseModel):
    """What the patient wants in their latest message."""
    intent: Literal["book", "reschedule", "my_appointments", "question", "greeting", "other"] = Field(
        description="book = wants a new appointment or is answering a booking question; "
                    "reschedule = wants to change an existing appointment; "
                    "my_appointments = wants to see their appointments; "
                    "question = asks about the clinic (fees, hours, location, payment, policies, doctors); "
                    "greeting = hello, thanks or goodbye; other = anything else")
    patient_name: Optional[str] = Field(None, description="The patient's name, only if they say it")


chat_llm = ChatOllama(model=CHAT_MODEL, temperature=0)
understanding_llm = chat_llm.with_structured_output(Understanding)  # always returns an Understanding

DOCTOR_LIST = "\n".join(f"- {d['specialty'].title()}: {d['name']}, works {', '.join(d['days'])}"
                        for d in DOCTORS.values())
TASKS = {"book", "reschedule", "my_appointments"}
DONE = {"task": None, "awaiting": None, "specialty": None, "day": None, "time": None, "booking_id": None}
PENDING_QUESTIONS = {
    "specialty": "Which category would you like to book?",
    "day": "Which day would you like?",
    "time": "Which time would you like?",
    "patient_name": "May I have the patient's name?",
    "booking_id": "Which booking id would you like to change?",
}


def understand_with_llm(text: str, state: dict) -> Understanding:
    last_reply = next((m.content for m in reversed(state["messages"][:-1]) if m.type == "ai"), "")
    prompt = (f"You classify messages for the appointment assistant of {CLINIC_NAME}.\n"
              f"The assistant's previous message was: \"{last_reply}\"\n"
              f"Current task: {state.get('task') or 'none'}. "
              f"The assistant is waiting for: {state.get('awaiting') or 'nothing'}.\n"
              "Choose the intent of the patient's latest message, and give the patient's name "
              "only if they say it.")
    try:
        return understanding_llm.invoke([SystemMessage(prompt), HumanMessage(text)])
    except Exception:  # if the LLM fails or returns something invalid, fall back to keywords
        return Understanding(intent=keyword_intent(text))


# ------------------------------------------------------------------
# 5. The graph state: the conversation + what the assistant has collected so far
# ------------------------------------------------------------------
class ClinicState(MessagesState):
    route: str                     # which node handles this message
    task: Optional[str]            # the job in progress: book / reschedule / my_appointments
    awaiting: Optional[str]        # what we last asked the patient for
    specialty: Optional[str]
    day: Optional[str]             # YYYY-MM-DD
    time: Optional[str]            # HH:MM
    patient_name: Optional[str]
    booking_id: Optional[int]


def say(text: str, **updates) -> dict:
    """A node's update: the assistant's reply plus any state changes."""
    return {"messages": [AIMessage(text)], **updates}


def categories_text() -> str:
    lines = ["We have the following categories:"]
    lines += [f"  {n}. {d['specialty'].title()} - {d['name']} ({', '.join(d['days'])})"
              for n, d in enumerate(DOCTORS.values(), start=1)]
    return "\n".join(lines)


def slots_text(doctor_name: str, d: date, times: list[str]) -> str:
    return (f"Available times with {doctor_name} on {d:%A, %d %B %Y}:\n  {', '.join(times)}\n"
            "Which time would you like?")


def confirmed(answer) -> bool:
    return str(answer).strip().lower() in {"yes", "y", "yes please", "confirm", "ok", "okay"}


# ------------------------------------------------------------------
# 6. Nodes
# ------------------------------------------------------------------
def understand(state: ClinicState):
    """Node 1: work out what the patient wants, and collect any details they gave."""
    text = state["messages"][-1].content
    task, awaiting = state.get("task"), state.get("awaiting")
    understanding = understand_with_llm(text, state)
    intent = understanding.intent
    backup = keyword_intent(text)
    # If the LLM missed an obvious request (e.g. "book an appointment"), trust the keywords,
    # unless the patient is clearly asking a question.
    if intent not in TASKS and backup in TASKS and not (intent == "question" and "?" in text):
        intent = backup
    if intent == "other":  # the LLM wasn't sure: let the keywords decide (e.g. a question about fees)
        intent = backup
    if backup == "my_appointments" and intent != "reschedule":
        intent = "my_appointments"  # "my appointment is confirmed right?" is about an existing booking

    name = clean_llm_name(understanding.patient_name, text)
    starting_other_task = intent in TASKS and intent != task
    if not name and awaiting == "patient_name" and not starting_other_task and intent != "question" and "?" not in text:
        name = parse_name(text)
    found = {
        "specialty": parse_specialty(text, awaiting),
        "day": parse_day(text),
        "time": parse_time(text, awaiting),
        "patient_name": name,
        "booking_id": parse_booking_id(text) if awaiting == "booking_id" else None,
    }
    details = {key: value for key, value in found.items() if value}

    if intent in TASKS and intent != task:                 # a new request: start fresh
        fresh = {**DONE, "task": intent, "route": intent}
        if intent == "book":
            fresh["patient_name"] = None  # a new booking may be for someone else: ask again
        return {**fresh, **details}
    if task and intent not in ("question", "greeting"):    # answering our question (even unclearly): carry on
        return {**details, "route": task}
    return {"route": "answer"}                              # a question, greeting or anything else


def answer(state: ClinicState):
    """Node 2 (RAG): answer from the clinic's documents, then remind them of any pending step."""
    question = state["messages"][-1].content
    context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
    prompt = (f"You are the friendly assistant of {CLINIC_NAME}. Today is {date.today():%A %d %B %Y}.\n"
              "Answer using ONLY the doctor list and clinic information below. If the answer is not "
              "there, say you don't know and suggest calling the reception. Never mention other "
              "doctors. Do not give medical advice. You can also book or change appointments: if the "
              "patient wants that, tell them to just ask. Keep it short.\n\n"
              f"Doctors:\n{DOCTOR_LIST}\n\n"
              f"This patient's appointments: {bookings_text(state.get('patient_name')) or 'none yet'}\n\n"
              f"Clinic information:\n{context}")
    reply = chat_llm.invoke([SystemMessage(prompt)] + state["messages"][-6:]).content
    pending = PENDING_QUESTIONS.get(state.get("awaiting"))
    if state.get("task") and pending:
        reply += f"\n\n{pending}"
    return say(reply)


def day_question(doctor: dict, state: dict, new: bool = False) -> str:
    """Ask for a day. If we already asked and couldn't read the answer, say so and give examples."""
    unclear = ("Sorry, I couldn't understand that day. You can say a weekday (e.g. Monday), "
               "'tomorrow', or a date like 21-10-2026.\n") if state.get("awaiting") == "day" else ""
    which = "Which new day would you like?" if new else "Which day would you like?"
    return f"{unclear}{doctor['name']} works on {', '.join(doctor['days'])}. {which}"


def time_note(time: Optional[str], state: dict) -> str:
    """Explain a problem with the chosen time, before showing the free times again."""
    if time:
        return f"Sorry, {time} is not available. "
    if state.get("awaiting") == "time":
        return "Sorry, I didn't catch the time. Please pick one of these (e.g. 10:30).\n"
    return ""


def no_times_text(doctor: dict, d: date) -> str:
    """Explain why a day has no free times, then ask for another day."""
    if d < date.today():
        reason = f"{d:%A, %d %B} has already passed."
    elif WEEKDAYS[d.weekday()] not in doctor["days"]:
        reason = f"{doctor['name']} doesn't work on {d:%A}s."
    elif d == date.today():
        reason = f"There are no free times left today with {doctor['name']}."
    else:
        reason = f"{doctor['name']} is fully booked on {d:%A, %d %B}."
    return f"Sorry, {reason} They work on {', '.join(doctor['days'])}. Which other day would you like?"


def book(state: ClinicState):
    """Node 3: the booking steps, in order. Each missing detail = one question."""
    specialty, day, time, name = (state.get(k) for k in ("specialty", "day", "time", "patient_name"))
    if not specialty:
        return say(categories_text() + "\nWhich category would you like to book?", awaiting="specialty")

    doctor_id = doctor_for(specialty)
    doctor = DOCTORS[doctor_id]
    if not day:
        return say(day_question(doctor, state), awaiting="day")

    d = date.fromisoformat(day)
    times = free_times(doctor_id, d) if d >= date.today() else []
    if not times:
        return say(no_times_text(doctor, d), day=None, time=None, awaiting="day")
    if time not in times:
        return say(time_note(time, state) + slots_text(doctor["name"], d, times), time=None, awaiting="time")
    if not name:
        return say("May I have the patient's name?", awaiting="patient_name")

    # Human-in-the-loop: pause the graph until the patient answers yes/no.
    # Everything above is read-only, so it is safe to run again when the graph resumes.
    if not confirmed(interrupt(f"Confirm: {name} with {doctor['name']} on {d:%A, %d %B %Y} at {time}?")):
        return say("No problem, nothing was booked. Tell me if you'd like another time.", **DONE)
    booking = {"id": len(BOOKINGS) + 1, "doctor_id": doctor_id, "date": d.isoformat(), "time": time,
               "patient": name}
    BOOKINGS.append(booking)
    return say(f"Your appointment is booked: {doctor['name']} on {d:%A, %d %B %Y} at {time}, "
               f"for {name}. Your booking id is {booking['id']}.", **DONE)


def reschedule(state: ClinicState):
    """Node 4: change an appointment's day and time (same doctor), with the 24-hour rule."""
    name = state.get("patient_name")
    if not name:
        return say("Sure. May I have the patient's name?", awaiting="patient_name")
    mine = [b for b in BOOKINGS if b["patient"].lower() == name.lower()]
    if not mine:
        return say(f"I couldn't find any appointments for {name}.", **DONE)
    listing = "\n".join(f"  Booking id {b['id']}: {DOCTORS[b['doctor_id']]['name']} on {b['date']} at {b['time']}"
                        for b in mine)

    booking_id = state.get("booking_id") or (mine[0]["id"] if len(mine) == 1 else None)
    booking = next((b for b in mine if b["id"] == booking_id), None)
    if booking is None:
        return say(f"Your appointments:\n{listing}\nWhich booking id would you like to change?",
                   booking_id=None, awaiting="booking_id")

    current = datetime.strptime(f"{booking['date']} {booking['time']}", "%Y-%m-%d %H:%M")
    if current - datetime.now() < timedelta(hours=MIN_HOURS_BEFORE_CHANGE):
        return say(f"Sorry, booking {booking['id']} is on {booking['date']} at {booking['time']}. "
                   f"Appointments can only be changed up to {MIN_HOURS_BEFORE_CHANGE} hours before.", **DONE)

    doctor = DOCTORS[booking["doctor_id"]]
    day, time = state.get("day"), state.get("time")
    if not day:
        current_text = f"Booking {booking['id']} is with {doctor['name']} on {booking['date']} at {booking['time']}.\n"
        return say(current_text + day_question(doctor, state, new=True), booking_id=booking["id"], awaiting="day")
    d = date.fromisoformat(day)
    times = free_times(booking["doctor_id"], d) if d >= date.today() else []
    if not times:
        return say(no_times_text(doctor, d), booking_id=booking["id"], day=None, time=None, awaiting="day")
    if time not in times:
        return say(time_note(time, state) + slots_text(doctor["name"], d, times),
                   booking_id=booking["id"], time=None, awaiting="time")

    if not confirmed(interrupt(f"Move booking {booking['id']} from {booking['date']} {booking['time']} "
                               f"to {d:%A, %d %B %Y} at {time}?")):
        return say("No problem, your appointment was not changed.", **DONE)
    booking["date"], booking["time"] = d.isoformat(), time
    return say(f"Done! Booking {booking['id']} is now with {doctor['name']} on {d:%A, %d %B %Y} at {time}.", **DONE)


def bookings_text(name: Optional[str]) -> str:
    """The patient's appointments as text ('' if none)."""
    if not name:
        return ""
    mine = [b for b in BOOKINGS if b["patient"].lower() == name.lower()]
    return "\n".join(f"  Booking id {b['id']}: {DOCTORS[b['doctor_id']]['name']} on "
                     f"{date.fromisoformat(b['date']):%A, %d %B %Y} at {b['time']}" for b in mine)


def my_appointments(state: ClinicState):
    """Node 5: show a patient's confirmed appointments."""
    name = state.get("patient_name")
    if not name:
        return say("May I have the patient's name, so I can look up the appointment?", awaiting="patient_name")
    listing = bookings_text(name)
    if not listing:
        return say(f"I couldn't find any appointments for {name}. Would you like to book one?", **DONE)
    return say(f"Yes, these appointments are confirmed for {name}:\n{listing}", **DONE)


# ------------------------------------------------------------------
# 7. The graph:  START -> understand -> (answer | book | reschedule | my_appointments) -> END
# ------------------------------------------------------------------
builder = StateGraph(ClinicState)
builder.add_node("understand", understand)
builder.add_node("answer", answer)
builder.add_node("book", book)
builder.add_node("reschedule", reschedule)
builder.add_node("my_appointments", my_appointments)
builder.add_edge(START, "understand")
builder.add_conditional_edges("understand", lambda state: state["route"],
                              ["answer", "book", "reschedule", "my_appointments"])
for node in ("answer", "book", "reschedule", "my_appointments"):
    builder.add_edge(node, END)
graph = builder.compile(checkpointer=InMemorySaver())  # memory: state saved per thread_id


# ------------------------------------------------------------------
# 8. Chat in the terminal
# ------------------------------------------------------------------
def run(payload, config) -> Optional[str]:
    """Run the graph. Returns the confirmation question if a node paused it, else None."""
    for update in graph.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in update:
            return update["__interrupt__"][0].value
        if SHOW_STEPS and "understand" in update:
            print(f"   [understood] -> {update['understand']['route']}")
    return None


def main():
    config = {"configurable": {"thread_id": "patient-1"}}
    print("=" * 60)
    print(f"  {CLINIC_NAME} - Appointment Assistant")
    print("=" * 60)
    print("Hello! I can book or change an appointment, or answer questions")
    print("about fees, timings and clinic policies. Type 'quit' to exit.\n")
    print(categories_text())

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit", "bye"}:
            print(f"Assistant: Thank you for contacting {CLINIC_NAME}. Take care!")
            break
        if not text:
            continue
        question = run({"messages": [HumanMessage(text)]}, config)
        while question:  # the graph is paused, waiting for the patient's yes/no
            answer_text = input(f"   {question} (yes/no): ")
            question = run(Command(resume=answer_text), config)
        print("Assistant:", graph.get_state(config).values["messages"][-1].content)


if __name__ == "__main__":
    main()