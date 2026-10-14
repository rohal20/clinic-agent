"""Clinic Appointment Agent

Same features as agent.py (RAG for clinic info, find slots, book with confirmation, memory),
but it talks to Ollama's API directly with `requests`, like the earlier agent.py practice file
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path
from datetime import date, datetime, timedelta

import requests

OLLAMA = "http://localhost:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
MAX_STEPS = 6  # stop the agent loop after this many model calls

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
SHOWN_SLOTS = set()  # keep track of which slots have been shown to the patient, and pairs of (doctor_id, date) that have been booked


def to_date(day: str) -> date:
    day = day.strip().lower()
    if day == "today":
        return date.today()
    if day == "tomorrow":
        return date.today() + timedelta(days=1)
    return date.fromisoformat(day)


def free_times(doctor_id: str, d: date) -> list[str]:
    doctor = DOCTORS[doctor_id]
    if WEEKDAYS[d.weekday()] not in doctor["days"]:
        return []
    taken = [b["time"] for b in BOOKINGS if b["doctor_id"] == doctor_id and b["date"] == d.isoformat()]
    now = datetime.now()
    free = []
    for t in TIMES:
        if (doctor_id, d.isoformat(), t) in SHOWN_SLOTS:
            continue  # don't show the same slot twice
        if d == now.date() and datetime.strptime(t, "%H:%M").time() <= now.time():
            continue  # don't show past times today
        free.append(t)
    return free

def ask_patient_to_confirm(doctor_id: str, day: str, time: str, patient_name: str) -> bool:
    doctor = DOCTORS.get(doctor_id, {}).get("name", doctor_id)
    answer = input(f"   Confirm: {doctor} on {day} at {time} for {patient_name}? (yes/no): ").strip().lower()
    return answer == "yes"

def list_doctors() -> str:
    lines = ["Available doctors:"]
    for doctor_id, doctor in DOCTORS.items():
        lines.append(f"{doctor['name']} (id {doctor_id}), specialty: {doctor['specialty']}, "
                     f"days: {', '.join(doctor['days'])}")
    return "\n".join(lines)

# ------------------------------------------------------------------
# 2. RAG by hand: embed each FAQ section once, then compare with cosine similarity
# ------------------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


faq_text = Path(__file__).with_name("faq.md").read_text(encoding="utf-8")
SECTIONS = [part.strip() for part in faq_text.split("## ")[1:]]  # chunking here, splits into sections 
SECTION_VECTORS = embed(SECTIONS) # then embedding it turning each section into a vector for similarity search


# ------------------------------------------------------------------
# 3. Tools: plain functions + JSON schemas that describe them to the model
# ------------------------------------------------------------------
def search_clinic_info(question: str) -> str:  # embeds the question and finds the top-2 most similar sections from the FAQ - retrieval part of RAG
    q = embed([question])[0]
    scores = [cosine(q, v) for v in SECTION_VECTORS] # compute cosine similarity between the question and each section vector
    best = sorted(range(len(SECTIONS)), key=lambda i: scores[i], reverse=True)[:2]  # top-2
    return "\n\n".join(SECTIONS[i] for i in best)


def find_available_slots(specialty: str, day: str) -> str:
    d = to_date(day)
    lines = [f"Date: {d:%A} {d.isoformat()}"]
    for doctor_id, doctor in DOCTORS.items():
        if doctor["specialty"] == specialty:
            times = free_times(doctor_id, d)
            lines.append(f"{doctor['name']} (id {doctor_id}): {', '.join(times) or 'not available'}")
            SHOWN_SLOTS.update((doctor_id, d.isoformat(), t) for t in times)  # mark these slots as shown
    return "\n".join(lines)


def book_appointment(doctor_id: str, day: str, time: str, patient_name: str) -> str:
    if doctor_id not in DOCTORS:
        return f"Error: unknown doctor id {doctor_id}"
    d = to_date(day)
    if (doctor_id, d.isoformat(), time) not in SHOWN_SLOTS:
        return "First show the patient the available slots for this doctor and day, then let them pick one."
    if time not in free_times(doctor_id, d):
        return f"Error: {time} is not available for doctor {doctor_id} on {d.isoformat()}"
    doctor = DOCTORS[doctor_id]["name"]
    if not ask_patient_to_confirm(doctor_id, d.isoformat(), time, patient_name):
        return "The patient did not confirm. Nothing was booked."
    booking = {"id": len(BOOKINGS) + 1, "doctor_id": doctor_id, "date": d.isoformat(), "time": time, "patient": patient_name}
    BOOKINGS.append(booking)
    return f"Booked {doctor} on {d:%A %Y-%m-%d} at {time} for {patient_name}. (booking id {booking['id']})"

def my_appointments(patient_name: str) -> str:
    lines = [f"Appointments for {patient_name}:"]
    found = False
    for booking in BOOKINGS:
        if booking["patient"].lower() == patient_name.lower():
            doctor = DOCTORS[booking["doctor_id"]]["name"]
            lines.append(f"- {booking['date']} at {booking['time']} with {doctor} (id {booking['id']})")
            found = True
    if not found:
        lines.append("No appointments found.")
    return "\n".join(lines)

def reschedule_appointment(booking_id: int, new_day: str, new_time: str) -> str:
    booking = next((b for b in BOOKINGS if b["id"] == booking_id), None)
    if booking is None:
        return f"Error: no booking found with id {booking_id}"
    current = datetime.strptime(f"{booking['date']} {booking['time']}", "%Y-%m-%d %H:%M")
    if current - datetime.now() < timedelta(hours=24):
        return "Error: cannot reschedule an appointment less than 24 hours in advance."
    doctor_id = booking["doctor_id"]
    d = to_date(new_day)
    if (doctor_id, d.isoformat(), new_time) not in SHOWN_SLOTS:
        return "First show the patient the available slots for this doctor and day, then let them pick one."
    if new_time not in free_times(doctor_id, d):
        return f"Error: {new_time} is not available for doctor {doctor_id} on {d.isoformat()}"
    doctor = DOCTORS[doctor_id]["name"]
    question = f"Confirm rescheduling appointment {booking_id} to {d.isoformat()} at {new_time} with {doctor}? (yes/no): "
    if not ask_patient_to_confirm(doctor_id, d.isoformat(), new_time, booking["patient"]):
        return "The patient did not confirm. Nothing was rescheduled."
    booking["date"] = d.isoformat()
    return f"Rescheduled appointment {booking_id} to {d:%A %Y-%m-%d} at {new_time} with {doctor}."


TOOLS = {
    "search_clinic_info": search_clinic_info,
    "find_available_slots": find_available_slots,
    "book_appointment": book_appointment,
    "my_appointments": my_appointments,
    "reschedule_appointment": reschedule_appointment,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_clinic_info",
        "description": "Search clinic information: fees, opening hours, location, payment and cancellation policy.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "What the patient wants to know"}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "find_available_slots",
        "description": "Find free appointment times for a specialty.",
        "parameters": {"type": "object", "properties": {
            "specialty": {"type": "string", "enum": ["general", "dermatology", "pediatrics"]},
            "day": {"type": "string", "description": "'today', 'tomorrow' or YYYY-MM-DD"}},
            "required": ["specialty", "day"]}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Book an appointment. Use only after the patient has picked a doctor and a time "
                       "and given their name.",
        "parameters": {"type": "object", "properties": {
            "doctor_id": {"type": "string", "description": "Doctor id, e.g. D2"},
            "day": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM, e.g. 10:30"},
            "patient_name": {"type": "string"}},
            "required": ["doctor_id", "day", "time", "patient_name"]}}},
    {"type": "function", "function": {
                    "name": "list_doctors",
                    "description": "List all doctors with their specialties and available days.",
                    "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "my_appointments",
        "description": "List all appointments for a patient.",
        "parameters": {"type": "object", "properties": {
            "patient_name": {"type": "string"}},
            "required": ["patient_name"]}}},
    {"type": "function", "function": {
        "name": "reschedule_appointment",
        "description": "Reschedule an existing appointment. Only allowed if the appointment is more than 24 hours away.",
        "parameters": {"type": "object", "properties": {
            "booking_id": {"type": "integer", "description": "The booking id returned when the appointment was booked"},
            "new_day": {"type": "string", "description": "'today', 'tomorrow' or YYYY-MM-DD"},
            "new_time": {"type": "string", "description": "'HH:MM', e.g. 10:30"}},
            "required": ["booking_id", "new_day", "new_time"]}}}

]

SYSTEM_PROMPT = f"""You are the appointment assistant for City Care Clinic (a demo clinic).
Today is {date.today():%A %Y-%m-%d}.

To book an appointment, follow these steps:
1. Ask the patient for their name.
2. Ask what specialty they want (general, dermatology, pediatrics).
3. Call the tool `find_available_slots` with the specialty and a day (today, tomorrow, or a specific date) to show available times.
4. Ask the patient to pick a doctor and a time from the available slots.
5. Call the tool `book_appointment` with the doctor_id, day, time, and patient_name. The tool will ask for confirmation before booking

To reschedule an appointment, follow these steps:
1. Ask the patient for their name.
2. Call the tool `my_appointments` with the patient_name to show their current appointments.
3. Ask the patient which appointment they want to reschedule (by booking id).
4. Ask the patient for a new day and time.
5. Call the tool `reschedule_appointment` with the booking_id, new_day,
    and new_time. The tool will ask for confirmation before rescheduling.

Other tools:
- For fees and other clinic information, call `search_clinic_info` with the patient's question.
- Do not give any medical advice. If the patient asks for medical advice, politely decline and suggest they consult a doctor.
- Keep answers concise and to the point. If the patient asks for more information, call `search_clinic_info` with their question.
- If the patient asks for a list of doctors, call `list_doctors."""


# ------------------------------------------------------------------
# 4. The agent loop (this is what LangGraph's agent -> tools -> agent graph does)
# ------------------------------------------------------------------
def call_model(messages: list[dict]) -> dict:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "messages": messages, "tools": TOOL_SCHEMAS,
        "stream": False, "options": {"temperature": 0}}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]


def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"Error: unknown tool {name}"
    print(f"   Running tool {name} with args {args}")
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error running tool {name}: {e}"


def run_agent(messages: list[dict]) -> str:
    """Call the model; run any tools it asks for; repeat until it answers in text."""
    for _ in range(MAX_STEPS):
        reply = call_model(messages)
        messages.append(reply)
        if not reply.get("tool_calls"):
            return reply.get("content", "")
        for call in reply["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):  # some models send arguments as a JSON string
                args = json.loads(args)
            messages.append({"role": "tool", "tool_name": name, "content": run_tool(name, args)})
    return "Sorry, I couldn't finish that. Please try again."


# ------------------------------------------------------------------
# 5. Chat in the terminal. Memory = this messages list, resent every turn.
# ------------------------------------------------------------------
def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("City Care Clinic assistant. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            messages.append({"role": "user", "content": text})
            print("Assistant:", run_agent(messages))


if __name__ == "__main__":
    main()
