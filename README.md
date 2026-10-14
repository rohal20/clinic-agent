# Clinic Appointment Agent

A simple agentic AI app. A patient chats with a clinic assistant, which can:

1. **Answer clinic questions** (fees, hours, policies) by searching `faq.md` (RAG)
2. **Find free appointment slots** with a tool
3. **Book an appointment**, but only after the patient confirms (human-in-the-loop)

It runs locally with **Ollama**. There are two versions that do exactly the same thing:

| File | How it's built | Run |
|---|---|---|
| `agent_plain.py` | Plain Python: calls Ollama's API with `requests` and runs the agent loop in a `for` loop | `python agent_plain.py` |
| `agent.py` | LangGraph: the same loop as a graph (agent and tools nodes) with `interrupt()` and a checkpointer | `python agent.py` |

**Side by side:**

| Part | Plain Python | LangGraph |
|---|---|---|
| Tool schemas | Written by hand as JSON | Generated from `@tool` functions and docstrings |
| Agent loop | `for` loop: call the model, run tools, repeat | `agent` node → conditional edge → `tools` node → back |
| Confirm before booking | `input()` inside `run_tool` | `interrupt()` + `Command(resume=...)` |
| Memory | The `messages` list, resent every turn | Checkpointer saves state per `thread_id` |
| RAG | `/api/embed` + a hand-written cosine function | `OllamaEmbeddings` + `InMemoryVectorStore` |
| Loop limit | `MAX_STEPS = 6` | `recursion_limit` |

The plain version is easiest to understand. The LangGraph version is better when the app grows:
pausing for approval inside a web app, saving memory to a database, or adding more branches.

## Run it

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python agent.py
```

## Example

```text
You: I need a skin doctor appointment tomorrow. What is the fee?
   [tool] search_clinic_info {'question': 'skin specialist fee'}
   [tool] find_available_slots {'specialty': 'dermatology', 'day': 'tomorrow'}
Assistant: The dermatology fee is Rs 2,500. Dr. Omar Malik is free tomorrow at 10:00, 10:30, 11:00...

You: Book 10:30 for Ali Khan
   Confirm: Dr. Omar Malik on 2026-10-15 at 10:30 for Ali Khan? (yes/no): yes
   [tool] book_appointment {...}
Assistant: Your appointment is booked: #1 with Dr. Omar Malik tomorrow at 10:30.
```

## How it works

```
START --> agent --(tool calls?)-- yes --> tools --+
            ^                                     |
            +-------------------------------------+
            |
            no --> END
```

- **agent node:** the LLM reads the conversation and decides to either reply or call tools.
- **tools node:** runs the tools the LLM asked for and adds the results to the conversation.
  For `book_appointment` it first calls `interrupt()`, which pauses the graph until the patient
  says yes or no.
- **conditional edge (`route`)**: tool calls go to `tools`; otherwise the run ends.
- **edge tools → agent:** after the tools run, the LLM sees the results and continues. This loop
  is the ReAct pattern: reason, act, observe, repeat.
- **checkpointer (`InMemorySaver`)**: saves the conversation for each `thread_id`, so the agent
  remembers earlier messages.

### One message, step by step

*"I need a skin doctor appointment tomorrow. What is the fee?"*

1. The message is added to the state (the list of messages).
2. **agent**: the LLM decides it needs two tools: `search_clinic_info` and `find_available_slots`.
3. **tools**:
   - `search_clinic_info` embeds the question and finds the closest FAQ sections (RAG).
   - `find_available_slots` turns "tomorrow" into a date in code and lists the free times.
4. **agent** again: the LLM writes the answer from the tool results. There are no more tool
   calls, so the run ends.

*"Book 10:30 for Ali Khan"*

5. **agent** calls `book_appointment`.
6. **tools** calls `interrupt()`. The graph pauses, and the app asks the patient to confirm.
7. The patient's "yes" is sent back with `Command(resume="yes")`. The booking is saved, and the
   agent confirms it to the patient.

## Concepts used

| Concept | In this app |
|---|---|
| Agent | The LLM decides which tools to call and when to stop |
| Tool calling | `@tool` functions; the docstrings tell the LLM when to use each one |
| RAG | `faq.md` is split into sections, embedded with `nomic-embed-text`, and searched by meaning |
| LangGraph state, nodes and edges | `MessagesState`, the `agent` and `tools` nodes, and the conditional edge `route` |
| Human-in-the-loop | `interrupt()` before booking, then `Command(resume=...)` |
| Memory | `InMemorySaver` checkpointer with a `thread_id` |
| Code for exact work | Dates and free slots are computed in Python, not by the LLM |
| Guardrails | The system prompt rules ("never make up fees", "no medical advice") and the booking checks |
