"""
CampusAssist AI — production Gradio app
Built from Week4_Capstone_CampusAssistAI.ipynb.

Pipeline:
  - RAG (BM25 retrieval) over a campus policy knowledge base for
    "what's the rule" questions
  - Agent (tool-calling) for "what's *my* status" questions, backed by
    dummy student records
  - Guardrails: prompt-injection detection + off-topic scope check,
    applied before any model call
  - A write tool (raise_complaint) requires confirmation before it runs

The Groq API key is read from the GROQ_API_KEY environment variable —
never hardcoded. Set it in Render under Environment Variables.
"""

import os
import re
import json
import random

import gradio as gr
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Part A — Client setup
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Add it in Render under Environment Variables before deploying."
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

MAIN_MODEL = "openai/gpt-oss-120b"  # RAG answers + agent decisions

# ---------------------------------------------------------------------------
# Part B — Knowledge base (AskDesk)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "kb_attendance",
        "section": "Attendance Policy",
        "content": (
            "Minimum 75% attendance is required in every subject to be "
            "eligible to sit for the semester-end. Students between "
            "65% and 75% can apply for condonation with a valid medical "
            "certificate or an approved leave letter, submitted to the "
            "class coordinator before the last week of the semester. "
            "Below 65%, the student is detained and must repeat the "
            "semester for that subject. Attendance is calculated from the "
            "first working day of the semester, not from the date of "
            "admission."
        ),
    },
    {
        "id": "kb_exam",
        "section": "Exam and Result Policy",
        "content": (
            "Each subject is evaluated out of 100: 30 marks internal "
            "(assignments, class tests, attendance) plus 70 marks external "
            "(semester-end written exam). Minimum passing is 40% overall "
            "and at least 35% in the external exam separately. If a "
            "student fails a subject it becomes a backlog and must be "
            "cleared in a later attempt alongside the next semester's "
            "regular subjects. Revaluation applications are open for 7 "
            "days after results are declared, with a fee of Rs 300 per "
            "subject."
        ),
    },
    {
        "id": "kb_fees",
        "section": "Fee Structure",
        "content": (
            "Semester tuition fee is due within the first two weeks of "
            "each semester. A late fee of Rs 500 per week applies after "
            "the due date, capped at Rs 2000. Exam form fee is separate "
            "and must be paid before the exam form deadline or the "
            "student cannot sit for that semester's exams. Students under "
            "SC/ST/EWS categories with a valid caste and income "
            "certificate are eligible for a government scholarship that "
            "covers tuition fully or partially, applied for through the "
            "state scholarship portal, not through the college directly."
        ),
    },
    {
        "id": "kb_hostel",
        "section": "Hostel Rules",
        "content": (
            "Hostel gate closes at 9:00 PM on weekdays and 10:00 PM on "
            "weekends; entry after that requires warden permission. "
            "Students leaving for home or an outing overnight must submit "
            "a leave application at least 24 hours in advance, countersigned "
            "by a parent or guardian for first-year students. Mess timings "
            "are 7:30-9:00 AM (breakfast), 12:30-2:00 PM (lunch), and "
            "7:30-9:00 PM (dinner). Outside food delivery is allowed only "
            "at the main gate, not inside hostel rooms."
        ),
    },
    {
        "id": "kb_calendar",
        "section": "Academic Calendar",
        "content": (
            "Odd semester runs from mid-June to early November, even "
            "semester from early December to April. Internal exams (unit "
            "tests) are held in the 8th and 14th week of each semester. "
            "Semester-end exams start the week after the semester "
            "officially ends. Diwali break is typically two weeks in "
            "late October/November, and summer break runs through May."
        ),
    },
    {
        "id": "kb_general",
        "section": "General FAQ",
        "content": (
            "The library is open 8:00 AM to 8:00 PM on working days and "
            "closes at 2:00 PM on Saturdays; it is shut on Sundays and "
            "public holidays. A lost ID card can be replaced by the "
            "admin office for a Rs 100 fee, needs a written application, "
            "takes about 3 working days. General queries not covered "
            "elsewhere can be emailed to the admin office or asked at the "
            "front desk during office hours, 9 AM to 5 PM."
        ),
    },
]


def build_chunks(kb):
    chunks = []
    for doc in kb:
        chunk_text = f"[{doc['section']}] {doc['content']}"
        chunks.append({"id": doc["id"], "section": doc["section"], "text": chunk_text})
    return chunks


CHUNKS = build_chunks(KNOWLEDGE_BASE)

# ---------------------------------------------------------------------------
# Part B.2 — BM25 retrieval
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
    "what", "how", "much", "does", "do", "are", "this", "that", "it", "its",
    "be", "as", "at", "by", "from", "was", "were", "will", "can", "i", "you", "my",
}


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


tokenized_chunks = [tokenize(c["text"]) for c in CHUNKS]
bm25_index = BM25Okapi(tokenized_chunks)


def retrieve(query, top_k=3):
    q_tokens = tokenize(query)
    scores = bm25_index.get_scores(q_tokens)
    ranked = sorted(zip(CHUNKS, scores), key=lambda x: x[1], reverse=True)
    return [
        {"id": c["id"], "section": c["section"], "text": c["text"], "score": s}
        for c, s in ranked[:top_k]
    ]


# ---------------------------------------------------------------------------
# Part C — Guardrails
# ---------------------------------------------------------------------------

SCOPE_KEYWORDS = {
    "attendance", "exam", "exams", "result", "results", "backlog",
    "revaluation", "fee", "fees", "scholarship", "hostel", "mess",
    "warden", "leave", "timetable", "calendar", "semester", "holiday",
    "library", "id card", "campusassist", "college", "diwali break",
    "internal", "external", "condonation", "detained", "admission",
    "student", "roll no", "id",
}

MIN_RETRIEVAL_SCORE = 3.0


def in_scope(query):
    q_lower = query.lower()
    if any(kw in q_lower for kw in SCOPE_KEYWORDS):
        return True
    top = retrieve(query, top_k=1)
    return bool(top) and top[0]["score"] >= MIN_RETRIEVAL_SCORE


INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above)",
    r"you are now",
    r"act as (if you|a) (were|are)",
    r"reveal (your |the )?system prompt",
    r"print (your |the )?(system|instructions)",
    r"do anything now",
    r"\bdan\b",
    r"pretend (you are|to be)",
    r"forget (everything|your instructions)",
    r"give (me )?(full|100%|90%) (marks|attendance)",
]


def is_injection_attempt(query):
    q_lower = query.lower()
    return any(re.search(p, q_lower) for p in INJECTION_PATTERNS)


OFF_TOPIC_REFUSAL = (
    "I can only help with CampusAssist topics — attendance, exams, fees, "
    "hostel, or the academic calendar. That question is outside what I can "
    "answer."
)
INJECTION_REFUSAL = (
    "I can't follow instructions embedded in a message like that. "
    "Happy to help with a genuine CampusAssist question though."
)

# ---------------------------------------------------------------------------
# Part C.3 — Tool guard (read-only tools run freely, write tools need confirmation)
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS = set()
WRITE_TOOLS = set()


class ToolGuardError(Exception):
    pass


def register_tool(name, read_only: bool):
    (READ_ONLY_TOOLS if read_only else WRITE_TOOLS).add(name)


def guarded_tool_call(tool_name, tool_fn, args, confirmed=False):
    if tool_name in WRITE_TOOLS and not confirmed:
        raise ToolGuardError(
            f"'{tool_name}' changes something and needs confirmation "
            f"before it runs. Ask the student to confirm first."
        )
    return tool_fn(**args)


# ---------------------------------------------------------------------------
# Part D — StatusCheck: dummy student records + tools
# ---------------------------------------------------------------------------

STUDENT_RECORDS = {
    "S101": {"name": "Rohan Patil", "attendance": 82, "fee_due": 0,
             "results": {"maths": "Pass (68)", "physics": "Pass (74)"}},
    "S102": {"name": "Aisha Shaikh", "attendance": 61, "fee_due": 5000,
             "results": {"maths": "Fail (32)", "physics": "Pass (55)"}},
    "S103": {"name": "Karan Deshmukh", "attendance": 91, "fee_due": 0,
             "results": {"maths": "Pass (79)", "physics": "Pass (81)"}},
}

# Generate S104-S163 with a fixed seed so records are reproducible across runs.
random.seed(42)
for i in range(104, 164):
    student_id = f"S{i}"
    attendance = random.randint(50, 99)
    fee_due = random.choice([0, 1000, 2500, 5000])
    maths_score = random.randint(20, 95)
    physics_score = random.randint(20, 95)
    STUDENT_RECORDS[student_id] = {
        "name": f"Student {i}",
        "attendance": attendance,
        "fee_due": fee_due,
        "results": {
            "maths": f"Pass ({maths_score})" if maths_score >= 40 else f"Fail ({maths_score})",
            "physics": f"Pass ({physics_score})" if physics_score >= 40 else f"Fail ({physics_score})",
        },
    }


def check_attendance(student_id):
    rec = STUDENT_RECORDS.get(student_id)
    if not rec:
        return f"No student found with ID {student_id}."
    return f"{rec['name']} ({student_id}) has {rec['attendance']}% attendance."


def check_exam_result(student_id, subject):
    rec = STUDENT_RECORDS.get(student_id)
    if not rec:
        return f"No student found with ID {student_id}."
    result = rec["results"].get(subject.lower())
    if not result:
        return f"No result on file for {subject} for {rec['name']}."
    return f"{rec['name']}'s result in {subject}: {result}."


def check_fee_due(student_id):
    rec = STUDENT_RECORDS.get(student_id)
    if not rec:
        return f"No student found with ID {student_id}."
    if rec["fee_due"] == 0:
        return f"{rec['name']} has no pending fee."
    return f"{rec['name']} has Rs {rec['fee_due']} pending in fees."


COMPLAINT_LOG = []


def raise_complaint(student_id, issue):
    COMPLAINT_LOG.append({"student_id": student_id, "issue": issue})
    return f"Complaint logged for {student_id}: '{issue}'. Admin office will follow up."


register_tool("check_attendance", read_only=True)
register_tool("check_exam_result", read_only=True)
register_tool("check_fee_due", read_only=True)
register_tool("raise_complaint", read_only=False)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "check_attendance",
        "description": "Get a student's current attendance percentage.",
        "parameters": {"type": "object", "properties": {
            "student_id": {"type": "string", "description": "e.g. S101"}
        }, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "check_exam_result",
        "description": "Get a student's exam result for a specific subject.",
        "parameters": {"type": "object", "properties": {
            "student_id": {"type": "string"},
            "subject": {"type": "string", "description": "e.g. maths, physics"},
        }, "required": ["student_id", "subject"]},
    }},
    {"type": "function", "function": {
        "name": "check_fee_due",
        "description": "Get a student's pending fee amount.",
        "parameters": {"type": "object", "properties": {
            "student_id": {"type": "string"}
        }, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "raise_complaint",
        "description": (
            "Log a complaint for a student. This WRITES a record and "
            "requires the student to confirm before it runs."
        ),
        "parameters": {"type": "object", "properties": {
            "student_id": {"type": "string"},
            "issue": {"type": "string"},
        }, "required": ["student_id", "issue"]},
    }},
]

TOOL_FUNCTIONS = {
    "check_attendance": check_attendance,
    "check_exam_result": check_exam_result,
    "check_fee_due": check_fee_due,
    "raise_complaint": raise_complaint,
}

# ---------------------------------------------------------------------------
# Part D.3 — ReAct agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the CampusAssist support assistant for Pimpri Chinchwad Polytechnic "
    "College. Only answer questions about attendance, exams, fees, hostel "
    "rules, or the academic calendar.\n\n"
    "For ANY question about a specific student's attendance, result, or "
    "fee status, you MUST call the matching tool (check_attendance, "
    "check_exam_result, check_fee_due) rather than guessing.\n"
    "For a complaint, call raise_complaint but tell the student it needs "
    "their confirmation first if it isn't already given.\n"
    "For general policy questions (attendance rules, exam structure, fee "
    "due dates, hostel timings, calendar dates), answer using ONLY the "
    "retrieved context below — never invent a rule or number that isn't "
    "in it. If the context doesn't cover it, say you don't have that "
    "information rather than guessing."
)

MAX_TURNS = 4


def agent(user_query, history=None, confirmed=False):
    """Runs the guardrails, retrieves context, then loops the tool-calling agent."""
    if is_injection_attempt(user_query):
        return INJECTION_REFUSAL, history or []
    if not in_scope(user_query):
        return OFF_TOPIC_REFUSAL, history or []

    context_chunks = retrieve(user_query, top_k=3)
    context_text = "\n\n".join(c["text"] for c in context_chunks)

    if history is None:
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context_text}"},
        ]
    else:
        messages = history.copy()
        messages[0] = {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context_text}"}

    messages.append({"role": "user", "content": user_query})

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=500,
            reasoning_effort="low",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, messages

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            try:
                result = guarded_tool_call(fn_name, TOOL_FUNCTIONS[fn_name], args, confirmed=confirmed)
            except ToolGuardError as e:
                result = str(e)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "Reached the step limit without a final answer — try rephrasing.", messages


# ---------------------------------------------------------------------------
# Part F — Gradio UI
# ---------------------------------------------------------------------------

# Keeps the running conversation (system + user + tool turns) between
# messages, so a follow-up like "S111" after "which student ID?" is
# understood in context. Resets whenever Gradio's own display history
# comes back empty (fresh page load or Clear button).
CONVERSATION_HISTORY = []


def chat_fn(message, history):
    global CONVERSATION_HISTORY
    if not history:
        CONVERSATION_HISTORY = []
    reply, CONVERSATION_HISTORY = agent(message, history=CONVERSATION_HISTORY, confirmed=True)
    return reply


demo = gr.ChatInterface(
    fn=chat_fn,
    title="CampusAssist AI",
    description=(
        "Ask about attendance, exams, fees, hostel rules, the academic "
        "calendar, or check a student's status (try any ID from S101 to S163)."
    ),
    examples=[
        "What's the minimum attendance required?",
        "What is S101's attendance?",
        "How do I apply for revaluation?",
        "What time does the hostel gate close?",
    ],
)

if __name__ == "__main__":
    # host 0.0.0.0 and the PORT env var are required for Render to detect the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
