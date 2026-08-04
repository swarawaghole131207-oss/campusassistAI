# CampusAssist AI

A chatbot for Sunrise Polytechnic, Pune. Made as my capstone project.

**Name:** Swara Waghole
**Date:** 8/4/2026
**Live link:** https://76dca53b3f33a9d29f.gradio.live

---

## What it does

Students always have to search around for basic info like attendance rules,
exam process, fees, hostel timings, etc. And there's no easy way to check
your own status (like attendance % or pending fee) without going to the
office.

CampusAssist AI tries to fix that. It's a chatbot that can:

- Answer general policy questions (attendance, exams, fees, hostel, calendar)
  by reading real documents I gave it
- Look up a specific student's attendance, exam result, or fee status
- Log a complaint if a student has an issue
- Refuse to answer anything outside these topics, or anything that tries to
  trick it into ignoring its rules

## How it's built

- **Model:** Groq API 
- **Retrieval:** BM25 keyword search (`rank_bm25` library) over 6 small
  knowledge base documents. No vector database, no embeddings, just plain
  keyword matching. It's simple but it works fine for a small doc set like
  this.
- **Agent / tools:** 4 Python functions the model can call —
  `check_attendance`, `check_exam_result`, `check_fee_due` (all read-only)
  and `raise_complaint` (this one writes something, so it needs
  confirmation first)
- **UI:** Gradio chat interface, plus a separate HTML/CSS/JS front-end page
  I built for a nicer look (this one runs its own simplified copy of the
  logic in JavaScript so it works without needing the API key — good for
  quick demo, but the real graded system is the notebook)


## How to run it

1. Open the notebook in Google Colab
2. Run the cells top to bottom
3. When it asks for a Groq API key, paste yours (free key from
   console.groq.com)
4. Part F launches the Gradio chat — that's the working demo
5. Part E prints the evaluation numbers

## Author
Swara Waghole
