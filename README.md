# IIT Jodhpur Academic Helpdesk

A small local academic helpdesk built around the actual IIT Jodhpur files supplied with the assignment. It uses a deterministic orchestrator and three specialist agents:

- **Policy Agent** — RAG over the supplied PDF/TXT academic documents.
- **Schedule Agent** — direct lookups and filters over the original course CSV.
- **Email Agent** — course-instructor lookup using the course CSV and faculty email CSV, followed by an email draft. It never sends email.

The orchestrator refuses questions outside the academic scope. It also refuses in-scope questions when the supplied sources do not contain enough evidence. Both types of refusal are logged in `unanswered_questions.csv`.

## Why this implementation is deliberately small

Google ADK is optional in the assignment, not mandatory. This implementation uses three separate Python agent classes and one orchestrator instead of introducing an additional framework. All agents share one locally hosted Ollama model; the Schedule Agent uses deterministic CSV operations and does not call the LLM for lookups or counts.

No cloud API, external database, or remote model is required at runtime. ChromaDB is persisted locally.

## Repository structure

```text
academic-helpdesk/
├── academic_helpdesk/
│   ├── agents.py              # Orchestrator and three specialist agents
│   ├── config.py              # Local paths and model configuration
│   ├── csv_tools.py           # Multi-table CSV parsing and lookup tools
│   ├── ingest.py              # One-time document indexing entry point
│   ├── main.py                # Interactive command-line helpdesk
│   ├── ollama_client.py       # Minimal local Ollama HTTP client
│   └── rag.py                 # PDF/TXT extraction, chunking, Chroma retrieval
├── data/
│   ├── documents/             # Original PDFs and TXT files from documents.zip
│   ├── manual_transcriptions/ # Page-attributed text for scanned PDF pages
│   ├── courses(1).csv         # Original course export, unchanged
│   └── faculty_emails.csv     # Original faculty export, unchanged
├── evaluation/
│   ├── questions.json         # 25 evaluation prompts with expected routing
│   └── run_eval.py            # Runs each prompt multiple times
├── reports/
│   └── findings.md            # Data audit, design, and evaluation report template
├── tests/
├── requirements.txt
└── .env.example
```

## Requirements

- Python 3.10 or newer.
- Ollama installed and running locally.
- 16 GB RAM is sufficient with one quantized 8B model loaded at a time. Avoid loading two chat models simultaneously.

Recommended Ollama models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

To compare against the second model in the assignment:

```bash
ollama pull llama3.1:8b
```

## Setup

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

The defaults in `.env.example` use `qwen3:8b` for chat and `nomic-embed-text` for embeddings. Edit `.env` to change models or the Ollama URL. The application loads `.env` automatically using `python-dotenv`. The Ollama chat request explicitly sets `think: false` for Qwen3, so Qwen3 reasoning mode is disabled.

Make sure Ollama is running. Then build the local ChromaDB index once:

```bash
python -m academic_helpdesk.ingest
```

Start the helpdesk:

```bash
python -m academic_helpdesk.main
```

Type `:quit` to exit. Re-run ingestion after replacing or updating any source documents. The ingestion script rebuilds the collection so it does not retain stale chunks.

## Example questions

```text
What is the attendance requirement to sit in the Major exam?
What is the slot for Autonomous Systems?
What is the course code for Autonomous Systems?
What is the email ID of the instructor of CSL7650?
Write an email to the instructor of CSL7650 that I need sick leave for 2 days.
How many Open Elective credits are required for a regular B.Tech degree?
What are the available dining options on the campus?
```

The provided course CSV identifies **Autonomous Systems as CSL7650**, slot **E**, taught by **Gaurav Harit**. The assignment example mentions CSL7570, but that code is not present in the supplied course export. The helpdesk intentionally does not silently replace CSL7570 with CSL7650.

## Source and citation behaviour

- RAG answers show the source filename and PDF page. TXT sources are cited by filename and text chunk.
- Course lookups cite `courses(1).csv` and its original row number.
- Faculty lookups cite `faculty_emails.csv` and the faculty source URL when available.
- The Email Agent returns a draft with placeholders for missing dates, student name, and roll number. It does not send the email.
- The RAG prompt requires the model to answer only from retrieved excerpts and include inline source markers. The retriever explicitly supplements attendance and examination queries with the relevant July 2024 amendments, which supersede earlier policies. If the excerpts do not establish an answer or the model fails to cite a retrieved passage, the agent refuses.
- Later amendments that explicitly supersede earlier policies should take precedence. The two July 2024 scanned circulars and one scanned page from the 2022 notification are represented by text transcriptions in `data/manual_transcriptions/`, with page markers that cite the original PDFs.

## Course CSV handling

`courses(1).csv` is intentionally kept byte-for-byte as uploaded. It contains several concatenated course tables with different header names and column positions. `csv_tools.py` detects each table header and maps its fields in memory. It retains the raw fields and maps slot, preferred slot, lab slot, programme/batch, year/semester and remarks separately. It does not export a cleaned replacement CSV.

A slot code is not the same as a clock time. The uploaded course export records the slot for Autonomous Systems as `E`, but the supplied files do not include a general slot-to-clock-time timetable. If a course row explicitly includes clock times in a timing or remarks field, the Schedule Agent returns them; otherwise, it says that the clock time is unavailable instead of inventing one.

Course counts are conservative. They are returned only when the course level, department and elective category are explicitly available. In the supplied Humanities/Liberal Arts entries, the 400-level record `LAL4070` has no category value; the system will not assume it is an elective. The dataset also does not consistently identify the semester for every course row. Such a count is therefore logged as not answerable from the current data.

## Evaluation

Run the regression tests after setup:

```bash
pip install pytest
python -m pytest -q
```

Run the 25 evaluation prompts three times:

```bash
python evaluation/run_eval.py --repeats 3
```

For each output row, manually fill:

- `manual_accuracy`: 1 if the factual answer is correct, otherwise 0.
- `manual_groundedness`: 1 if the cited source supports the answer, otherwise 0.
- `manual_refusal_correct`: 1 if the refusal decision is correct, otherwise 0.

To compare models, set `LLM_MODEL` in the shell and run the evaluation again, for example:

```bash
LLM_MODEL=llama3.1:8b python evaluation/run_eval.py --repeats 3 --output evaluation/results_llama.csv
```

Compare answer strings grouped by `question_id` to identify run-to-run changes. Do not report model accuracy or groundedness until the rows have been manually checked against the citations.

## Troubleshooting

**Ollama connection error:** start Ollama and check `http://localhost:11434`. Pull both configured models.

**Empty ChromaDB collection:** run `python -m academic_helpdesk.ingest` from the repository root.

**Scanned PDF warning:** the two July 2024 amendment PDFs have image-only pages. Their relevant text is supplied in page-attributed transcription files. Other scanned pages, if added later, need an OCR/transcription step before they can be searched.

**Missing answer:** inspect `unanswered_questions.csv`. An in-scope unanswered question usually indicates a missing source document, a failed extraction, or insufficient retrievable evidence. Do not loosen refusal behaviour merely to force an answer.

## Scope boundary

This helpdesk covers only the academic questions supported by the supplied IIT Jodhpur documents and CSV exports. It does not answer dining, transport, weather, accommodation, or other non-academic questions. Missing or ambiguous academic records are also refused and logged.

### Email requests

Requests such as “write and email to the course instructor of CSL7650 for help” are routed to the Email Agent before generic course-code routing. The Email Agent uses the course CSV to find the instructor and the faculty CSV to find the email address, then drafts a **Request for Assistance** email. It does not misclassify a general help request as a leave request, and it never sends the email.

### 400-level course counting

Questions such as `How many courses are available as 400-level courses?` are handled by the deterministic Schedule Agent, not the Policy RAG agent. The level is derived from the first numeric digit in the course code (for example, `CSL4010` is 400-level). The current supplied export contains 50 parsed numeric 400-level course records across all departments. Wildcard codes such as `CYL4XX0` are excluded because their exact level cannot be established from the code.

For elective counts, the system counts only records whose category is explicitly marked as an elective. If matching rows have blank categories, it reports the confirmed count and lists the unclassified rows rather than silently treating them as electives or non-electives. The supplied CSV does not consistently identify the semester of offering, so current-semester counts cannot be asserted from this file.
