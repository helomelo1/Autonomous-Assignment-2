# Academic Helpdesk — Implementation and Data Findings

## 1. Objective

Build a laptop-runnable IIT Jodhpur academic helpdesk that answers document-backed policy questions, performs exact course/faculty lookups, drafts emails for student review, cites evidence, and refuses unsupported requests.

## 2. Implementation summary

The implementation uses a deterministic orchestrator with three specialist Python agents:

1. **Policy Agent:** ChromaDB retrieval over extracted PDF/TXT chunks, with Ollama embeddings and a local Ollama chat model for evidence-constrained answer generation.
2. **Schedule Agent:** direct parsing and filtering of the original multi-table course CSV. No LLM is used for slot, credits, instructor, and count operations.
3. **Email Agent:** course lookup, faculty-name match, faculty email retrieval, and LLM-assisted email drafting. A draft is returned only; nothing is sent.

Out-of-scope queries and in-scope queries with insufficient source data are refused and logged separately by reason in `unanswered_questions.csv`.

Google ADK was not added because the assignment makes it optional and a small custom orchestrator is simpler to install and debug. The three agents remain separate components with narrow responsibilities.

## 3. Supplied data audit

### Documents

The uploaded ZIP contains 36 source files: 11 PDFs and 25 TXT files. The collection includes academic regulations, academic calendars, curriculum pages, and faculty pages. Two PDFs dated 8 July 2024 are image-only scans and return no extractable text through ordinary PDF text extraction.

To avoid losing those amendments, the repository includes page-attributed manual transcriptions for:

- `academic_regulations_7.1_...pdf` — attendance policy amendment.
- `academic_regulations_7.2_...pdf` — Minor/Major examination policy amendment.
- `academic_regulations_5_...pdf` — page 1 of the 2022 Joint Medical Technologies notification.

The original PDFs remain unchanged. Citations in the transcription metadata point to the original PDF pages.

### Course CSV

The original `courses(1).csv` contains 763 physical CSV rows and many concatenated tables with different headers and column orders. It is not a single rectangular table with one header. A fixed `pandas.read_csv()` header interpretation would misalign fields in several sections.

The parser detects each table header independently and returns an in-memory list of normalized records while preserving the original CSV. The current parser recognizes approximately 600 course records; records with incomplete source fields remain incomplete rather than being filled with guesses.

**Verified example from the supplied CSV:** `Autonomous Systems` is listed as course code `CSL7650`, slot `E`, 3 credits, L-T-P `3-0-0`, instructor `Gaurav Harit`, for batch `UG4-AI` (source CSV row 399). The assignment's sample code `CSL7570` is not present in the uploaded course CSV; the application does not silently map it to another code.

**Timing limitation:** course rows contain slot/preferred-slot and, in some cases, lab-slot or timing details in remarks. The uploaded source files do not provide a general mapping from a slot letter (such as E) to a wall-clock interval. The system therefore returns explicit per-course timing if present, and otherwise states that the clock time is not available.

**Elective-count limitation:** the 400-level Humanities/Liberal Arts record visible in the supplied table, `LAL4070` (Health Economics), has a blank category field. The source also does not consistently identify the semester for every course row. A reliable count of current-semester electives cannot be computed from those fields, so the Schedule Agent refuses the count and logs the missing evidence instead of treating an unknown category as an elective.

### Faculty CSV

`faculty_emails.csv` contains 316 faculty records with name, department, email, designation, source URL, and related fields. Its `course_code` column is not populated in the supplied data. The Email Agent therefore finds the instructor from the course CSV first, then performs a conservative faculty-name match in the faculty CSV. It refuses ambiguous or missing matches.

## 4. Data quality controls

- The course CSV and faculty CSV are preserved as supplied.
- Course tables are parsed section by section using each section's own header row.
- Course-code lookup is exact; an unlisted code is not autocorrected.
- Fuzzy faculty matching is accepted only above a high similarity threshold and is rejected when the best matches are ambiguous.
- Elective counts require explicit department, course level, and category evidence.
- Clock-time answers require an explicit time/remarks value; no slot-time mapping is invented.
- Policy answers must be supported by retrieved source chunks.
- Later dated amendments explicitly marked as superseding older rules are given precedence in the RAG prompt.

## 5. Evaluation design

`evaluation/questions.json` contains 25 prompts across attendance policy, examination policy, degree requirements, registration, academic calendar, course slots/details, faculty lookup, email drafting, and out-of-scope refusal.

Run each prompt three times per model and manually score:

- **Answer accuracy:** correct factual answer compared with the cited source.
- **Groundedness/citation correctness:** cited source and page/chunk support the relevant claim.
- **Correct refusal:** non-academic or unsupported questions are refused, while supported academic questions are answered.

The evaluation script writes raw results and blank manual score columns. This report intentionally does not claim measured accuracy or groundedness before the local Ollama models have been run and the outputs reviewed.

## 6. Results table (fill after running evaluation)

| Model | Runs per question | Answer accuracy | Citation/groundedness | Correct refusal rate | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3 8B | 3 | Pending | Pending | Pending | Run `evaluation/run_eval.py` |
| Llama 3.1 8B | 3 | Pending | Pending | Pending | Run with `LLM_MODEL=llama3.1:8b` |

## 7. Known limitations and next data to collect

1. Add the official course timetable or slot-to-clock-time schedule if users must receive actual lecture timings.
2. Add the current-semester course offering export with complete department, semester, level, and course-category columns to support exact elective counts.
3. Add course syllabi/first course handouts if detailed course-content questions must be answered. Course names and slots alone are not sufficient evidence for syllabus questions.
4. OCR or manually transcribe any further scanned PDFs that do not have a text layer.

## 8. Conclusion

The design keeps the assignment's essential capabilities while avoiding unnecessary infrastructure. Structured course and faculty facts use deterministic lookups; policy facts use local RAG; email generation uses a bounded draft workflow; and unsupported queries are refused and logged. The known data gaps are explicit, so the system does not present guesses as IIT Jodhpur policy.


## Model thinking configuration

The Ollama chat client sends `think: false` in `/api/chat` requests. This disables Qwen3 thinking mode for the helpdesk workload. If a different Ollama model is selected, verify that model supports the `think` option; models that do not support it may ignore the field.

## Additional validation: 400-level course queries

The Schedule Agent now handles all 400-level count questions, including non-elective-specific counts. In the supplied course export, 50 parsed records have numeric 400-level course codes across all departments. The CSE subset contains two records: CSL4010 (Optimization for ML) and CSL4030 (Data Engineering). Neither has an explicit category in the relevant CSV table, so the system does not claim either is an elective. The Humanities subset contains LAL4070 (Health Economics), also with a blank category. Semester-of-offering metadata is not consistently present, so the system explicitly labels current-semester elective counts as unverified.

Regression tests for generic 400-level counts, CSE acronym recognition, partial elective evidence, and routing were added. Current test result: 17 passed.
