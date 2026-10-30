# Source data inventory

- `documents/` contains the 36 source files extracted from the user's `documents.zip` (11 PDFs and 25 TXT files). These files are retained without content edits.
- `courses(1).csv` and `faculty_emails.csv` are the two original uploaded CSV exports. They are retained without content edits.
- `manual_transcriptions/` contains page-attributed text transcriptions for scanned pages whose text could not be extracted. Each transcription names the original PDF and page so answer citations still point to the original source.

The course CSV parser normalizes fields only in memory. It does not rewrite or export a transformed copy of either original CSV.
