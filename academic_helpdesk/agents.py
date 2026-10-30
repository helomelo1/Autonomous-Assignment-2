"""Three small specialist agents and a deterministic helpdesk orchestrator."""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .csv_tools import (
    count_elective_courses,
    instructor_email_for_course,
    load_courses,
    load_faculty,
    norm,
    search_courses,
)
from .ollama_client import OllamaClient, OllamaError
from .rag import AcademicRAG

COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4}\s?-?\s?\d[A-Z0-9]{3,4})\b", re.IGNORECASE)


def extract_course_code(query: str, courses: list[dict[str, Any]] | None = None) -> str:
    """Extract a code only when it is present in the supplied course export.

    Matching against known codes prevents dates/words such as THE2024 or AY2026
    from being mistaken for course codes.
    """
    candidates = [
        re.sub(r"[\s-]+", "", match.group(1)).upper()
        for match in COURSE_CODE_RE.finditer(query)
    ]
    if courses is not None:
        known = {norm(course.get("course_code", "")) for course in courses if course.get("course_code")}
        known_prefixes = {
            match.group(1).upper()
            for course in courses
            if (match := re.match(r"^([A-Z]{2,4})", re.sub(r"[\s-]+", "", course.get("course_code", "").upper())))
        }
        for candidate in candidates:
            prefix = re.match(r"^[A-Z]{2,4}", candidate)
            if norm(candidate) in known or (prefix and prefix.group(0) in known_prefixes):
                # A familiar department prefix with an unknown suffix is still an attempted course code;
                # downstream exact lookup will refuse it without autocorrecting.
                return candidate
        return ""
    return candidates[0] if candidates else ""


def _citation(meta: dict[str, Any], index: int | None = None) -> str:
    source = meta.get("source", "Unknown source")
    page = meta.get("page", 0)
    chunk = meta.get("chunk", 0)
    if page:
        return f"{source}, p. {page}"
    return f"{source}, text chunk {chunk}"


def _course_source(course: dict[str, Any]) -> str:
    return f"courses(1).csv, row {course.get('source_row', '?')}"


class RefusalLogger:
    def __init__(self, path: Path = config.UNANSWERED_LOG):
        self.path = path

    def log(self, question: str, agent: str, reason: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "question", "agent", "reason"])
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "agent": agent,
                "reason": reason,
            })


class PolicyAgent:
    def __init__(self, rag: AcademicRAG, llm: OllamaClient):
        self.rag = rag
        self.llm = llm

    def _retrieve_relevant(self, question: str) -> list[dict[str, Any]]:
        """Retrieve the question context and force in relevant superseding amendments."""
        q = question.lower()
        targeted_queries: list[str] = []
        if "attendance" in q or "absent" in q or "absence" in q:
            targeted_queries.append(
                "IIT Jodhpur Circular 51 dated 8 July 2024 revised attendance policy; "
                "75 percent attendance, valid medical or bona fide leave, de-registration, "
                "automatic semester withdrawal; supersedes earlier policy"
            )
        if any(term in q for term in ("major", "minor", "examination", "exam")):
            targeted_queries.append(
                "IIT Jodhpur Circular 52 dated 8 July 2024 one Minor and one Major examination; "
                "Major duration three hours, minimum 40 percent weightage; supersedes earlier evaluation policy"
            )
        if any(term in q for term in ("make-up", "makeup", "missed examination", "missed exam")):
            targeted_queries.append(
                "IIT Jodhpur academic regulations procedure to apply for make-up examination after illness; "
                "course instructor, Dean Academics, deadline, supporting documents"
            )

        # Put focused amendment retrievals first, then supplement with general semantic results.
        # Dedupe by original source/page/chunk so one passage is not repeated in the prompt.
        focused: list[dict[str, Any]] = []
        for query in targeted_queries:
            focused.extend(self.rag.retrieve(query, top_k=2))
        general = self.rag.retrieve(question, top_k=config.TOP_K)
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for item in focused + general:
            meta = item.get("metadata", {})
            key = (str(meta.get("source", "")), int(meta.get("page", 0)), int(meta.get("chunk", 0)))
            if key not in seen:
                seen.add(key)
                merged.append(item)
            if len(merged) >= config.TOP_K + len(targeted_queries) * 2:
                break
        return merged

    def answer(self, question: str) -> dict[str, Any]:
        try:
            retrieved = self._retrieve_relevant(question)
        except Exception as exc:
            return {"ok": False, "reason": f"Document retrieval is unavailable: {exc}", "agent": "policy_agent", "system_error": True}
        if not retrieved:
            return {"ok": False, "reason": "No relevant academic document chunks were retrieved.", "agent": "policy_agent"}

        source_lines = []
        for index, item in enumerate(retrieved, start=1):
            source_lines.append(
                f"[S{index}] Source: {_citation(item['metadata'])}\n{item['text']}"
            )
        system = (
            "You are the policy specialist in an IIT Jodhpur academic helpdesk. "
            "Answer only from the supplied source excerpts. Do not use outside knowledge or guess. "
            "If the excerpts do not explicitly provide enough information to answer, output exactly [[NO_EVIDENCE]]. "
            "When sources conflict, prefer a later dated amendment that explicitly says it supersedes earlier policy. "
            "For each factual claim, cite the supporting source marker such as [S1]. "
            "Do not infer a rule from an older regulation when a newer amendment changes it. "
            "Be precise about conditions, exceptions, dates, and programme type."
        )
        user = (
            f"Question: {question}\n\nSource excerpts:\n" + "\n\n".join(source_lines) +
            "\n\nAnswer using only these excerpts. Use concise prose and source markers."
        )
        try:
            answer = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except OllamaError as exc:
            return {"ok": False, "reason": str(exc), "agent": "policy_agent", "system_error": True}
        if "[[NO_EVIDENCE]]" in answer or not answer.strip():
            return {"ok": False, "reason": "The available academic documents do not provide enough evidence for this question.", "agent": "policy_agent"}

        cited_indices = sorted({int(value) for value in re.findall(r"\[S(\d+)\]", answer)})
        valid_indices = [index for index in cited_indices if 1 <= index <= len(retrieved)]
        if not valid_indices:
            return {
                "ok": False,
                "reason": "The model did not provide a verifiable citation to a retrieved source.",
                "agent": "policy_agent",
            }
        citations = list(dict.fromkeys(_citation(retrieved[index - 1]["metadata"]) for index in valid_indices))
        return {"ok": True, "answer": answer, "sources": citations, "agent": "policy_agent"}


class ScheduleAgent:
    def __init__(self, courses: list[dict[str, Any]]):
        self.courses = courses

    def _find(self, question: str) -> list[dict[str, Any]]:
        code = extract_course_code(question, self.courses)
        if code:
            return search_courses(self.courses, course_code=code, limit=8)
        return search_courses(self.courses, query=question, limit=8)

    @staticmethod
    def _format_course(course: dict[str, Any], asks_timing: bool = False) -> str:
        lines = [f"**{course.get('course_name') or 'Course'}** ({course.get('course_code') or 'code not listed'})"]
        for label, key in [
            ("Programme/batch", "program"),
            ("Department", "department"),
            ("Instructor", "instructor"),
            ("Credits", "credits"),
            ("L-T-P", "ltp"),
            ("Course category/type", "category"),
            ("Slot", "slot"),
            ("Preferred slot", "preferred_slot"),
            ("Lab slot", "lab_slot"),
            ("Recorded timing", "timing"),
        ]:
            value = course.get(key)
            if value and label not in {"Preferred slot"}:
                lines.append(f"- {label}: {value}")
        extras = []
        for key, value in course.get("raw_fields", {}).items():
            normalized = re.sub(r"[^a-z]", "", key.lower())
            if normalized in {"remarks", "comment", "comments", "offeredfor", "program", "programme", "year"} and value:
                extras.append(f"{key}: {value}")
        if extras:
            lines.append("- Additional source details: " + "; ".join(dict.fromkeys(extras)))
        if asks_timing:
            time_text = " ".join([course.get("timing", ""), course.get("remarks", ""), *course.get("raw_fields", {}).values()]).lower()
            explicit_clock = bool(re.search(r"\b\d{1,2}(?::\d{2}|\.\d{2})\s*(?:am|pm)?\b|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", time_text))
            if not explicit_clock:
                lines.append("- Clock-time: Not listed for this course in the supplied CSV. The slot code alone does not establish a clock-time, and no general slot-to-time timetable was included in the uploaded documents.")
        lines.append(f"- Source: {_course_source(course)}")
        return "\n".join(lines)

    def answer(self, question: str) -> dict[str, Any]:
        q = question.lower()
        is_count = any(term in q for term in ("how many", "count", "number of"))
        if is_count and re.search(r"\b(100|200|300|400|500|600|700|800)\s*[- ]?level\b|\b([1-8])00\s*[- ]?level\b", q):
            level_match = re.search(r"\b(100|200|300|400|500|600|700|800)\s*[- ]?level\b|\b([1-8])00\s*[- ]?level\b", q)
            token = level_match.group(1) or (level_match.group(2) + "00")
            level = int(token)
            department = ""
            department_patterns = [
                (r"\b(humanities|liberal arts|school of liberal arts)\b", "Humanities"),
                (r"\b(cse|computer science and engineering)\b", "Computer Science and Engineering"),
                (r"\b(ee|electrical engineering)\b", "Electrical Engineering"),
                (r"\b(me|mechanical engineering)\b", "Mechanical Engineering"),
                (r"\b(chemistry)\b", "Chemistry"),
                (r"\b(physics)\b", "Physics"),
                (r"\b(mathematics|maths)\b", "Mathematics"),
            ]
            for pattern, canonical in department_patterns:
                if re.search(pattern, q):
                    department = canonical
                    break

            is_elective_query = "elective" in q
            asks_current = any(term in q for term in ("this semester", "current semester", "in the semester", "this term"))
            if department:
                dept_key = re.sub(r"[^a-z0-9]", "", department.lower())
                level_records = [c for c in self.courses if c.get("course_level") == level and
                                 re.sub(r"[^a-z0-9]", "", (c.get("department") or "").lower()) == dept_key]
                # Humanities is the School of Liberal Arts in the source export.
                if department == "Humanities":
                    level_records = [c for c in self.courses if c.get("course_level") == level and
                                     "liberalarts" in re.sub(r"[^a-z0-9]", "", (c.get("department") or "").lower())]
            else:
                level_records = [c for c in self.courses if c.get("course_level") == level]

            if not is_elective_query:
                count = len(level_records)
                scope = f" in {department}" if department else " across all departments"
                qualifier = " The level is inferred from the first digit of the numeric course code; records with wildcard codes such as 4XX are not included." 
                return {
                    "ok": True,
                    "answer": f"I found {count} course records at the {level}-level{scope} in the supplied course CSV.{qualifier}",
                    "sources": [f"courses(1).csv, rows {', '.join(str(c.get('source_row')) for c in level_records)}"],
                    "agent": "schedule_agent",
                }

            if not department:
                return {"ok": False, "reason": "Please specify a department to count 400-level electives; the CSV does not reliably mark elective categories for every course.", "agent": "schedule_agent"}
            result = count_elective_courses(self.courses, department=department, level=level)
            missing = result["missing_category_records"]
            details = ", ".join(f"{x.get('course_code')} {x.get('course_name')} (CSV row {x.get('source_row')})" for x in missing[:8])
            semester_note = " The CSV does not consistently identify the semester of offering, so this is not a current-semester count." if asks_current else ""
            if missing:
                known_count = result["count"]
                return {
                    "ok": True,
                    "answer": (
                        f"The CSV explicitly marks {known_count} {level}-level {department} course record(s) as electives. "
                        f"It also contains {len(missing)} matching {level}-level record(s) with no category, so the true elective count cannot be confirmed. "
                        f"Unclassified records: {details}.{semester_note}"
                    ),
                    "sources": ["courses(1).csv"],
                    "agent": "schedule_agent",
                    "partial": True,
                }
            answer = f"The supplied CSV contains {result['count']} records explicitly marked as electives for {department} at the {level}-level.{semester_note}"
            return {"ok": True, "answer": answer, "sources": ["courses(1).csv"], "agent": "schedule_agent"}

        matches = self._find(question)
        if not matches:
            code = extract_course_code(question, self.courses)
            reason = f"Course code {code} was not found in the supplied CSV." if code else "No sufficiently reliable course-name match was found in the supplied CSV."
            return {"ok": False, "reason": reason, "agent": "schedule_agent"}

        # Prefer exact code or course-name mention; otherwise refuse rather than use a weak fuzzy match.
        q_norm = re.sub(r"[^a-z0-9]", "", question.lower())
        strong = [m for m in matches if (
            (m.get("course_code") and re.sub(r"[^a-z0-9]", "", m["course_code"].lower()) in q_norm) or
            (m.get("course_name") and re.sub(r"[^a-z0-9]", "", m["course_name"].lower()) in q_norm)
        )]
        if strong:
            matches = strong
        else:
            return {"ok": False, "reason": "The course name did not exactly match a known course record. Include the exact course name or course code; I will not guess from a fuzzy match.", "agent": "schedule_agent"}

        # Collapse exact duplicates, but keep distinct offerings with different programmes/slots.
        unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for course in matches:
            key = (course.get("course_code", ""), course.get("course_name", ""), course.get("program", ""), course.get("slot", ""))
            unique[key] = course
        matches = list(unique.values())[:5]
        asks_timing = any(x in q for x in ("timing", "time", "when is the class", "class time"))
        answer = "\n\n".join(self._format_course(course, asks_timing=asks_timing) for course in matches)
        asks_current = any(term in q for term in ("this semester", "current semester", "this term"))
        if asks_current and len(matches) > 1:
            answer += (
                "\n\nThe CSV contains multiple programme-specific offerings with this course name. "
                "It does not identify one universal instructor for all students. Please use the programme/batch "
                "that applies to you to select the relevant offering."
            )
        return {"ok": True, "answer": answer, "sources": list(dict.fromkeys(_course_source(x) for x in matches)), "agent": "schedule_agent"}


class EmailAgent:
    def __init__(self, courses: list[dict[str, Any]], faculty: list[dict[str, str]], llm: OllamaClient):
        self.courses = courses
        self.faculty = faculty
        self.llm = llm

    def answer(self, question: str) -> dict[str, Any]:
        code = extract_course_code(question, self.courses)
        if not code:
            matches = search_courses(self.courses, query=question, limit=3)
            exact_names = [m for m in matches if m.get("course_name") and re.sub(r"[^a-z0-9]", "", m["course_name"].lower()) in re.sub(r"[^a-z0-9]", "", question.lower())]
            if len(exact_names) == 1:
                code = exact_names[0].get("course_code", "")
        if not code:
            return {"ok": False, "reason": "Please include the course code or exact course name so the instructor can be looked up without guessing.", "agent": "email_agent"}

        lookup = instructor_email_for_course(self.courses, self.faculty, code)
        if not lookup.get("ok"):
            return {"ok": False, "reason": lookup.get("reason", "Instructor email lookup failed."), "agent": "email_agent"}
        course = lookup["course"]
        instructors = lookup["instructors"]
        recipient_lines = [f"{p['name']} <{p['email']}>" for p in instructors]
        sources = [_course_source(course)] + [f"faculty_emails.csv: {p['name']} record; {p.get('source_url', '')}" for p in instructors]

        q_lower = question.lower()
        is_draft = (
            any(term in q_lower for term in ("write", "draft", "compose", "prepare", "email to", "sick leave", "leave for", "makeup", "make-up", "permission"))
            or ("email" in q_lower and any(term in q_lower for term in ("help", "request", "ask", "instructor")))
        )
        if not is_draft:
            answer = (
                f"Course: {course.get('course_name', '')} ({course.get('course_code', '')})\n"
                f"Instructor(s): {', '.join(p['name'] for p in instructors)}\n"
                f"Email: {', '.join(p['email'] for p in instructors)}\n"
                f"Course data source: {_course_source(course)}"
            )
            return {"ok": True, "answer": answer, "sources": sources, "agent": "email_agent"}

        duration_match = re.search(r"\bfor\s+(\d+)\s+days?\b", question, flags=re.IGNORECASE)
        duration = f"{duration_match.group(1)} day(s)" if duration_match else "the requested period"
        is_makeup = "makeup" in q_lower or "make-up" in q_lower
        is_leave = any(term in q_lower for term in ("sick leave", "leave for", "need leave", "request leave", "absent", "absence"))
        subject_label = "Make-up Examination" if is_makeup else ("Leave" if is_leave else "Assistance")
        subject = f"Request for {subject_label} – {course.get('course_code', code)}"
        recipient = "; ".join(recipient_lines)
        prompt = (
            "Draft a polite, concise email for an IIT Jodhpur student to the course instructor. "
            "Do not invent dates, student name, roll number, medical details, or approvals. "
            "Use bracketed placeholders for information not supplied. Do not claim that leave has been approved. "
            "Return only the email body, including an appropriate salutation and sign-off.\n\n"
            f"Course: {course.get('course_name', '')} ({course.get('course_code', code)})\n"
            f"Instructor: {', '.join(p['name'] for p in instructors)}\n"
            f"Request: {'permission to take a make-up examination' if is_makeup else ('leave' if is_leave else 'academic help/assistance')}\n"
            f"Duration: {duration if is_leave else 'Not applicable'}\n"
            "If this is a general request for help, politely ask the instructor for guidance and include a placeholder for the specific topic/question. "
            f"Student's wording: {question}"
        )
        try:
            body = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
        except OllamaError:
            # Keep the workflow usable even when the local model is stopped.
            reason = "absence due to illness" if "sick" in q_lower or "ill" in q_lower else "the circumstances described below"
            request_text = (
                f"I am writing to request permission to take a make-up examination for {course.get('course_code', code)}. "
                "I was unable to attend the examination due to [brief reason]. I can provide supporting documents if required."
                if is_makeup else
                f"I am writing to request leave for {duration} from [start date] to [end date] due to {reason}. "
                f"I will coordinate with the course team regarding any missed academic work for {course.get('course_code', code)}."
                if is_leave else
                f"I am taking {course.get('course_name', 'this course')} ({course.get('course_code', code)}) and would appreciate your guidance regarding [briefly describe the topic or issue]. "
                "Could you please let me know a suitable time or the appropriate way to discuss this?"
            )
            body = (
                f"Dear {', '.join(p['name'] for p in instructors)},\n\n"
                f"{request_text}\n\nThank you for your consideration.\n\nRegards,\n[Your Name]\n[Roll Number]"
            )
        answer = (
            f"To: {recipient}\nSubject: {subject}\n\n{body}\n\n"
            "Please review and fill in the bracketed details before sending. This is a draft only; no email has been sent."
        )
        return {"ok": True, "answer": answer, "sources": sources, "agent": "email_agent"}


class AcademicHelpdesk:
    """A small orchestrator that routes queries to three specialist agents."""

    OUT_OF_SCOPE_TERMS = (
        "dining", "food options", "mess menu", "restaurant", "cafeteria", "transport",
        "reach the campus", "route to campus", "taxi", "bus route", "weather", "tourist",
        "hotel", "accommodation booking", "movie", "sports score",
    )
    ACADEMIC_TERMS = (
        "academic", "attendance", "exam", "examination", "major", "minor", "makeup", "make-up",
        "credit", "degree", "b.tech", "btech", "curriculum", "course", "slot", "timetable",
        "time table", "schedule", "semester", "registration", "elective", "programme", "program",
        "instructor", "faculty", "professor", "syllabus", "academic calendar", "leave", "course code",
    )

    def __init__(self):
        self.llm = OllamaClient()
        self.rag = AcademicRAG(self.llm)
        self.courses = load_courses(config.COURSES_CSV)
        self.faculty = load_faculty(config.FACULTY_CSV)
        self.policy_agent = PolicyAgent(self.rag, self.llm)
        self.schedule_agent = ScheduleAgent(self.courses)
        self.email_agent = EmailAgent(self.courses, self.faculty, self.llm)
        self.logger = RefusalLogger()

    def route(self, question: str) -> str:
        q = question.lower()
        if any(term in q for term in self.OUT_OF_SCOPE_TERMS):
            return "out_of_scope"
        has_code = bool(extract_course_code(question, self.courses))
        email_lookup = any(term in q for term in ("email id", "email address", "email of", "instructor email", "faculty email", "contact details"))
        email_draft = (
            any(term in q for term in ("write an email", "draft an email", "compose an email", "email to the instructor", "write and email", "write an email to", "send an email"))
            or ("email" in q and "instructor" in q and any(term in q for term in ("write", "draft", "compose", "help", "request", "ask")))
        )
        if email_lookup or email_draft:
            return "email_agent"
        count_query = (
            any(term in q for term in ("how many", "count", "number of"))
            and re.search(r"\b(100|200|300|400|500|600|700|800)\s*[- ]?level\b|\b([1-8])00\s*[- ]?level\b", q)
            and "credit" not in q
        )
        schedule_query = any(term in q for term in (
            "slot", "timing", "class time", "when is the class", "when does the class",
            "timetable", "time table", "course offered", "offered this semester",
            "instructor of", "who teaches", "who is teaching", "who teaches the",
            "who is the instructor", "who is taking", "teacher of", "taught by",
            "course code", "course details",
        ))
        normalized_question = norm(question)
        mentions_known_course = any(
            course.get("course_name")
            and len(norm(course.get("course_name"))) >= 7
            and norm(course.get("course_name")) in normalized_question
            for course in self.courses
        )
        structured_course_detail = any(term in q for term in (
            "credit", "l-t-p", "ltp", "instructor", "who teaches", "teaching", "teacher",
            "taught by", "slot", "timing", "class time", "course code", "course details", "what is the code"
        ))
        if count_query or schedule_query or has_code or (mentions_known_course and structured_course_detail):
            return "schedule_agent"
        if any(term in q for term in self.ACADEMIC_TERMS):
            return "policy_agent"
        return "out_of_scope"

    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"ok": False, "answer": "Please enter an academic question.", "agent": "orchestrator"}
        agent_name = self.route(question)
        if agent_name == "out_of_scope":
            reason = "The question is outside the academic-helpdesk scope or is not supported by the uploaded academic sources."
            self.logger.log(question, "orchestrator", reason)
            return {
                "ok": False,
                "answer": "I do not have reliable information for this question in the IIT Jodhpur academic documents and datasets provided. I cannot answer it without guessing.",
                "agent": "orchestrator",
                "refusal": True,
            }

        agent = {
            "policy_agent": self.policy_agent,
            "schedule_agent": self.schedule_agent,
            "email_agent": self.email_agent,
        }[agent_name]
        result = agent.answer(question)
        if not result.get("ok") and not result.get("system_error"):
            self.logger.log(question, agent_name, result.get("reason", "Insufficient evidence"))
            result["answer"] = (
                "I could not answer this reliably from the supplied IIT Jodhpur sources. "
                + result.get("reason", "The available data is insufficient.")
            )
            result["refusal"] = True
        elif result.get("system_error"):
            result["answer"] = result.get("reason", "A local service is unavailable.")
        return result
