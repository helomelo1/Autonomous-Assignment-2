"""Read the supplied, unmodified IITJ CSV exports and expose safe lookups."""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
        return ""
    return re.sub(r"\s+", " ", text)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


ALIASES: dict[str, set[str]] = {
    "course_code": {"coursecode", "subjectcode", "code"},
    "course_name": {"coursename", "coursetitle", "nameofcourse", "nameofthecourse", "subjectname", "name"},
    "ltp": {"ltp", "ltpstructure", "creditltp", "ltpcredits", "creditltp"},
    "credits": {"credits", "credit", "noofcredits"},
    "instructor": {
        "instructor", "instructors", "instructorsname", "instructorname",
        "nameoffaculty", "instructorcoordinator", "coursecoordinator",
        "coordinator", "faculty", "facultyname",
    },
    "co_instructors": {"coinstructors", "coinstructor"},
    "slot": {"slot", "slots"},
    "preferred_slot": {"preferredslot", "preferredslots"},
    "lab_slot": {"labslot", "labtutorial", "labtutorialslot"},
    "category": {"category", "coursecategory", "categorypcpeic"},
    "course_type": {"coursetype", "type"},
    "program": {"program", "programme", "offeredfor", "batch"},
    "year": {"year", "semester", "offeredsemester"},
    "department": {"department", "dept"},
    "timing": {"timing", "time", "classtime", "classtiming", "lecturetime", "lecturetiming"},
    "remarks": {"remarks", "remark", "comment", "comments"},
}


def canonical_header(value: str) -> str | None:
    key = norm(value)
    for canonical, aliases in ALIASES.items():
        if key in aliases:
            return canonical
    return None


DEPARTMENT_PREFIXES = {
    "LAL": "School of Liberal Arts",
    "LAN": "School of Liberal Arts",
    "LAP": "School of Liberal Arts",
    "DHL": "School of Liberal Arts",
    "DHP": "School of Liberal Arts",
    "CSL": "Computer Science and Engineering",
    "CSP": "Computer Science and Engineering",
    "EEL": "Electrical Engineering",
    "EEP": "Electrical Engineering",
    "EEN": "Electrical Engineering",
    "EET": "Electrical Engineering",
    "ECL": "Electronics Engineering",
    "ECP": "Electronics Engineering",
    "ECT": "Electronics Engineering",
    "AEL": "Aerospace Engineering",
    "AEP": "Aerospace Engineering",
    "CIL": "Civil and Infrastructure Engineering",
    "CIP": "Civil and Infrastructure Engineering",
    "MEL": "Mechanical Engineering",
    "MEP": "Mechanical Engineering",
    "MEN": "Mechanical Engineering",
    "MED": "Mechanical Engineering",
    "MTL": "Materials Engineering",
    "MTN": "Materials Engineering",
    "MTO": "Materials Engineering",
    "CHL": "Chemical Engineering",
    "CYL": "Chemistry",
    "CYP": "Chemistry",
    "PHL": "Physics",
    "PHO": "Physics",
    "PHT": "Physics",
    "MAL": "Mathematics",
    "MAP": "Mathematics",
    "BBL": "Bioscience and Bioengineering",
    "BBP": "Bioscience and Bioengineering",
    "MSL": "School of Management and Entrepreneurship",
    "MSN": "School of Management and Entrepreneurship",
    "MSO": "School of Management and Entrepreneurship",
}


def department_from_code(code: str) -> str:
    code = re.sub(r"\s+", "", clean(code)).upper()
    for prefix in sorted(DEPARTMENT_PREFIXES, key=len, reverse=True):
        if code.startswith(prefix):
            return DEPARTMENT_PREFIXES[prefix]
    return ""


def course_level_from_code(code: str) -> int | None:
    code = re.sub(r"\s+", "", clean(code)).upper()
    match = re.match(r"^[A-Z]{2,4}(\d)", code)
    return int(match.group(1)) * 100 if match else None


def _header_map(row: list[str]) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for index, value in enumerate(row):
        field = canonical_header(value)
        if field and field not in mapped:
            mapped[field] = index
    return mapped


def _looks_like_header(mapping: dict[str, int]) -> bool:
    return ("course_code" in mapping or "course_name" in mapping) and len(mapping) >= 3


def _get(row: list[str], mapping: dict[str, int], field: str) -> str:
    index = mapping.get(field)
    if index is None or index >= len(row):
        return ""
    return clean(row[index])


def load_courses(path: str | Path) -> list[dict[str, Any]]:
    """Parse each concatenated table using its own header; never rewrite the CSV."""
    records: list[dict[str, Any]] = []
    current_map: dict[str, int] | None = None
    current_headers: list[str] = []
    section_context = ""
    seen: set[tuple[str, str, str, str]] = set()

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    for row_number, row in enumerate(rows, start=1):
        row = [clean(cell) for cell in row]
        nonempty = [cell for cell in row if cell]
        if not nonempty:
            continue

        mapping = _header_map(row)
        if _looks_like_header(mapping):
            current_map = mapping
            current_headers = row
            continue

        # Standalone section/title rows appear between some tables.
        if len(nonempty) == 1 and (not current_map or not any(k in mapping for k in ("course_code", "course_name"))):
            title = nonempty[0]
            if len(title) > 2 and not re.fullmatch(r"\d+", title):
                section_context = title
            continue

        if not current_map:
            continue

        values = {
            field: _get(row, current_map, field)
            for field in ALIASES
        }
        code = values["course_code"]
        name = values["course_name"]
        if not code and not name:
            continue
        # Skip malformed rows that contain only a serial number or table noise.
        if name and norm(name) in {"coursename", "coursetitle", "coursecode", "coursecategory"}:
            continue

        raw_fields: dict[str, str] = {}
        for index, header in enumerate(current_headers):
            if header and index < len(row) and row[index]:
                raw_fields[header] = row[index]

        department = values["department"] or department_from_code(code)
        category = values["category"] or values["course_type"]
        level = course_level_from_code(code)
        # The semester/program labels are retained verbatim; do not infer semester from a row number.
        record = {
            "course_code": code,
            "course_name": name,
            "ltp": values["ltp"],
            "credits": values["credits"],
            "instructor": values["instructor"],
            "co_instructors": values["co_instructors"],
            "slot": values["slot"],
            "preferred_slot": values["preferred_slot"],
            "lab_slot": values["lab_slot"],
            "category": category,
            "course_type": values["course_type"],
            "program": values["program"],
            "year_or_semester": values["year"],
            "department": department,
            "timing": values["timing"],
            "remarks": values["remarks"],
            "section_context": section_context,
            "course_level": level,
            "source_row": row_number,
            "raw_fields": raw_fields,
        }
        key = (
            norm(code), norm(name), norm(values["program"] or section_context),
            norm(values["slot"]),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def search_courses(
    courses: list[dict[str, Any]], query: str = "", course_code: str = "", limit: int = 8
) -> list[dict[str, Any]]:
    query_n = norm(query)
    code_n = norm(course_code)
    if code_n:
        exact = [c for c in courses if norm(c.get("course_code")) == code_n]
        return exact[:limit]
    if not query_n:
        return []

    ranked: list[tuple[float, dict[str, Any]]] = []
    for course in courses:
        name = norm(course.get("course_name"))
        code = norm(course.get("course_code"))
        haystack = " ".join(norm(course.get(k, "")) for k in ("course_name", "course_code", "instructor", "program"))
        if not name and not code:
            continue
        if query_n == name or query_n == code:
            score = 1.0
        elif name and (query_n in name or name in query_n):
            score = 0.96
        elif query_n in haystack:
            score = 0.88
        else:
            score = SequenceMatcher(None, query_n, name).ratio()
        if score >= 0.64:
            ranked.append((score, course))
    ranked.sort(key=lambda item: (item[0], bool(item[1].get("course_code"))), reverse=True)
    return [course for _, course in ranked[:limit]]


def _normalize_department(value: str) -> str:
    n = norm(value)
    if n in {"humanities", "humanitiesandsocialsciences", "schoolofliberalarts", "liberalarts"}:
        return "schoolofliberalarts"
    aliases = {
        "cse": "computerscienceandengineering",
        "computer science": "computerscienceandengineering",
        "ee": "electricalengineering",
        "electrical": "electricalengineering",
    }
    return aliases.get(value.lower().strip(), n)


def count_elective_courses(
    courses: list[dict[str, Any]], department: str, level: int | None = None,
    semester: str = "", category_terms: tuple[str, ...] = ("elective", "pe", "oe", "se", "me")
) -> dict[str, Any]:
    """Count only records with explicit evidence for every requested filter."""
    wanted_department = _normalize_department(department)
    matches: list[dict[str, Any]] = []
    missing_category: list[dict[str, Any]] = []
    missing_semester: list[dict[str, Any]] = []
    for course in courses:
        course_department = _normalize_department(course.get("department", ""))
        if course_department != wanted_department:
            continue
        if level is not None and course.get("course_level") != level:
            continue
        category = clean(course.get("category", "")).lower()
        category_key = norm(category)
        explicit_elective_categories = {"pe", "oe", "se", "me", "elective", "openelective", "professionalelective", "technicalelective", "specializationelective"}
        elective = category_key in explicit_elective_categories or "elective" in category_key
        if not category:
            missing_category.append(course)
            continue
        if not elective:
            # An explicitly non-elective category is valid evidence to exclude this row.
            continue
        if semester:
            hay = " ".join([
                clean(course.get("year_or_semester", "")), clean(course.get("program", "")),
                clean(course.get("section_context", "")),
            ]).lower()
            if semester.lower() not in hay:
                missing_semester.append(course)
                continue
        matches.append(course)
    return {
        "count": len(matches),
        "matches": matches,
        "missing_category_records": missing_category,
        "missing_semester_evidence": missing_semester,
    }


def load_faculty(path: str | Path) -> list[dict[str, str]]:
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [clean(column) for column in df.columns]
    expected = {"name", "email", "department"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Faculty CSV must contain columns {sorted(expected)}; found {list(df.columns)}")
    return [
        {key: clean(row.get(key, "")) for key in ("name", "email", "department", "designation", "source_url")}
        for row in df.to_dict(orient="records")
        if clean(row.get("name")) and clean(row.get("email"))
    ]


def lookup_faculty(faculty: list[dict[str, str]], instructor_name: str) -> dict[str, Any] | None:
    target = norm(instructor_name)
    if not target:
        return None
    exact = [person for person in faculty if norm(person.get("name")) == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return {"ambiguous": True, "matches": exact}

    scored = sorted(
        ((SequenceMatcher(None, target, norm(person.get("name"))).ratio(), person) for person in faculty),
        key=lambda pair: pair[0], reverse=True,
    )
    if scored and scored[0][0] >= 0.92:
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.015:
            return {"ambiguous": True, "matches": [scored[0][1], scored[1][1]]}
        return scored[0][1]
    return None


def instructor_email_for_course(
    courses: list[dict[str, Any]], faculty: list[dict[str, str]], course_code: str
) -> dict[str, Any]:
    found = search_courses(courses, course_code=course_code)
    if not found:
        return {"ok": False, "reason": f"Course code {course_code} is not present in the supplied course CSV."}
    if len(found) > 1:
        return {"ok": False, "reason": f"More than one record matches {course_code}; instructor identity is ambiguous.", "courses": found}
    course = found[0]
    instructor = course.get("instructor", "")
    if not instructor:
        return {"ok": False, "reason": f"No instructor is recorded for {course_code}.", "course": course}
    # Split multi-instructor strings conservatively. Return all only when every identity resolves.
    names = re.split(r"\s*(?:;|/|\band\b|&)\s*", instructor, flags=re.IGNORECASE)
    people: list[dict[str, str]] = []
    for name in names:
        name = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
        if not name:
            continue
        person = lookup_faculty(faculty, name)
        if not person or person.get("ambiguous"):
            return {"ok": False, "reason": f"Could not uniquely match instructor '{name}' to the faculty email CSV.", "course": course}
        people.append(person)
    if not people:
        return {"ok": False, "reason": f"No faculty email match was found for instructor '{instructor}'.", "course": course}
    return {"ok": True, "course": course, "instructors": people}
