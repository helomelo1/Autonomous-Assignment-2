from pathlib import Path

from academic_helpdesk.csv_tools import (
    instructor_email_for_course,
    load_courses,
    load_faculty,
    search_courses,
)
from academic_helpdesk.agents import AcademicHelpdesk

ROOT = Path(__file__).resolve().parents[1]


def test_multi_header_course_csv_and_autonomous_systems():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    matches = search_courses(courses, query="What is the slot for Autonomous Systems?")
    assert matches
    assert matches[0]["course_code"] == "CSL7650"
    assert matches[0]["course_name"] == "Autonomous Systems"
    assert matches[0]["slot"] == "E"
    assert matches[0]["credits"] == "3"
    assert matches[0]["ltp"] == "3-0-0"


def test_does_not_guess_course_code_typo():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    faculty = load_faculty(ROOT / "data" / "faculty_emails.csv")
    result = instructor_email_for_course(courses, faculty, "CSL7570")
    assert result["ok"] is False


def test_instructor_email_lookup():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    faculty = load_faculty(ROOT / "data" / "faculty_emails.csv")
    result = instructor_email_for_course(courses, faculty, "CSL7650")
    assert result["ok"] is True
    assert result["instructors"][0]["email"] == "gharit@iitj.ac.in"


def test_course_rows_with_alternate_headers_are_parsed():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    item = next(x for x in courses if x["course_code"] == "EDL1010")
    assert item["course_name"] == "Evolution of Indian Education"
    assert item["category"] == "PC"
    assert item["slot"] == "CE"


def test_orchestrator_routes_out_of_scope():
    helpdesk = AcademicHelpdesk()
    assert helpdesk.route("What are the available dining options on campus?") == "out_of_scope"
    assert helpdesk.route("What is the slot for Autonomous Systems?") == "schedule_agent"
    assert helpdesk.route("How many Open Elective credits are required for B.Tech?") == "policy_agent"
    assert helpdesk.route("How many 400 level electives have been floated by Humanities this semester?") == "schedule_agent"
    assert helpdesk.route("What is the policy for a make-up examination?") == "policy_agent"


def test_preferred_slot_is_not_mislabelled_as_actual_slot():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    preferred_only = [item for item in courses if item.get("preferred_slot")]
    assert preferred_only
    assert any(not item.get("slot") for item in preferred_only)


def test_policy_agent_prioritizes_superseding_attendance_amendment():
    from academic_helpdesk.agents import PolicyAgent

    class FakeRAG:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k):
            self.calls.append(query)
            if "Circular 51" in query:
                return [{"text": "Revised attendance policy; 75 percent minimum; supersedes prior policy.",
                         "metadata": {"source": "circular_51.pdf", "page": 1, "chunk": 1}}]
            return [{"text": "Older attendance policy.",
                     "metadata": {"source": "old_regulations.pdf", "page": 4, "chunk": 1}}]

    class FakeLLM:
        def chat(self, messages):
            return "The revised attendance circular sets a 75% attendance expectation and supersedes the earlier policy [S1]."

    rag = FakeRAG()
    result = PolicyAgent(rag, FakeLLM()).answer("What is the attendance requirement to sit in the Major exam?")
    assert result["ok"] is True
    assert result["sources"] == ["circular_51.pdf, p. 1"]
    assert "Circular 51" in rag.calls[0]


def test_policy_agent_rejects_uncited_llm_answer():
    from academic_helpdesk.agents import PolicyAgent

    class FakeRAG:
        def retrieve(self, query, top_k):
            return [{"text": "A source excerpt.", "metadata": {"source": "rules.pdf", "page": 2, "chunk": 1}}]

    class FakeLLM:
        def chat(self, messages):
            return "The answer is 75 percent."

    result = PolicyAgent(FakeRAG(), FakeLLM()).answer("What is the attendance rule?")
    assert result["ok"] is False
    assert "citation" in result["reason"].lower()


def test_current_semester_elective_count_is_refused_without_semester_metadata():
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    from academic_helpdesk.agents import ScheduleAgent
    result = ScheduleAgent(courses).answer(
        "How many 400-level electives have been floated by the Humanities department this semester?"
    )
    assert result["ok"] is True
    assert result.get("partial") is True
    assert "not a current-semester count" in result["answer"].lower()


def test_course_code_extractor_ignores_years_but_keeps_unknown_codes_with_known_prefix():
    from academic_helpdesk.agents import extract_course_code
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    assert extract_course_code("under the 2024 examination policy", courses) == ""
    assert extract_course_code("in AY 2026-27", courses) == ""
    assert extract_course_code("CSL7570", courses) == "CSL7570"


def test_schedule_agent_does_not_answer_from_a_fuzzy_course_name_guess():
    from academic_helpdesk.agents import ScheduleAgent
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    result = ScheduleAgent(courses).answer("What is the slot for Autonomus Systems?")
    assert result["ok"] is False
    assert "exactly match" in result["reason"].lower()


def test_email_draft_request_routes_to_email_agent_even_with_course_code():
    helpdesk = AcademicHelpdesk()
    query = "I want to write and email to the course instructor of csl7650 for help"
    assert helpdesk.route(query) == "email_agent"


def test_generic_help_email_is_not_misclassified_as_leave():
    from academic_helpdesk.agents import EmailAgent

    class FakeLLM:
        def chat(self, messages, temperature=0.1):
            prompt = messages[-1]["content"]
            assert "academic help/assistance" in prompt
            return "Dear Professor,\\n\\nI would appreciate your guidance on [topic].\\n\\nRegards,\\n[Your Name]"

    courses = load_courses(ROOT / "data" / "courses(1).csv")
    faculty = load_faculty(ROOT / "data" / "faculty_emails.csv")
    agent = EmailAgent(courses, faculty, FakeLLM())
    result = agent.answer("I want to write and email to the course instructor of csl7650 for help")

    assert result["ok"] is True
    assert result["agent"] == "email_agent"
    assert "gharit@iitj.ac.in" in result["answer"]
    assert "Request for Assistance" in result["answer"]
    assert "no email has been sent" in result["answer"]
    assert "Request for Leave" not in result["answer"]


def test_400_level_course_count_is_deterministic_and_not_sent_to_policy():
    from academic_helpdesk.agents import ScheduleAgent
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    result = ScheduleAgent(courses).answer("how many courses are available as 400 level courses")
    assert result["ok"] is True
    assert "50 course records" in result["answer"]
    assert result["agent"] == "schedule_agent"


def test_cse_abbreviation_is_recognized_and_missing_categories_are_reported():
    from academic_helpdesk.agents import ScheduleAgent
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    result = ScheduleAgent(courses).answer("How many 400 level electives have been floated by CSE department?")
    assert result["ok"] is True
    assert result.get("partial") is True
    assert "2 matching 400-level record(s) with no category" in result["answer"]
    assert "CSL4010" in result["answer"] and "CSL4030" in result["answer"]


def test_humanities_count_reports_partial_evidence_and_missing_current_semester():
    from academic_helpdesk.agents import ScheduleAgent
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    result = ScheduleAgent(courses).answer(
        "How many 400 level electives have been floated by the Humanities department in this semester?"
    )
    assert result["ok"] is True
    assert result.get("partial") is True
    assert "LAL4070 Health Economics" in result["answer"]
    assert "not a current-semester count" in result["answer"]


def test_400_level_count_routes_to_schedule_agent():
    # Exercise the router without initializing Ollama or ChromaDB.
    helpdesk = object.__new__(AcademicHelpdesk)
    helpdesk.courses = load_courses(ROOT / "data" / "courses(1).csv")
    assert helpdesk.route("how many courses are available as 400 level courses") == "schedule_agent"
    assert helpdesk.route("How many 400 level electives have been floated by CSE department?") == "schedule_agent"


def test_artificial_intelligence_instructor_query_routes_to_schedule_agent():
    helpdesk = object.__new__(AcademicHelpdesk)
    helpdesk.courses = load_courses(ROOT / "data" / "courses(1).csv")
    query = "Who is teaching the Aritifical Intelligence course this semester?"
    assert helpdesk.route(query) == "schedule_agent"


def test_artificial_intelligence_returns_all_programme_specific_offerings():
    from academic_helpdesk.agents import ScheduleAgent
    courses = load_courses(ROOT / "data" / "courses(1).csv")
    result = ScheduleAgent(courses).answer(
        "Who is teaching the Artificial Intelligence course this semester?"
    )
    assert result["ok"] is True
    assert "Pratik Majumder" in result["answer"]
    assert "Deeksha Varshney" in result["answer"]
    assert "multiple programme-specific offerings" in result["answer"]
