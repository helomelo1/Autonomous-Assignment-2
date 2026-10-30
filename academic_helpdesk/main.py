from __future__ import annotations

from .agents import AcademicHelpdesk


def main() -> None:
    print("IIT Jodhpur Academic Helpdesk")
    print("Ask academic questions. Type :quit to exit.\n")
    helpdesk = AcademicHelpdesk()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {":quit", ":exit", "exit", "quit"}:
            print("Goodbye.")
            break
        result = helpdesk.ask(question)
        agent_labels = {
            "orchestrator": "Helpdesk",
            "policy_agent": "Policy Agent",
            "schedule_agent": "Schedule Agent",
            "email_agent": "Email Agent",
        }
        print(f"\n[{agent_labels.get(result.get('agent', 'orchestrator'), 'Helpdesk')}]")
        print(result.get("answer", "No answer was generated."))
        if result.get("sources"):
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        print()


if __name__ == "__main__":
    main()
