from __future__ import annotations

from .rag import AcademicRAG


def main() -> None:
    rag = AcademicRAG()
    result = rag.ingest()
    print(f"Indexed {result['chunks']} chunks from {result['files']} source files.")
    if result["warnings"]:
        print("\nExtraction warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
        print("\nScanned attendance/examination amendments are covered by manual transcriptions in data/manual_transcriptions/.")


if __name__ == "__main__":
    main()
