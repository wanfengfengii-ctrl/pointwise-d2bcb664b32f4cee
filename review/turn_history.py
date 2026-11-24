from .turn_ledger import load_turn_report


def get_turn_report(conn, report_id: str) -> dict | None:
    return load_turn_report(conn, report_id)
