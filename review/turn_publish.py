import sqlite3
from collections.abc import Mapping

from .turn_ledger import commit_turn_report
from .turn_request import build_turn_report, validate_turn_request


def sign_turn(
    conn: sqlite3.Connection,
    receipt_ids: tuple[str, str] | list[str],
    weights: Mapping[str, object],
    energy_limit: object,
) -> dict:
    request, left_item, right_item = validate_turn_request(
        conn, receipt_ids, weights, energy_limit
    )
    report = build_turn_report(
        request, left_item, right_item, left_item.evidence, right_item.evidence
    )
    return commit_turn_report(conn, report)
