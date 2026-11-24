import json
import sqlite3
from uuid import uuid4


def initialize_turn_report_store(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS turn_reports (
            report_id TEXT PRIMARY KEY NOT NULL,
            snapshot TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS turn_receipts (
            receipt_id TEXT PRIMARY KEY NOT NULL,
            report_id TEXT NOT NULL UNIQUE REFERENCES turn_reports(report_id)
        )
    ''')


def commit_turn_report(conn: sqlite3.Connection, report: dict) -> dict:
    report_id = uuid4().hex
    receipt_id = uuid4().hex
    snapshot = dict(report)
    snapshot['report_id'] = report_id
    snapshot['receipt'] = {'receipt_id': receipt_id, 'report_id': report_id}
    conn.execute('BEGIN IMMEDIATE')
    with conn:
        conn.execute(
            'INSERT INTO turn_reports (report_id, snapshot) VALUES (?, ?)',
            (report_id, json.dumps(snapshot, ensure_ascii=True, sort_keys=True)),
        )
        conn.execute(
            'INSERT INTO turn_receipts (receipt_id, report_id) VALUES (?, ?)',
            (receipt_id, report_id),
        )
    return snapshot


def load_turn_report(conn: sqlite3.Connection, report_id: str) -> dict | None:
    row = conn.execute(
        'SELECT snapshot FROM turn_reports WHERE report_id = ?', (report_id,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])
