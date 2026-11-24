import json
import sqlite3

from calibration.references import ReferenceQualification
from calibration.snapshots import get_fixed_analysis
from measurements.bands import require_name

from .export import ExchangeEnvelope, pack_exchange
from .identity import SourceBinding, turn_artifact_identity
from .issue import IssuedArtifact, get_issued_artifact
from .turn_response import TurnEvidence, build_turn_body, decode_turn_body
from .window import parse_normals


def initialize_turn_store(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS turn_checks (
            artifact_id TEXT PRIMARY KEY NOT NULL
                REFERENCES issued_artifacts(artifact_id),
            analysis_id TEXT NOT NULL REFERENCES fixed_analyses(analysis_id),
            normals TEXT NOT NULL
        )
    ''')


def issue_turn(conn: sqlite3.Connection, analysis_id: str, normals: object) -> IssuedArtifact:
    require_name(analysis_id, 'analysis id')
    parsed_normals = parse_normals(normals)
    analysis = get_fixed_analysis(conn, analysis_id)
    if analysis is None:
        raise ValueError(f'unknown fixed analysis {analysis_id!r}')
    if analysis.inspection.qualification is not ReferenceQualification.UNANCHORED:
        raise ValueError('turn artifacts require an unanchored fixed analysis')
    source = SourceBinding(analysis.station, analysis.revision_id, analysis.analysis_id)
    body = build_turn_body(analysis, parsed_normals)
    artifact_id = turn_artifact_identity(source, body)
    exchange = pack_exchange(ExchangeEnvelope(artifact_id, source, body, version=2))
    normal_payload = json.dumps([[str(value.real), str(value.imag)] for value in parsed_normals])
    conn.execute('BEGIN IMMEDIATE')
    with conn:
        conn.execute('''
            INSERT INTO issued_artifacts (artifact_id, analysis_id, exchange) VALUES (?, ?, ?)
            ON CONFLICT (artifact_id) DO NOTHING
        ''', (artifact_id, analysis_id, exchange))
        conn.execute('''
            INSERT INTO turn_checks (artifact_id, analysis_id, normals) VALUES (?, ?, ?)
            ON CONFLICT (artifact_id) DO NOTHING
        ''', (artifact_id, analysis_id, normal_payload))
        result = get_issued_artifact(conn, artifact_id)
    if result is None:
        raise sqlite3.IntegrityError('issued artifact is missing from the station ledger')
    return result


def get_turn_checks(conn: sqlite3.Connection, artifact_id: str) -> tuple[tuple[object, object], ...] | None:
    row = conn.execute('SELECT normals FROM turn_checks WHERE artifact_id = ?',
                       (artifact_id,)).fetchone()
    if row is None:
        return None
    return tuple((normal[0], normal[1]) for normal in json.loads(row[0]))


def verify_turn_delivery(envelope: ExchangeEnvelope) -> TurnEvidence:
    if envelope.version != 2:
        raise ValueError('turn exchange must use version 2')
    evidence = decode_turn_body(envelope.body)
    recomputed = turn_artifact_identity(envelope.source, envelope.body)
    if recomputed != envelope.artifact_id:
        raise ValueError('exchange identity does not match its source and turn evidence')
    return evidence
