import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from calibration import ReferenceRecord, freeze
from measurements import record
from phase_evidence import initialize_station, issue, issue_turn
from receiving import accept
from review import get_report, get_turn_report, initialize_center, sign, sign_turn

RUN = "pump-turn"
READINGS = (("low", 1, 0),)
WEIGHTS = {"low": 1}
POSITIVE_REAL = (1, 0)
NEGATIVE_REAL = (-1, 0)
POSITIVE_IMAG = (0, 1)
WORK_ORDERS = (
    ("right-half", (POSITIVE_REAL,), 2, "only -1"),
    ("left-half", (NEGATIVE_REAL,), 2, "only +1"),
    ("quarter-arc", (POSITIVE_REAL, POSITIVE_IMAG), 2, "closed arc from +i to -1"),
    ("opposite-points", (POSITIVE_REAL, NEGATIVE_REAL), 2, "the two points +/-1"),
    ("opposite-two-arcs", (POSITIVE_REAL, NEGATIVE_REAL), 3, "two arcs, |Im| <= 1/2"),
    ("opposite-empty", (POSITIVE_REAL, NEGATIVE_REAL), 1, "empty"),
    ("opposite-full", (POSITIVE_REAL, NEGATIVE_REAL), 4, "full circle"),
)


def prepare_analysis(db, station, *, anchored=False, previous_revision_id=None, readings=READINGS):
    revision_id = record(db, station, RUN, readings, previous_revision_id)
    reference = ReferenceRecord(revision_id, POSITIVE_REAL) if anchored else None
    return freeze(db, revision_id, reference)


@contextmanager
def reject_commit(db):
    denied = []

    def authorize(action, operation, unused, schema, trigger):
        if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT":
            denied.append(operation)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    db.set_authorizer(authorize)
    try:
        yield denied
    finally:
        db.set_authorizer(None)


def check_seed():
    """The original V1 low-band ruling and the SQLite commit-reject fixture."""
    with closing(sqlite3.connect(":memory:", cached_statements=0)) as center:
        initialize_center(center)
        with closing(sqlite3.connect(":memory:")) as left, closing(sqlite3.connect(":memory:")) as right:
            initialize_station(left)
            initialize_station(right)
            a = issue(left, prepare_analysis(left, "A").analysis_id)
            b = issue(right, prepare_analysis(right, "B", anchored=True).analysis_id)
            receipts = tuple(accept(center, item.exchange, item.artifact_id) for item in (a, b))
            assert accept(center, a.exchange, a.artifact_id) == receipts[0]
        ids = tuple(receipt.receipt_id for receipt in receipts)
        original = sign(center, ids, WEIGHTS)
        assert original.outcome.status == "undetermined" and original.outcome.energy is None
        assert get_report(center, original.report_id) == original
        before = tuple(center.iterdump())
        with reject_commit(center) as denied:
            try:
                sign(center, ids, WEIGHTS)
            except sqlite3.DatabaseError:
                assert denied
            else:
                raise AssertionError("the fixture did not interrupt the report commit")
        assert not center.in_transaction
        assert tuple(center.iterdump()) == before
        retry = sign(center, ids, WEIGHTS)
        assert retry.report_id != original.report_id
        assert get_report(center, retry.report_id) == retry
        print("V1 low-band review:", original.outcome.status, "energy:", original.outcome.energy)
        print("Commit rejected, previous report preserved and retry stored:", retry.outcome.status)


def _open_station(path, station):
    db = sqlite3.connect(path)
    initialize_station(db)
    return db


def run_turn_workflow():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        left_path = folder / "left.db"
        right_path = folder / "right.db"
        center_path = folder / "center.db"

        with closing(_open_station(left_path, "A")) as left, \
                closing(_open_station(right_path, "B")) as right:
            left_analysis = prepare_analysis(left, "A")
            right_analysis = prepare_analysis(right, "B", anchored=True)
            right_artifact = issue(right, right_analysis.analysis_id)

        with closing(sqlite3.connect(center_path)) as center:
            initialize_center(center)
            right_receipt = accept(center, right_artifact.exchange, right_artifact.artifact_id)

            print("\nTurn work orders after both station libraries close:")
            report_ids = []
            for name, normals, limit, description in WORK_ORDERS:
                with closing(sqlite3.connect(left_path)) as left:
                    initialize_station(left)
                    artifact = issue_turn(left, left_analysis.analysis_id, normals)
                    # Reissuing the same source and window returns the same piece.
                    assert issue_turn(left, left_analysis.analysis_id, normals).artifact_id \
                        == artifact.artifact_id
                receipt = accept(center, artifact.exchange, artifact.artifact_id)
                # A repeated delivery keeps the first receipt id and exchange.
                assert accept(center, artifact.exchange, artifact.artifact_id) == receipt
                report = sign_turn(
                    center,
                    (receipt.receipt_id, right_receipt.receipt_id),
                    WEIGHTS, limit,
                )
                report_ids.append((name, report["report_id"]))
                print(f"  {name:18s} limit={limit} U^2 -> {description}")
                print("    " + json.dumps(report["directions"], separators=(",", ":")))

            # Re-issuing the same artifact id with another phase window is refused.
            with closing(sqlite3.connect(left_path)) as left:
                initialize_station(left)
                original = issue_turn(left, left_analysis.analysis_id, [POSITIVE_REAL])
                envelope = json.loads(original.exchange)
                envelope["body"]["phase_window"] = {
                    "kind": "points", "points": [["0", "-1"], ["0", "1"]]
                }
                before = center.execute("SELECT count(*) FROM receipts").fetchone()[0]
                try:
                    accept(center, json.dumps(envelope), original.artifact_id)
                except ValueError:
                    pass
                else:
                    raise AssertionError("altered window under a known id must be rejected")
                assert center.execute("SELECT count(*) FROM receipts").fetchone()[0] == before

            # A same-station correction produces a new analysis, piece and report,
            # while every earlier report stays retrievable by its own id.
            with closing(sqlite3.connect(left_path)) as left:
                initialize_station(left)
                corrected_analysis = prepare_analysis(
                    left, "A", previous_revision_id=left_analysis.revision_id,
                    readings=(("low", -1, 0),),
                )
                replacement = issue_turn(left, corrected_analysis.analysis_id, [POSITIVE_REAL])
            replacement_receipt = accept(center, replacement.exchange, replacement.artifact_id)
            changed = sign_turn(
                center,
                (replacement_receipt.receipt_id, right_receipt.receipt_id),
                WEIGHTS, 2,
            )
            print("\nAfter the A-station correction the new conclusion is:",
                  json.dumps(changed["directions"], separators=(",", ":")))

            # Commit-failure fixture on the turn ledger: no new row, then retry works.
            snapshot = tuple(center.iterdump())
            with reject_commit(center) as denied:
                try:
                    sign_turn(center,
                              (replacement_receipt.receipt_id, right_receipt.receipt_id),
                              WEIGHTS, 2)
                except sqlite3.DatabaseError:
                    assert denied
                else:
                    raise AssertionError("turn commit was not interrupted")
            assert not center.in_transaction
            assert tuple(center.iterdump()) == snapshot
            retried = sign_turn(center,
                                (replacement_receipt.receipt_id, right_receipt.receipt_id),
                                WEIGHTS, 2)
            assert get_turn_report(center, retried["report_id"]) == retried

        # Reopen the center library and reclaim every stored snapshot by id.
        with closing(sqlite3.connect(center_path)) as reopened:
            initialize_center(reopened)
            for name, report_id in report_ids:
                stored = get_turn_report(reopened, report_id)
                assert stored is not None and stored["report_id"] == report_id
                assert stored["receipt"]["report_id"] == report_id
                assert [parent["source"]["station"] for parent in stored["parents"]] == ["A", "B"]
            assert get_turn_report(reopened, changed["report_id"]) == changed
            assert get_turn_report(reopened, "unknown-id") is None
            print("\nReopened center library: all", len(report_ids) + 1,
                  "turn reports reclaimed with request, parents, directions and receipt.")


def main():
    check_seed()
    run_turn_workflow()


if __name__ == "__main__":
    main()
