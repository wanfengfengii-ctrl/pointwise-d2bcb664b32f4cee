import json
import math
import sqlite3
from fractions import Fraction

from calibration import ReferenceRecord, freeze
from phase_evidence import get_issued_artifact, issue, issue_turn
from phase_evidence.export import pack_exchange, unpack_exchange
from phase_evidence.identity import turn_artifact_identity
from receiving import accept, lookup_received
from review import get_turn_report, initialize_center, sign_turn

from tests import RUN, WorkbenchTestCase


POSITIVE_REAL = (1, 0)
NEGATIVE_REAL = (-1, 0)
POSITIVE_IMAG = (0, 1)


def radical_value(coordinate):
    return Fraction(coordinate['a']) + Fraction(coordinate['b']) * math.sqrt(int(coordinate['d']))


class TurnIssuanceTests(WorkbenchTestCase):
    def prepare(self, station='A', anchored=False):
        db = self.station(station)
        revision_id = record_simple(db, station)
        reference = ReferenceRecord(revision_id, POSITIVE_REAL) if anchored else None
        return db, freeze(db, revision_id, reference)

    def test_window_shapes(self):
        db, analysis = self.prepare()
        single = issue_turn(db, analysis.analysis_id, [POSITIVE_REAL])
        body = unpack_exchange(single.exchange).body
        self.assertEqual(set(body), {'bands', 'phase_window'})
        self.assertEqual(body['phase_window'],
                         {'kind': 'arc', 'start': ['0', '-1'], 'end': ['0', '1']})
        points = issue_turn(db, analysis.analysis_id, [POSITIVE_REAL, NEGATIVE_REAL])
        self.assertEqual(unpack_exchange(points.exchange).body['phase_window'],
                         {'kind': 'points', 'points': [['0', '-1'], ['0', '1']]})
        quarter = issue_turn(db, analysis.analysis_id, [POSITIVE_REAL, POSITIVE_IMAG])
        self.assertEqual(unpack_exchange(quarter.exchange).body['phase_window'],
                         {'kind': 'arc', 'start': ['1', '0'], 'end': ['0', '1']})

    def test_identity_and_reissue(self):
        db, analysis = self.prepare()
        first = issue_turn(db, analysis.analysis_id, [POSITIVE_REAL])
        envelope = unpack_exchange(first.exchange)
        self.assertEqual(envelope.version, 2)
        self.assertEqual(turn_artifact_identity(envelope.source, envelope.body), first.artifact_id)
        reissued = issue_turn(db, analysis.analysis_id, [(Fraction(1), Fraction(0))])
        self.assertEqual(reissued, first)
        self.assertEqual(get_issued_artifact(db, first.artifact_id).exchange, first.exchange)
        # A different phase window under the same source gets a different id.
        other = issue_turn(db, analysis.analysis_id, [NEGATIVE_REAL])
        self.assertNotEqual(other.artifact_id, first.artifact_id)

    def test_requires_unanchored_analysis(self):
        left_db, _ = self.prepare('A')
        right_db, anchored = self.prepare('B', anchored=True)
        with self.assertRaisesRegex(ValueError, 'unanchored'):
            issue_turn(right_db, anchored.analysis_id, [POSITIVE_REAL])
        with self.assertRaisesRegex(ValueError, 'unknown fixed analysis'):
            issue_turn(left_db, 'missing', [POSITIVE_REAL])

    def test_rejects_bad_normals(self):
        db, analysis = self.prepare()
        for normals in ([], [POSITIVE_REAL, POSITIVE_IMAG, NEGATIVE_REAL],
                        [(2, 0)], [('x', 0)], POSITIVE_REAL, [(1, 0.5)],
                        [(True, 0)]):
            with self.subTest(normals=normals), self.assertRaises(ValueError):
                issue_turn(db, analysis.analysis_id, normals)
        self.assertEqual(db.execute('SELECT count(*) FROM turn_checks').fetchone()[0], 0)


def record_simple(db, station, rows=(('low', 1, 0),)):
    from measurements import record
    return record(db, station, RUN, rows)


class TurnReceivingTests(WorkbenchTestCase):
    def setUp(self):
        super().setUp()
        self.left_db, self.left_analysis = self._make('A', (('low', 1, 0),), False)
        self.right_db, self.right_analysis = self._make('B', (('low', 1, 0),), True)

    def _make(self, station, rows, anchored):
        db = self.station(station)
        from measurements import record
        revision_id = record(db, station, RUN, rows)
        reference = ReferenceRecord(revision_id, POSITIVE_REAL) if anchored else None
        return db, freeze(db, revision_id, reference)

    def test_accept_and_lookup(self):
        center = self.center()
        artifact = issue_turn(self.left_db, self.left_analysis.analysis_id, [POSITIVE_REAL])
        receipt = accept(center, artifact.exchange, artifact.artifact_id)
        duplicate = accept(center, artifact.exchange, artifact.artifact_id)
        self.assertEqual(duplicate, receipt)
        item = lookup_received(center, receipt.receipt_id)
        self.assertEqual(item.delivery.version, 2)
        self.assertEqual(item.delivery.exchange, artifact.exchange)
        self.assertEqual(item.source, artifact.source)

    def test_tampered_window_rejected_without_rows(self):
        center = self.center()
        artifact = issue_turn(self.left_db, self.left_analysis.analysis_id, [POSITIVE_REAL])
        envelope = unpack_exchange(artifact.exchange)
        tampered_body = dict(envelope.body)
        tampered_body['phase_window'] = {
            'kind': 'points', 'points': [['0', '-1'], ['0', '1']]
        }
        exchange = pack_exchange(type(envelope)(
            envelope.artifact_id, envelope.source, tampered_body, 2))
        with self.assertRaisesRegex(ValueError, 'identity'):
            accept(center, exchange, artifact.artifact_id)
        self.assertEqual(center.execute('SELECT count(*) FROM received_artifacts').fetchone()[0], 0)
        self.assertEqual(center.execute('SELECT count(*) FROM receipts').fetchone()[0], 0)

    def test_wrong_declared_identity(self):
        center = self.center()
        artifact = issue_turn(self.left_db, self.left_analysis.analysis_id, [POSITIVE_REAL])
        with self.assertRaisesRegex(ValueError, 'declared identity'):
            accept(center, artifact.exchange, '0' * 64)


class TurnSigningTests(WorkbenchTestCase):
    EXPECTED = {
        (POSITIVE_REAL,): {2: 'point -1'},
        (NEGATIVE_REAL,): {2: 'point +1'},
        (POSITIVE_REAL, POSITIVE_IMAG): {2: 'arc +i..-1'},
        (POSITIVE_REAL, NEGATIVE_REAL): {
            2: 'points +/-1',
            3: 'two arcs',
            1: 'empty',
            4: 'full',
        },
    }

    def setUp(self):
        super().setUp()
        self.left_db = self.station('A')
        self.right_db = self.station('B')
        center = self.center()
        self.center_db = center
        from measurements import record
        left_revision = record(self.left_db, 'A', RUN, (('low', 1, 0),))
        right_revision = record(self.right_db, 'B', RUN, (('low', 1, 0),))
        left_analysis = freeze(self.left_db, left_revision)
        right_analysis = freeze(
            self.right_db, right_revision, ReferenceRecord(right_revision, POSITIVE_REAL)
        )
        self.right_artifact = issue(self.right_db, right_analysis.analysis_id)
        self.right_receipt = accept(center, self.right_artifact.exchange,
                                    self.right_artifact.artifact_id)
        self.left_analysis = left_analysis

    def issue_and_accept(self, normals):
        artifact = issue_turn(self.left_db, self.left_analysis.analysis_id, normals)
        receipt = accept(self.center_db, artifact.exchange, artifact.artifact_id)
        return artifact, receipt

    def sign(self, normals, limit):
        _, receipt = self.issue_and_accept(normals)
        return sign_turn(
            self.center_db,
            (receipt.receipt_id, self.right_receipt.receipt_id),
            {'low': 1}, limit,
        )

    def test_work_order_table(self):
        report = self.sign([POSITIVE_REAL], 2)
        self.assertEqual(report['directions'],
                         {'kind': 'parts', 'arcs': [],
                          'points': [[{'a': '-1', 'b': '0', 'd': '0'},
                                      {'a': '0', 'b': '0', 'd': '0'}]]})
        report = self.sign([NEGATIVE_REAL], 2)
        self.assertEqual(report['directions']['points'][0][0]['a'], '1')

        quarter = self.sign([POSITIVE_REAL, POSITIVE_IMAG], 2)
        arc = quarter['directions']['arcs'][0]
        self.assertEqual(arc['start'], [{'a': '0', 'b': '0', 'd': '0'},
                                        {'a': '1', 'b': '0', 'd': '0'}])
        self.assertEqual(arc['end'], [{'a': '-1', 'b': '0', 'd': '0'},
                                      {'a': '0', 'b': '0', 'd': '0'}])

        opposite_two = self.sign([POSITIVE_REAL, NEGATIVE_REAL], 2)
        points = opposite_two['directions']['points']
        self.assertEqual({(p[0]['a'], p[1]['a']) for p in points}, {('-1', '0'), ('1', '0')})

        two_arcs = self.sign([POSITIVE_REAL, NEGATIVE_REAL], 3)
        self.assertEqual(len(two_arcs['directions']['arcs']), 2)
        for arc in two_arcs['directions']['arcs']:
            for endpoint in (arc['start'], arc['end']):
                x, y = radical_value(endpoint[0]), radical_value(endpoint[1])
                self.assertAlmostEqual(x * x + y * y, 1.0, places=9)
        # The exact endpoints carry the sqrt(3)/2 and 1/2 radical boundaries.
        flat = json.dumps(two_arcs['directions'])
        self.assertIn('"d": "3"', flat)
        self.assertIn('"1/2"', flat)

        self.assertEqual(self.sign([POSITIVE_REAL, NEGATIVE_REAL], 1)['directions'],
                         {'kind': 'empty'})
        self.assertEqual(self.sign([POSITIVE_REAL, NEGATIVE_REAL], 4)['directions'],
                         {'kind': 'full'})

    def test_report_shape_and_replay(self):
        report = self.sign([POSITIVE_REAL], 2)
        self.assertEqual(set(report), {'report_id', 'request', 'parents', 'directions', 'receipt'})
        self.assertEqual(report['request']['weights'], {'low': '1'})
        self.assertEqual(report['request']['energy_limit'], '2')
        self.assertEqual(len(report['request']['receipt_ids']), 2)
        self.assertEqual([parent['source']['station'] for parent in report['parents']], ['A', 'B'])
        self.assertEqual(report['receipt']['report_id'], report['report_id'])
        json.dumps(report)  # directly JSON serializable
        reloaded = get_turn_report(self.center_db, report['report_id'])
        self.assertEqual(reloaded, report)
        self.assertIsNone(get_turn_report(self.center_db, 'missing'))

    def test_role_and_input_rejection(self):
        _, left_receipt = self.issue_and_accept([POSITIVE_REAL])
        good_ids = (left_receipt.receipt_id, self.right_receipt.receipt_id)
        # reversed roles: V1 on the left
        with self.assertRaisesRegex(ValueError, 'V2'):
            sign_turn(self.center_db, (self.right_receipt.receipt_id, left_receipt.receipt_id),
                      {'low': 1}, 2)
        with self.assertRaisesRegex(ValueError, 'unknown receipt'):
            sign_turn(self.center_db, ('missing', self.right_receipt.receipt_id), {'low': 1}, 2)
        for weights, limit in (
            ({'low': -1}, 2), ({'low': 0}, 2), ({}, 2),
            ({'low': 0.25}, 2), ({'low': 'bad'}, 2), ({'low': 1}, -1),
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                sign_turn(self.center_db, good_ids, weights, limit)
        self.assertEqual(self.center_db.execute('SELECT count(*) FROM turn_reports').fetchone()[0], 0)
        self.assertEqual(self.center_db.execute('SELECT count(*) FROM turn_receipts').fetchone()[0], 0)

    def test_commit_failure_leaves_no_report(self):
        _, receipt = self.issue_and_accept([POSITIVE_REAL])
        ids = (receipt.receipt_id, self.right_receipt.receipt_id)
        before = tuple(self.center_db.iterdump())
        self.center_db.execute('''
            CREATE TEMP TRIGGER fail_turn_receipt BEFORE INSERT ON turn_receipts
            BEGIN
                SELECT RAISE(ABORT, 'turn receipt write rejected');
            END
        ''')
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'turn receipt write rejected'):
            sign_turn(self.center_db, ids, {'low': 1}, 2)
        self.assertFalse(self.center_db.in_transaction)
        self.assertEqual(tuple(self.center_db.iterdump()), before)
        self.center_db.execute('DROP TRIGGER fail_turn_receipt')
        report = sign_turn(self.center_db, ids, {'low': 1}, 2)
        self.assertEqual(get_turn_report(self.center_db, report['report_id']), report)

    def test_reopen_database_replays_snapshot(self):
        report = self.sign([POSITIVE_REAL, NEGATIVE_REAL], 3)
        report_id = report['report_id']
        path = self.center_db.execute('PRAGMA database_list').fetchone()[2]
        self.center_db.close()
        reopened = sqlite3.connect(path)
        self.addCleanup(reopened.close)
        initialize_center(reopened)
        self.assertEqual(get_turn_report(reopened, report_id), report)

    def test_corrected_analysis_issues_new_artifact_and_report(self):
        from measurements import record
        first = self.sign([POSITIVE_REAL], 2)
        corrected_revision = record(
            self.left_db, 'A', RUN, (('low', -1, 0),),
            self.left_analysis.revision_id,
        )
        corrected_analysis = freeze(self.left_db, corrected_revision)
        artifact = issue_turn(self.left_db, corrected_analysis.analysis_id, [POSITIVE_REAL])
        receipt = accept(self.center_db, artifact.exchange, artifact.artifact_id)
        second = sign_turn(
            self.center_db,
            (receipt.receipt_id, self.right_receipt.receipt_id),
            {'low': 1}, 2,
        )
        self.assertNotEqual(second['report_id'], first['report_id'])
        self.assertEqual(second['parents'][0]['artifact_id'], artifact.artifact_id)
        # The original report keeps its original snapshot.
        self.assertEqual(get_turn_report(self.center_db, first['report_id']), first)
