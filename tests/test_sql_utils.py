import hashlib
import importlib.util
import os
import pickle
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


UTILS_PATH = Path(__file__).resolve().parents[1] / 'part2' / 'utils.py'
SPEC = importlib.util.spec_from_file_location('a4_sql_utils', UTILS_PATH)
sql_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sql_utils)


class SqlExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.database = self.root / 'test.db'
        with sqlite3.connect(self.database) as connection:
            connection.execute('CREATE TABLE sample (value INTEGER)')
            connection.executemany('INSERT INTO sample VALUES (?)', [(1,), (2,), (3,)])
        self.database_patch = mock.patch.object(sql_utils, 'DB_PATH', str(self.database))
        self.database_patch.start()
        self.environment_patch = mock.patch.dict(os.environ, {'A4_SQL_THREADS': '2', 'A4_SQL_TIMEOUT_SECONDS': '1'})
        self.environment_patch.start()

    def tearDown(self):
        self.environment_patch.stop()
        self.database_patch.stop()
        self.directory.cleanup()

    def test_order_and_invalid_queries_preserved(self):
        records, errors = sql_utils.compute_records([
            'SELECT value FROM sample ORDER BY value',
            'this is invalid SQL',
            'SELECT value FROM sample WHERE value = 2',
            '',
        ])
        self.assertEqual(records, [[(1,), (2,), (3,)], [], [(2,)], []])
        self.assertEqual(errors[0], '')
        self.assertTrue(errors[1])
        self.assertEqual(errors[2], '')
        self.assertTrue(errors[3])

    def test_database_is_read_only_and_control_queries_denied(self):
        initial_digest = hashlib.sha256(self.database.read_bytes()).hexdigest()
        records, errors = sql_utils.compute_records([
            'UPDATE sample SET value = 99',
            'DROP TABLE sample',
            "ATTACH DATABASE ':memory:' AS other",
            'PRAGMA query_only = OFF',
            "SELECT load_extension('nonexistent')",
        ])
        self.assertEqual(records, [[], [], [], [], []])
        self.assertTrue(all(errors))
        self.assertEqual(hashlib.sha256(self.database.read_bytes()).hexdigest(), initial_digest)
        self.assertEqual(sql_utils.compute_record(0, 'SELECT COUNT(*) FROM sample')[1], [(3,)])

    def test_expensive_query_is_interrupted(self):
        query = 'WITH RECURSIVE numbers(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM numbers WHERE x < 100000000) SELECT SUM(x) FROM numbers'
        started = time.monotonic()
        query_id, records, error = sql_utils.compute_record(7, query, timeout_secs=0.01)
        self.assertEqual(query_id, 7)
        self.assertEqual(records, [])
        self.assertEqual(error, 'Query timed out')
        self.assertLess(time.monotonic() - started, 2)

    def test_export_roundtrip_and_metric_denominator(self):
        reference_sql = self.root / 'reference.sql'
        reference_records = self.root / 'reference.pkl'
        predicted_sql = self.root / 'predicted.sql'
        predicted_records = self.root / 'predicted.pkl'
        sql_utils.save_queries_and_records(['SELECT 1', 'SELECT 2'], reference_sql, reference_records)
        sql_utils.save_queries_and_records(['SELECT 1', 'invalid'], predicted_sql, predicted_records)
        with predicted_records.open('rb') as handle:
            saved = pickle.load(handle)
        self.assertIsInstance(saved, tuple)
        self.assertEqual(len(saved), 2)
        self.assertEqual(len(saved[0]), 2)
        self.assertEqual(len(saved[1]), 2)
        sql_em, record_em, record_f1, errors = sql_utils.compute_metrics(reference_sql, predicted_sql, reference_records, predicted_records)
        self.assertEqual(sql_em, 0.5)
        self.assertEqual(record_em, 0.5)
        self.assertAlmostEqual(record_f1, 0.5)
        self.assertEqual(len(errors), 2)
        self.assertTrue(errors[1])

    def test_count_mismatches_and_multiline_exports_rejected(self):
        reference_sql = self.root / 'reference.sql'
        predicted_sql = self.root / 'predicted.sql'
        records_path = self.root / 'records.pkl'
        reference_sql.write_text('SELECT 1\nSELECT 2\n')
        predicted_sql.write_text('SELECT 1\n')
        with self.assertRaisesRegex(ValueError, 'Reference/prediction counts differ'):
            sql_utils.compute_metrics(reference_sql, predicted_sql)
        with records_path.open('wb') as handle:
            pickle.dump(([[(1,)]], ['', '']), handle)
        with self.assertRaisesRegex(ValueError, 'SQL, record, and error counts differ'):
            sql_utils.load_queries_and_records(predicted_sql, records_path)
        with self.assertRaisesRegex(ValueError, 'without line breaks'):
            sql_utils.save_queries_and_records(['SELECT\n1'], self.root / 'bad.sql', self.root / 'bad.pkl')

    def test_supplied_set_and_empty_record_scoring_retained(self):
        self.assertEqual(sql_utils.compute_record_exact_match([[(1,), (1,)]], [[(1,)]]), 1)
        self.assertAlmostEqual(sql_utils.compute_record_F1([[]], [[]]), 1)
        self.assertEqual(sql_utils.compute_record_F1([[(1,)]], [[]]), 0)


if __name__ == '__main__':
    unittest.main()
