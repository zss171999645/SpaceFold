"""Check output publication against operating-system races and process interruption."""

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from spacefold._atomic import _rename_exclusive, result_directory


class ResultDirectoryTests(unittest.TestCase):
    def test_interrupted_rename_leaves_the_destination_available_for_retry(self):
        worker = """
import os
import sys
from pathlib import Path
from spacefold._atomic import result_directory

destination = Path(sys.argv[1])
def interrupt(event, args):
    if event == "os.rename" and os.fsdecode(args[1]) == str(destination):
        os._exit(73)
sys.addaudithook(interrupt)
with result_directory(destination) as stage:
    (stage / "result.txt").write_text("complete")
"""
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp) / "probe"
            probe.mkdir()
            if not _rename_exclusive(probe, Path(temp) / "moved"):
                self.skipTest("the filesystem requires a separate directory reservation")
            output = Path(temp) / "output"
            proc = subprocess.run([sys.executable, "-B", "-c", worker, str(output)], timeout=15)
            self.assertEqual(proc.returncode, 73)
            self.assertFalse(os.path.lexists(output))
            with result_directory(output) as stage:
                (stage / "result.txt").write_text("retried")
            self.assertEqual((output / "result.txt").read_text(), "retried")

    def test_directory_created_by_someone_else_during_work_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            with self.assertRaises(FileExistsError), result_directory(output) as stage:
                (stage / "result.txt").write_text("complete")
                output.mkdir()
                reserved_inode = output.stat().st_ino
            self.assertEqual(output.stat().st_ino, reserved_inode)
            self.assertEqual(list(output.iterdir()), [])

    def test_concurrent_writers_publish_only_the_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            ready = Barrier(2)

            def write(name):
                try:
                    with result_directory(output) as stage:
                        (stage / name).write_text(name)
                        ready.wait(timeout=5)
                except FileExistsError:
                    return None
                return name

            with ThreadPoolExecutor(max_workers=2) as pool:
                winners = [name for name in pool.map(write, ("first", "second")) if name]
            self.assertEqual(len(winners), 1)
            self.assertEqual(list(output.iterdir()), [output / winners[0]])
            self.assertEqual((output / winners[0]).read_text(), winners[0])
