from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from monotm.cli import main
from monotm.config import RunConfig


class CLITests(unittest.TestCase):
    def test_config_command_roundtrips_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            self.assertEqual(main(["config", "--output", str(path)]), 0)
            self.assertEqual(RunConfig.from_dict(json.loads(path.read_text())).interpretation.k, 16)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["config", "--output", str(path)]), 1)

    def test_config_stdout_is_valid_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["config"]), 0)
        self.assertEqual(json.loads(out.getvalue())["mixture"]["k"], 256)

    def test_invalid_nested_config_is_a_user_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"mixture": null}')
            with redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(["run", "--input", "unused.csv", "--output", "unused", "--topics", "3", "--config", str(path)]), 1)
            self.assertIn("JSON object", error.getvalue())
