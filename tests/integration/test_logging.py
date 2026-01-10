import os
from pathlib import Path

from typer.testing import CliRunner

from markit.cli.main import app

runner = CliRunner()


def test_task_logging_creation(tmp_path):
    # We need a valid input file
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello World", encoding="utf-8")

    # Change CWD to tmp_path so logs are created there
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Run convert command with dry-run
        result = runner.invoke(app, ["convert", str(input_file), "--dry-run"])
        assert result.exit_code == 0

        # Check if logs directory exists (default: .logs)
        log_dir = Path(".logs")
        assert log_dir.exists()

        # Check if log file exists (convert_*.log pattern)
        log_files = list(log_dir.glob("convert_*.log"))
        assert len(log_files) >= 1

        log_content = log_files[0].read_text(encoding="utf-8")

        # Check for Header (Task Configuration)
        assert "Task Configuration" in log_content

        # Run real conversion
        result = runner.invoke(app, ["convert", str(input_file)])
        assert result.exit_code == 0

        # There should be a new log file
        log_files = list(log_dir.glob("convert_*.log"))
        assert len(log_files) >= 2

        # Find the latest log
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        content = latest_log.read_text(encoding="utf-8")

        assert "Task Configuration" in content
        assert "Task Completed Successfully" in content
        assert "output_path" in content

    finally:
        os.chdir(original_cwd)
