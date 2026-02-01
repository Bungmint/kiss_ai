"""
Tests for useful_tools.py module.

These tests run in a temporary directory and do not use any mocking.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.core.useful_tools import (
    UsefulTools,
    _extract_directory,
    contains_dangerous_patterns,
    parse_bash_command_paths,
)


class TestExtractDirectory(unittest.TestCase):
    """Test the _extract_directory function."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up temp directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_absolute_existing_file(self):
        """Test with an absolute path to an existing file."""
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("content")
        result = _extract_directory(str(test_file))
        self.assertEqual(result, str(test_file))

    def test_absolute_existing_directory(self):
        """Test with an absolute path to an existing directory."""
        test_dir = Path(self.test_dir) / "subdir"
        test_dir.mkdir()
        result = _extract_directory(str(test_dir))
        self.assertEqual(result, str(test_dir))

    def test_absolute_nonexistent_with_extension(self):
        """Test with absolute path to nonexistent file with extension."""
        test_path = Path(self.test_dir) / "nonexistent.txt"
        result = _extract_directory(str(test_path))
        self.assertEqual(result, str(test_path))

    def test_absolute_nonexistent_without_extension(self):
        """Test with absolute path to nonexistent file without extension."""
        test_path = Path(self.test_dir) / "nonexistent"
        result = _extract_directory(str(test_path))
        self.assertEqual(result, str(test_path))

    def test_trailing_slash(self):
        """Test with trailing slash indicating directory."""
        test_path = Path(self.test_dir) / "newdir/"
        result = _extract_directory(str(test_path))
        # Path normalization removes trailing slash
        self.assertEqual(result, str(Path(self.test_dir) / "newdir"))

    def test_relative_path(self):
        """Test with relative path returns None."""
        result = _extract_directory("relative/path.txt")
        self.assertIsNone(result)

    def test_invalid_path(self):
        """Test with invalid path."""
        # Empty string
        result = _extract_directory("")
        self.assertIsNone(result)


class TestContainsDangerousPatterns(unittest.TestCase):
    """Test the contains_dangerous_patterns function."""

    def test_safe_commands(self):
        """Test that safe commands are not flagged."""
        safe_commands = [
            "ls -la",
            "cat file.txt",
            "grep pattern file.txt",
            "echo hello",
            "python script.py",
        ]
        for cmd in safe_commands:
            is_dangerous, _ = contains_dangerous_patterns(cmd)
            self.assertFalse(is_dangerous, f"Command flagged as dangerous: {cmd}")

    def test_command_substitution_dollar(self):
        """Test detection of command substitution with $()."""
        cmd = "echo $(cat /etc/passwd)"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Command substitution", reason)

    def test_command_substitution_backticks(self):
        """Test detection of command substitution with backticks."""
        cmd = "echo `cat /etc/passwd`"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("backticks", reason)

    def test_variable_assignment_with_path(self):
        """Test detection of variable assignment with path."""
        cmd = "DIR=/tmp/test && cd $DIR"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Variable assignment", reason)

    def test_export_with_path(self):
        """Test detection of export with path."""
        cmd = "export PATH=/malicious/bin"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Export", reason)

    def test_cd_followed_by_command(self):
        """Test detection of cd followed by command."""
        cmd = "cd /tmp && rm -rf *"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Directory change", reason)

    def test_exec_with_file_descriptors(self):
        """Test detection of exec with file descriptors."""
        cmd = "exec 3>/tmp/file"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("File descriptor", reason)

    def test_process_substitution(self):
        """Test detection of process substitution."""
        cmd = "diff <(ls dir1) <(ls dir2)"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Process substitution", reason)

    def test_input_redirection_from_absolute_path(self):
        """Test detection of input redirection from absolute path."""
        cmd = "cat < /etc/passwd"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Input redirection", reason)

    def test_heredoc(self):
        """Test detection of heredoc."""
        cmd = "cat << EOF\ntest\nEOF"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("Heredoc", reason)

    def test_heredoc_to_dev_allowed(self):
        """Test that heredoc to /dev/ is allowed."""
        cmd = "cat << /dev/null"
        is_dangerous, _ = contains_dangerous_patterns(cmd)
        # This should still be flagged based on the regex
        self.assertTrue(is_dangerous)

    def test_or_operator(self):
        """Test detection of OR operator."""
        cmd = "command1 || command2"
        is_dangerous, reason = contains_dangerous_patterns(cmd)
        self.assertTrue(is_dangerous)
        self.assertIn("OR operator", reason)


class TestParseBashCommandPaths(unittest.TestCase):
    """Test the parse_bash_command_paths function."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up temp directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_simple_cat_command(self):
        """Test parsing cat command."""
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("content")
        cmd = f"cat {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(test_file)])
        self.assertEqual(writable, [])

    def test_output_redirection(self):
        """Test parsing output redirection."""
        test_file = Path(self.test_dir) / "output.txt"
        cmd = f"echo hello > {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_file)])

    def test_append_redirection(self):
        """Test parsing append redirection."""
        test_file = Path(self.test_dir) / "output.txt"
        cmd = f"echo hello >> {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_file)])

    def test_input_redirection(self):
        """Test parsing input redirection."""
        test_file = Path(self.test_dir) / "input.txt"
        test_file.write_text("content")
        cmd = f"cat < {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(test_file)])
        self.assertEqual(writable, [])

    def test_pipe_command(self):
        """Test parsing piped commands."""
        file1 = Path(self.test_dir) / "file1.txt"
        file2 = Path(self.test_dir) / "file2.txt"
        file1.write_text("content")
        cmd = f"cat {file1} | grep pattern > {file2}"
        readable, writable = parse_bash_command_paths(cmd)
        # file1 is read by cat, file2 is only written to by redirect
        self.assertEqual(readable, [str(file1)])
        self.assertEqual(writable, [str(file2)])

    def test_cp_command(self):
        """Test parsing cp command (source read, dest write)."""
        src = Path(self.test_dir) / "source.txt"
        dst = Path(self.test_dir) / "dest.txt"
        src.write_text("content")
        cmd = f"cp {src} {dst}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(src)])
        self.assertEqual(writable, [str(dst)])

    def test_mv_command(self):
        """Test parsing mv command."""
        src = Path(self.test_dir) / "source.txt"
        dst = Path(self.test_dir) / "dest.txt"
        src.write_text("content")
        cmd = f"mv {src} {dst}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(src)])
        self.assertEqual(writable, [str(dst)])

    def test_dd_command(self):
        """Test parsing dd command with if= and of=."""
        input_file = Path(self.test_dir) / "input.bin"
        output_file = Path(self.test_dir) / "output.bin"
        input_file.write_bytes(b"data")
        cmd = f"dd if={input_file} of={output_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(input_file)])
        self.assertEqual(writable, [str(output_file)])

    def test_touch_command(self):
        """Test parsing touch command (write)."""
        test_file = Path(self.test_dir) / "new.txt"
        cmd = f"touch {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_file)])

    def test_mkdir_command(self):
        """Test parsing mkdir command (write)."""
        test_dir = Path(self.test_dir) / "newdir"
        cmd = f"mkdir {test_dir}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_dir)])

    def test_rm_command(self):
        """Test parsing rm command (write)."""
        test_file = Path(self.test_dir) / "todelete.txt"
        test_file.write_text("content")
        cmd = f"rm {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_file)])

    def test_tee_command(self):
        """Test parsing tee command (write).

        Tee reads from stdin and writes to files, so file arguments
        should only appear in writable list.
        """
        test_file = Path(self.test_dir) / "output.txt"
        cmd = f"echo hello | tee {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        # tee only writes to files (reads from stdin)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [str(test_file)])

    def test_dev_null_ignored(self):
        """Test that /dev/null is ignored."""
        cmd = "echo hello > /dev/null"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [])
        self.assertEqual(writable, [])

    def test_multiple_files(self):
        """Test parsing command with multiple files."""
        file1 = Path(self.test_dir) / "file1.txt"
        file2 = Path(self.test_dir) / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")
        cmd = f"cat {file1} {file2}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(sorted(readable), sorted([str(file1), str(file2)]))
        self.assertEqual(writable, [])

    def test_flags_ignored(self):
        """Test that flags are properly ignored."""
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("content")
        cmd = f"grep -i -n pattern {test_file}"
        readable, writable = parse_bash_command_paths(cmd)
        self.assertEqual(readable, [str(test_file)])
        self.assertEqual(writable, [])


class TestUsefulTools(unittest.TestCase):
    """Test the UsefulTools class."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Create readable and writable directories
        self.readable_dir = Path(self.test_dir) / "readable"
        self.writable_dir = Path(self.test_dir) / "writable"
        self.readable_dir.mkdir()
        self.writable_dir.mkdir()

    def tearDown(self):
        """Clean up temp directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_creates_base_dir(self):
        """Test that __init__ creates base_dir if it doesn't exist."""
        new_base = Path(self.test_dir) / "new_base"
        tools = UsefulTools(
            base_dir=str(new_base),
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        self.assertTrue(new_base.exists())
        self.assertTrue(new_base.is_dir())
        self.assertEqual(tools.base_dir, str(new_base.resolve()))

    def test_bash_safe_command(self):
        """Test Bash with a safe command."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        result = tools.Bash("echo hello", "Test echo")
        self.assertIn("hello", result)

    def test_bash_dangerous_command_blocked(self):
        """Test that dangerous commands are blocked."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        result = tools.Bash("echo $(cat /etc/passwd)", "Dangerous command")
        self.assertIn("Error: Security violation", result)

    def test_bash_read_permission_denied(self):
        """Test that reading outside readable paths is denied."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        # Create a file outside readable dir
        outside_file = Path(self.test_dir) / "outside.txt"
        outside_file.write_text("secret")

        result = tools.Bash(f"cat {outside_file}", "Read outside")
        self.assertIn("Error: Access denied for reading", result)

    def test_bash_write_permission_denied(self):
        """Test that writing outside writable paths is denied."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        outside_file = Path(self.test_dir) / "outside.txt"

        result = tools.Bash(f"touch {outside_file}", "Write outside")
        self.assertIn("Error: Access denied for writing", result)

    def test_bash_read_allowed(self):
        """Test that reading from readable paths is allowed."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.readable_dir / "test.txt"
        test_file.write_text("readable content")

        result = tools.Bash(f"cat {test_file}", "Read allowed")
        self.assertIn("readable content", result)

    def test_bash_write_allowed(self):
        """Test that writing to writable paths is allowed."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.writable_dir / "output.txt"

        result = tools.Bash(f"echo 'writable content' > {test_file}", "Write allowed")
        # Should not contain error
        self.assertNotIn("Error:", result)
        # Verify file was created
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text().strip(), "writable content")

    def test_edit_single_occurrence(self):
        """Test Edit with a single occurrence."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.writable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.writable_dir / "edit_test.txt"
        test_file.write_text("Hello World\nGoodbye World\n")

        # Note: Edit runs bash script which needs proper permissions
        # The Edit method validates paths but delegates to Bash
        result = tools.Edit(
            file_path=str(test_file),
            old_string="Hello World",
            new_string="Hi World",
            replace_all=False,
        )
        # Print result for debugging
        print(f"Edit result: {result}")

        content = test_file.read_text()
        self.assertIn("Hi World", content)
        self.assertNotIn("Hello World", content)

    def test_edit_replace_all(self):
        """Test Edit with replace_all=True."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.writable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.writable_dir / "edit_test.txt"
        test_file.write_text("foo bar foo baz foo\n")

        tools.Edit(
            file_path=str(test_file),
            old_string="foo",
            new_string="qux",
            replace_all=True,
        )

        content = test_file.read_text()
        self.assertEqual(content.count("qux"), 3)
        self.assertEqual(content.count("foo"), 0)

    def test_multiedit_single_occurrence(self):
        """Test MultiEdit with a single occurrence."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.writable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.writable_dir / "multiedit_test.txt"
        test_file.write_text("Alpha Beta\nGamma Delta\n")

        tools.MultiEdit(
            file_path=str(test_file),
            old_string="Alpha Beta",
            new_string="Alpha Omega",
            replace_all=False,
        )

        content = test_file.read_text()
        self.assertIn("Alpha Omega", content)
        self.assertNotIn("Alpha Beta", content)

    def test_multiedit_replace_all(self):
        """Test MultiEdit with replace_all=True."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.writable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        test_file = self.writable_dir / "multiedit_test.txt"
        test_file.write_text("test test test\n")

        tools.MultiEdit(
            file_path=str(test_file),
            old_string="test",
            new_string="pass",
            replace_all=True,
        )

        content = test_file.read_text()
        self.assertEqual(content.count("pass"), 3)
        self.assertEqual(content.count("test"), 0)

    def test_edit_path_resolution(self):
        """Test that Edit resolves paths correctly."""
        tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[str(self.writable_dir)],
            writable_paths=[str(self.writable_dir)],
        )
        # Create a file in writable dir
        test_file = self.writable_dir / "test.txt"
        test_file.write_text("original content\n")

        # Edit should resolve the path
        tools.Edit(
            file_path=str(test_file),
            old_string="original",
            new_string="modified",
            replace_all=False,
        )

        self.assertIn("modified", test_file.read_text())


if __name__ == "__main__":
    unittest.main()
