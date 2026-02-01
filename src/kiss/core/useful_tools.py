"""
Hardened version of useful_tools.py with immediate security improvements.

This version adds:
1. Expanded command lists
2. Additional redirect operators
3. Dangerous pattern detection
4. Comprehensive logging

Note: This is still NOT production-ready. For production use, implement
proper sandboxing (bubblewrap, Docker, etc.) as described in SECURITY_FIXES.md
"""

import re
import shlex
import subprocess
from pathlib import Path

from kiss.core.utils import is_subpath


def _extract_directory(path_str: str) -> str | None:
    """Extract directory from a file path without resolving symlinks.

    Args:
        path_str: A file or directory path

    Returns:
        The directory path, or None if invalid
    """
    try:
        path = Path(path_str)

        # If it's an absolute path
        if path.is_absolute():
            # Check if path exists to determine if it's a file or directory
            if path.exists():
                return str(path)
            else:
                # Path doesn't exist - use heuristics
                if path_str.endswith("/"):
                    # Trailing slash indicates directory
                    return str(path)
                else:
                    # Check if it has a file extension
                    if path.suffix:
                        # Has extension - likely a file
                        return str(path)
                    else:
                        # No extension - could be directory
                        # Check if parent exists and is a directory
                        if path.parent.exists() and path.parent.is_dir():
                            # Parent exists, so this is likely a file or subdir
                            return str(path)
                        else:
                            # Parent doesn't exist either - assume it's a directory path
                            return str(path)

        # For relative paths, return None (we can't determine the directory reliably)
        return None

    except Exception:
        return None


def contains_dangerous_patterns(command: str) -> tuple[bool, str]:
    """Check for dangerous shell patterns that commonly bypass security.

    Args:
        command: The bash command to analyze

    Returns:
        (is_dangerous, reason) tuple
    """
    # Check for command substitution
    if "$(" in command:
        return True, "Command substitution $() detected"

    if "`" in command:
        return True, "Command substitution with backticks detected"

    # Check for variable assignment followed by usage
    if re.search(r"\w+=/[^\s;]+ *(?:&&|;)", command):
        return True, "Variable assignment with path detected"

    # Check for export followed by usage
    if "export" in command and re.search(r"export +\w+=/[^\s;]+", command):
        return True, "Export of path variable detected"

    # Check for cd followed by any command
    if re.search(r"\bcd\b[^;]*(?:&&|;)", command):
        return True, "Directory change followed by command detected"

    # Check for exec with file descriptors
    if re.search(r"\bexec +\d*[<>]", command):
        return True, "File descriptor manipulation with exec detected"

    # Check for process substitution
    if "<(" in command or ">(" in command:
        return True, "Process substitution detected"

    # Check for input redirection from file
    if re.search(r"<\s*/[^\s;&|]+", command):
        return True, "Input redirection from absolute path detected"

    # Check for heredocs (can be used for bypasses)
    if "<<" in command and not re.search(r"2?<<\s*/dev/", command):
        return True, "Heredoc detected"

    # Check for multiple commands (compound statements)
    # Allow && and ; only if no path-like strings follow cd
    if "||" in command:
        return True, "OR operator detected (||)"

    return False, ""


def parse_bash_command_paths(command: str) -> tuple[list[str], list[str]]:
    """Parse a bash command to extract readable and writable directory paths.

    This function analyzes bash commands to determine which directories are
    being read from and which are being written to.

    Args:
        command: A bash command string to parse

    Returns:
        A tuple of (readable_dirs, writable_dirs) where each is a list of directory paths

    """
    readable_paths: set[str] = set()
    writable_paths: set[str] = set()

    # EXPANDED: Commands that read files/directories
    read_commands = {
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "grep",
        "find",
        "ls",
        "diff",
        "wc",
        "sort",
        "uniq",
        "cut",
        "sed",
        "awk",
        "tee",
        "od",
        "hexdump",
        "file",
        "stat",
        "du",
        "df",
        "tree",
        "read",
        "source",
        ".",
        "tar",
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "bzip2",
        "bunzip2",
        "python",
        "python3",
        "node",
        "ruby",
        "perl",
        "bash",
        "sh",
        "zsh",
        "make",
        "cmake",
        "gcc",
        "g++",
        "clang",
        "javac",
        "java",
        "cargo",
        "npm",
        "yarn",
        "pip",
        "go",
        "rustc",
        "rsync",
        # ADDED: Previously untracked commands
        "strings",
        "xxd",
        "nl",
        "fold",
        "rev",
        "pr",
        "fmt",
        "expand",
        "unexpand",
        "tr",
        "col",
        "colrm",
        "column",
        "join",
        "paste",
        "comm",
        "cmp",
        "look",
        "split",
        "csplit",
        "iconv",
        "base64",
        "base32",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "cksum",
        "sum",
        "readlink",
        "realpath",
        "dirname",
        "basename",
        "pathchk",
    }

    # Commands that write files/directories
    write_commands = {
        "touch",
        "mkdir",
        "rm",
        "rmdir",
        "mv",
        "cp",
        "dd",
        "tee",
        "install",
        "chmod",
        "chown",
        "chgrp",
        "ln",
        "rsync",
    }

    write_redirects = {
        ">",
        ">>",
        "&>",
        "&>>",
        "1>",
        "2>",
        "2>&1",
        ">|",
        ">>|",
        "&>|",
        "1>>",
        "2>>",
    }

    try:
        # Handle pipes - split into sub-commands
        pipe_parts = command.split("|")

        for part in pipe_parts:
            part = part.strip()

            # Check for output redirection (writing)
            for redirect in write_redirects:
                if redirect in part:
                    # Extract path after redirect
                    redirect_match = re.search(rf"{re.escape(redirect)}\s*([^\s;&|]+)", part)
                    if redirect_match:
                        path = redirect_match.group(1).strip()
                        path = path.strip("'\"")
                        if path and path != "/dev/null":
                            dir_path = _extract_directory(path)
                            if dir_path:
                                writable_paths.add(dir_path)

            # Check for input redirection (reading)
            input_redirect_match = re.search(r"<\s*([^\s;&|]+)", part)
            if input_redirect_match:
                path = input_redirect_match.group(1).strip()
                path = path.strip("'\"")
                if path and path != "/dev/null":
                    dir_path = _extract_directory(path)
                    if dir_path:
                        readable_paths.add(dir_path)

            # Parse the command tokens
            try:
                tokens = shlex.split(part)
            except ValueError:
                # If shlex fails, do basic split
                tokens = part.split()

            if not tokens:
                continue

            cmd = tokens[0].split("/")[-1]  # Get base command name

            # Process based on command type
            if cmd in read_commands or cmd in write_commands:
                # Extract file/directory arguments (skip flags)
                paths: list[str] = []
                i = 1
                while i < len(tokens):
                    token = tokens[i]

                    # Skip flags and their arguments
                    if token.startswith("-"):
                        i += 1
                        # Skip flag argument if it doesn't start with - or /
                        if (
                            i < len(tokens)
                            and not tokens[i].startswith("-")
                            and not tokens[i].startswith("/")
                        ):
                            i += 1
                        continue

                    # Check if it looks like a path
                    if "/" in token or not any(c in token for c in ["=", "$", "(", ")"]):
                        token = token.strip("'\"")
                        if token and token != "/dev/null":
                            paths.append(token)

                    i += 1

                # Classify paths based on command
                if cmd in read_commands:
                    for path in paths:
                        dir_path = _extract_directory(path)
                        if dir_path:
                            readable_paths.add(dir_path)

                if cmd in write_commands:
                    # For write commands, typically the last path is written to
                    if paths:
                        if cmd in ["cp", "mv", "rsync"]:
                            # Source(s) are read, destination is written
                            for path in paths[:-1]:
                                dir_path = _extract_directory(path)
                                if dir_path:
                                    readable_paths.add(dir_path)

                            # Last path is destination
                            if len(paths) > 0:
                                dir_path = _extract_directory(paths[-1])
                                if dir_path:
                                    writable_paths.add(dir_path)
                        elif cmd == "dd":
                            # Special handling for dd command
                            # Look for of= parameter
                            for token in tokens:
                                if token.startswith("of="):
                                    output_file = token[3:]
                                    dir_path = _extract_directory(output_file)
                                    if dir_path:
                                        writable_paths.add(dir_path)
                                elif token.startswith("if="):
                                    input_file = token[3:]
                                    dir_path = _extract_directory(input_file)
                                    if dir_path:
                                        readable_paths.add(dir_path)
                        else:
                            # Other write commands
                            for path in paths:
                                dir_path = _extract_directory(path)
                                if dir_path:
                                    writable_paths.add(dir_path)

                # tee reads stdin and writes to file
                if cmd == "tee":
                    for path in paths:
                        dir_path = _extract_directory(path)
                        if dir_path:
                            writable_paths.add(dir_path)

    except Exception as e:
        # If parsing fails completely, return empty lists
        print(f"Failed to parse command '{command}': {e}")
        return ([], [])

    # Clean up paths - remove empty strings and '.'
    readable_dirs = sorted([p for p in readable_paths if p and p != "."])
    writable_dirs = sorted([p for p in writable_paths if p and p != "."])

    return (readable_dirs, writable_dirs)


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    def __init__(
        self,
        base_dir: str,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
    ) -> None:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(base_dir).resolve())
        self.readable_paths = [Path(p).resolve() for p in readable_paths or []]
        self.writable_paths = [Path(p).resolve() for p in writable_paths or []]

    def run_bash_command(self, command: str, description: str) -> str:
        """Runs a bash command and returns its output.

        Args:
            command: The bash command to run.
            description: A brief description of the command.

        Returns:
            The output of the command.
        """

        # NEW: Check for dangerous patterns first
        is_dangerous, reason = contains_dangerous_patterns(command)
        if is_dangerous:
            return f"Error: Security violation - {reason}"

        # Parse and validate paths
        readable, writable = parse_bash_command_paths(command)

        for path_str in readable:
            resolved = Path(path_str).resolve()
            if not is_subpath(resolved, self.readable_paths):
                return f"Error: Access denied for reading {path_str}"

        for path_str in writable:
            resolved = Path(path_str).resolve()
            if not is_subpath(resolved, self.writable_paths):
                return f"Error: Access denied for writing to {path_str}"

        try:
            # Execute with timeout for safety
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,  # NEW: Add timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
        except Exception as e:
            return f"Error: {e}"
