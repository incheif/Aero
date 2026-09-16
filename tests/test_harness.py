"""
test_harness.py

Automated unit tests for AERO Execution Harness:
- AST code patcher and validator
- Static ROS graph introspector
- Colcon build error isolation
- Safe process cleaner invocation
"""

import os
import tempfile
import unittest

from agent_harness.patcher import validate_python_syntax, patch_controller_code, revert_controller_code
from agent_harness.introspector import inspect_controller_interfaces
from agent_harness.builder import extract_key_errors, BuildResult
from agent_harness.process_cleaner import cleanup_ros_and_gazebo
from agent_harness.runner import EvaluationResult


class TestPatcher(unittest.TestCase):

    def test_valid_syntax(self):
        valid_code = "def foo():\n    return 42\n"
        ok, err = validate_python_syntax(valid_code)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_invalid_syntax_rejected(self):
        broken_code = "def foo(:\n    return 42\n"
        ok, err = validate_python_syntax(broken_code)
        self.assertFalse(ok)
        self.assertIn("SyntaxError", err)

    def test_patch_and_revert(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Original code\n")
            target_path = f.name

        try:
            new_code = "print('Updated code')\n"
            ok, msg = patch_controller_code(new_code, target_filepath=target_path, backup=True)
            self.assertTrue(ok)

            with open(target_path, 'r') as f:
                self.assertEqual(f.read(), new_code)

            # Test revert
            rev_ok, rev_msg = revert_controller_code(target_filepath=target_path)
            self.assertTrue(rev_ok)
            with open(target_path, 'r') as f:
                self.assertEqual(f.read(), "# Original code\n")
        finally:
            if os.path.exists(target_path):
                os.remove(target_path)
            if os.path.exists(f"{target_path}.bak"):
                os.remove(f"{target_path}.bak")


class TestIntrospector(unittest.TestCase):

    def test_inspect_actual_controller(self):
        # Should find /cmd_vel publisher and /odom subscriber in the created controller
        res = inspect_controller_interfaces()
        self.assertTrue(res.valid)
        self.assertIn('/cmd_vel', res.publishers)
        self.assertIn('/odom', res.subscribers)
        self.assertIn('/scan', res.subscribers)

    def test_missing_interface_detection(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class EmptyNode:\n    pass\n")
            target_path = f.name

        try:
            res = inspect_controller_interfaces(target_path)
            self.assertFalse(res.valid)
            self.assertTrue(len(res.missing_interfaces) >= 2)
        finally:
            if os.path.exists(target_path):
                os.remove(target_path)


class TestBuilderErrorExtraction(unittest.TestCase):

    def test_extract_traceback(self):
        raw_log = (
            "Starting >>> robot_controller\n"
            "--- stderr: robot_controller\n"
            "Traceback (most recent call last):\n"
            "  File 'setup.py', line 10, in <module>\n"
            "SyntaxError: invalid syntax\n"
            "--- failed <<<\n"
        )
        extracted = extract_key_errors(raw_log)
        self.assertIn("SyntaxError: invalid syntax", extracted)


class TestProcessCleaner(unittest.TestCase):

    def test_cleaner_runs_safely(self):
        # Must execute cleanly without raising unhandled exceptions
        count = cleanup_ros_and_gazebo(timeout_sec=0.5)
        self.assertIsInstance(count, int)


class TestEvaluationResult(unittest.TestCase):

    def test_evaluation_result_parsing(self):
        res = EvaluationResult(
            status="PASSED",
            reason="SUCCESS",
            final_distance=0.08,
            time_elapsed_sec=14.5,
            min_obstacle_distance=0.45,
            error_log_snippet=""
        )
        self.assertTrue(res.is_passed())
        self.assertIn("PASSED", res.get_summary())


if __name__ == '__main__':
    unittest.main()
