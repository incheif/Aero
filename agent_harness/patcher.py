"""
patcher.py

Safe code modification and AST syntax validation engine for AERO.
Allows agents to patch or replace the target controller code with pre-validation
and automatic backup creation.
"""

import ast
import os
import shutil
from typing import Optional, Tuple


def validate_python_syntax(code_content: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the provided code is valid Python syntax.
    
    Returns:
        (is_valid, error_message_if_any)
    """
    try:
        ast.parse(code_content)
        return True, None
    except SyntaxError as e:
        error_msg = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  Line: {e.text}"
        return False, error_msg


def patch_controller_code(
    new_code: str,
    target_filepath: Optional[str] = None,
    backup: bool = True
) -> Tuple[bool, str]:
    """
    Validates and writes new code to the robot controller node.

    Args:
        new_code: Full python source code for the new controller.
        target_filepath: Path to controller_node.py. Auto-detected if None.
        backup: If True, saves target_filepath.bak before writing.

    Returns:
        (success, message)
    """
    if target_filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_filepath = os.path.join(
            os.path.dirname(current_dir),
            'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py'
        )

    # 1. Validate AST syntax
    is_valid, syntax_error = validate_python_syntax(new_code)
    if not is_valid:
        return False, f"Code rejected by pre-validator: {syntax_error}"

    # 2. Ensure target directory exists
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)

    # 3. Create backup if target file exists
    if backup and os.path.exists(target_filepath):
        backup_path = f"{target_filepath}.bak"
        shutil.copyfile(target_filepath, backup_path)

    # 4. Write new code atomically
    temp_target = f"{target_filepath}.tmp"
    with open(temp_target, 'w', encoding='utf-8') as f:
        f.write(new_code)
    os.replace(temp_target, target_filepath)

    return True, f"Successfully patched {os.path.basename(target_filepath)}"


def revert_controller_code(target_filepath: Optional[str] = None) -> Tuple[bool, str]:
    """Revert target controller to the last .bak file if available."""
    if target_filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_filepath = os.path.join(
            os.path.dirname(current_dir),
            'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py'
        )

    backup_path = f"{target_filepath}.bak"
    if not os.path.exists(backup_path):
        return False, "No backup file found to revert."

    shutil.copyfile(backup_path, target_filepath)
    return True, f"Reverted {os.path.basename(target_filepath)} from backup."
