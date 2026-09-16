"""
introspector.py

Static ROS 2 graph and interface introspection tool for AERO.
Analyzes the target controller Python AST before simulation launch to verify that
required interface contracts (/cmd_vel publisher, /odom subscriber) are present.
"""

import ast
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GraphInspectionResult:
    """Outcome of static ROS graph introspection on controller code."""
    valid: bool
    publishers: List[str] = field(default_factory=list)
    subscribers: List[str] = field(default_factory=list)
    missing_interfaces: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        if self.valid:
            return (
                f"Graph Introspection PASSED. "
                f"Publishers: {self.publishers}, Subscribers: {self.subscribers}"
            )
        return (
            f"Graph Introspection FAILED. Missing required interfaces: {self.missing_interfaces}\n"
            f"Warnings: {self.warnings}"
        )


class ROSInterfaceVisitor(ast.NodeVisitor):
    """AST visitor searching for create_publisher and create_subscription calls."""

    def __init__(self) -> None:
        self.publishers: List[str] = []
        self.subscribers: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Check method name: create_publisher / create_subscription
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if func_name == 'create_publisher' and len(node.args) >= 2:
                topic_arg = node.args[1]
                if isinstance(topic_arg, ast.Constant) and isinstance(topic_arg.value, str):
                    self.publishers.append(topic_arg.value)
            elif func_name == 'create_subscription' and len(node.args) >= 2:
                topic_arg = node.args[1]
                if isinstance(topic_arg, ast.Constant) and isinstance(topic_arg.value, str):
                    self.subscribers.append(topic_arg.value)

        self.generic_visit(node)


def inspect_controller_interfaces(filepath: Optional[str] = None) -> GraphInspectionResult:
    """
    Statically analyzes controller_node.py to ensure expected topic interfaces exist.
    Required:
      - Publisher on '/cmd_vel'
      - Subscription to '/odom'
    Recommended:
      - Subscription to '/scan'
    """
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(
            os.path.dirname(current_dir),
            'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py'
        )

    if not os.path.exists(filepath):
        return GraphInspectionResult(
            valid=False,
            missing_interfaces=['controller_node.py not found'],
            warnings=[f"Target file missing: {filepath}"]
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
    except SyntaxError as e:
        return GraphInspectionResult(
            valid=False,
            missing_interfaces=[f"SyntaxError in controller code: {e.msg}"],
            warnings=[]
        )

    visitor = ROSInterfaceVisitor()
    visitor.visit(tree)

    missing = []
    warnings = []

    # Mandatory interfaces
    if '/cmd_vel' not in visitor.publishers:
        missing.append("create_publisher(Twist, '/cmd_vel', ...)")
    if '/odom' not in visitor.subscribers:
        missing.append("create_subscription(Odometry, '/odom', ...)")

    # Recommended interfaces
    if '/scan' not in visitor.subscribers:
        warnings.append("No '/scan' subscription found. Robot may not avoid obstacles.")

    is_valid = (len(missing) == 0)
    return GraphInspectionResult(
        valid=is_valid,
        publishers=visitor.publishers,
        subscribers=visitor.subscribers,
        missing_interfaces=missing,
        warnings=warnings
    )


if __name__ == '__main__':
    res = inspect_controller_interfaces()
    print(res.get_summary())
