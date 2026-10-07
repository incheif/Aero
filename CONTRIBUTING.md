# Contributing to Aero VLA Robotics Studio

Thank you for your interest in contributing to the **Aero VLA Robotics Studio**! We welcome contributions ranging from bug fixes and algorithm optimizations to documentation improvements and new robotic manipulation skills.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to [epost.dhruv@gmail.com](mailto:epost.dhruv@gmail.com).

---

## 🔒 Security & Secret Hygiene (CRITICAL)

To protect your credentials and prevent accidental exposure:
1. **NEVER commit API keys or credentials**: Do NOT hardcode Google Gemini API keys, tokens, passwords, or secret URIs into any code or test file.
2. **Use `.env` locally**: Store secret keys only in a local `.env` file (copied from `.env.example`).
3. **Verify `.gitignore`**: Git is configured to ignore `.env`, `*.key`, `*.pem`, and `secrets.json`. Before staging changes, verify with `git status` that no private files are being tracked.
4. **Pre-commit scan**: Check your diff using `git diff --staged` to ensure no accidental secrets or tokens are present.

If you suspect you have accidentally exposed an API key, revoke it immediately at [Google AI Studio](https://aistudio.google.com/) and follow the disclosure process in [SECURITY.md](SECURITY.md).

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- (Optional) ROS 2 (Humble / Iron / Rolling)
- A modern web browser with WebGL 2.0 support (Chrome, Edge, Firefox)

### 2. Clone the Repository
```bash
git clone https://github.com/incheif/Aero.git
cd Aero
```

### 3. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```
*(Optionally add your Google AI Studio API key in `.env` for cloud Gemini inference. The embedded VLA policy works offline without an API key).*

---

## Development Workflow

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Run the Studio Locally**:
   ```bash
   python run_dashboard.py
   ```
   Navigate to `http://127.0.0.1:8000` to verify your changes.

3. **Coding Standards**:
   - Follow **PEP 8** style guidelines for Python code.
   - Use clear type annotations (`typing.Optional`, `typing.List`, etc.).
   - Write descriptive docstrings and meaningful commit messages.
   - For web frontend code, ensure clean, modern Vanilla CSS/JS without unvetted external dependencies.

4. **Testing ROS 2 Nodes (if modifying `ros2_ws`)**:
   ```bash
   cd ros2_ws
   colcon build --packages-select vla_robot_arm
   source install/setup.bash
   ros2 launch vla_robot_arm vla_robot_arm.launch.py
   ```

---

## Submitting Pull Requests

1. **Commit Convention**:
   Use structured commit messages:
   - `feat: <description>` for new capabilities
   - `fix: <description>` for bug fixes
   - `docs: <description>` for documentation changes
   - `refactor: <description>` for non-breaking code restructuring
   - `test: <description>` for tests

2. **Push to Your Fork / Branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request**:
   - Provide a clear summary of what changes were made and why.
   - Link any related issues or discussions.
   - Confirm that all local tests and simulation runs pass.
