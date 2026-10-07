# Security Policy

## Supported Versions

We provide security fixes and updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

---

## 🔑 API Key & Secret Management Policy

The **Aero VLA Robotics Studio** interfaces with large multimodal models (such as Google Gemini via `google-genai`). Protecting API keys and sensitive credentials is of the highest priority.

### How We Handle API Keys
1. **Client-Side Isolation**: When entered through the Web Dashboard UI, your `GEMINI_API_KEY` is retained solely in your local browser's `localStorage`.
2. **Environment Variable Fallback**: For headless or script execution, the runtime reads `GEMINI_API_KEY` from the system environment or a local `.env` file.
3. **Zero Telemetry Leakage**: API keys are never included in WebSocket telemetry broadcasts, server log files, error traces, or simulation recordings.
4. **Git Protection**: `.gitignore` is strictly configured to ignore:
   - `.env`, `.env.*`, `*.env`
   - `secrets.json`, `credentials.json`
   - `api_key.txt`, `api_keys/`, `*.key`, `*.pem`, `token.json`
   - Temporary scratch scripts and logs (`scratch/`, `*.log`, `logs/`)

### Contributor Checklist
- [ ] Never hardcode API keys, tokens, or credentials in source code or sample scripts.
- [ ] Always test with `.env.example` placeholders (e.g. `your_gemini_api_key_here`).
- [ ] Always check `git status` and `git diff --staged` before committing to verify no secret files are tracked.

### What To Do If An API Key Is Exposed
If you accidentally commit an API key:
1. **Immediately revoke the key** in the [Google AI Studio Console](https://aistudio.google.com/app/apikey).
2. Generate a new API key for your local development.
3. If the commit was pushed to a public branch, rewrite the Git history using `git filter-repo` or BFG Repo-Cleaner, and notify the repository maintainer.

---

## Reporting a Vulnerability

If you discover a security vulnerability or credential leak within this repository:

1. **Do NOT report security vulnerabilities via public GitHub issues.**
2. Send an email directly to **[epost.dhruv@gmail.com](mailto:epost.dhruv@gmail.com)** with the subject line `[SECURITY] Vulnerability Report - Aero`.
3. Please include:
   - A detailed description of the vulnerability.
   - Steps to reproduce or proof-of-concept code.
   - Any proposed remediation or mitigation steps.

We will acknowledge receipt within 48 hours and work with you on a responsible disclosure timeline before publishing a patch.
