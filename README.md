# Loan Eligibility Notification Agent (ALENA)

ALENA is a production-grade automated middleware agent designed to manage, translate, and verify customer loan eligibility notifications. It processes underwriting data from retail banking datasets, applies business rules like deduplication and cooldown limits, generates human-friendly explanations for credit decisions, and crafts localized messages in regional Indian languages using Gemini, Groq, or local Ollama instances.

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com)
[![APScheduler](https://img.shields.io/badge/APScheduler-3.10.4-orange.svg)](https://apscheduler.github.io/apscheduler/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

1. **Multi-provider LLM Abstraction:** Seamlessly switch between Google Gemini, Groq, and offline Ollama via simple configuration files.
2. **Watermark-based Incremental Processing:** Keeps track of the last processed customer row to process new data increments efficiently.
3. **7-Day Cooldown Deduplication:** SQLite-based registry guarantees customers don't receive spam or redundant messages within a 7-day period.
4. **RBI-Compliant Explainability:** Automatically translates raw underwriting logic strings into user-friendly bullet points detailing credit score rankings and tier classifications.
5. **Multi-language Support:** Automatically detects customer language preferences and generates notifications in 8 major Indian languages.
6. **FastAPI Integration:** Provides operational endpoints to trigger runs, fetch notification statistics, verify connection health, and query running logs.
7. **Docker Ready:** Complete container configuration including health checks and isolated local directory mappings.
8. **Full Audit Trail:** Every transaction is logged sequentially to database registries, CSV log files, and rotating log outputs.

---

## Project Structure

```text
loan_agent/
├── main.py                  # Server entry point; initializes logs, starts FastAPI server, and launches scheduler threads
├── config.yaml              # Global application configuration file managing datasets, tiers, cooldowns, and LLM backends
├── requirements.txt         # Production and development dependencies with locked package versions
├── Dockerfile               # Builds the lightweight Python slim image container
├── docker-compose.yml       # Combines service volume mounts, ports, and environment variables
├── .env.example             # Template for API keys and endpoint environment configurations
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions workflow for linting, type checks, and automated unit testing
├── agent/
│   ├── __init__.py          # Core package exports mapping DataLoader, checkers, selectors, and orchestrators
│   ├── config.py            # Pydantic schemas validating configuration parameters and directory initialization
│   ├── loader.py            # Interfaces with Excel and updates database watermarks
│   ├── eligibility.py       # Filters customers into approved (P1, P2) or declined (P3) categories
│   ├── dedup.py             # SQLite transaction manager checking cooldown expiration windows
│   ├── explainer.py         # Parses semicolon-delimited credit variables into readable list strings
│   ├── multilang.py         # Resolves translation mapping falls-backs to English
│   ├── generator.py         # Formats customer prompts and coordinates text generation with the LLM
│   ├── scheduler.py         # Batch runner orchestrating the entire processing cycle loop
│   ├── api.py               # Exposes FastAPI routes for operational management
│   └── llm/
│       ├── __init__.py      # Re-exports provider structures
│       ├── base.py          # Abstract Base Class and exception definitions for LLM providers
│       ├── gemini.py        # Google Gemini integration via google-genai Client
│       ├── groq.py          # Groq Cloud integration via the official groq SDK
│       ├── ollama.py        # Offline local LLM chat completions integration using HTTP requests
│       └── factory.py       # Dispatches provider instances based on config settings
└── tests/
    ├── __init__.py          # Test package initialization
    ├── test_loader.py       # Validates data parsing, watermark filtering, and database updates
    ├── test_eligibility.py  # Checks tier evaluation, nan handling, and outlier indicators
    ├── test_dedup.py        # Asserts database registry logs, cooldown flags, and statistics queries
    └── test_explainer.py    # Tests RBI explainability formatting and reason-splitting logic
```

---

## Quick Start (Local)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd internship
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   copy .env.example .env
   ```

5. **Place the dataset in the root folder:**
   Ensure the input file `Bank_Loan_DS_Project.xlsx` is saved in the workspace root directory.

6. **Run the application:**
   ```bash
   python main.py
   ```

7. **Access the API Documentation:**
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to inspect or test the API endpoints.

---

## LLM Provider Setup

### Gemini (Free Tier)
- **Get API Key:** Visit [aistudio.google.com](https://aistudio.google.com/app/apikey).
- **Environment config:** Set `GEMINI_API_KEY=your_key_here` in `.env`.
- **YAML config:** Set `provider: "gemini"` and `gemini_model: "gemini-1.5-flash"` in `config.yaml`.

### Groq (Free Tier)
- **Get API Key:** Visit [console.groq.com](https://console.groq.com).
- **Environment config:** Set `GROQ_API_KEY=your_key_here` in `.env`.
- **YAML config:** Set `provider: "groq"` and `groq_model: "llama3-8b-8192"` in `config.yaml`.

### Ollama (Local/Offline)
- **Install:** Download and install Ollama from [https://ollama.com](https://ollama.com).
- **Download Model:** Download the target model locally:
  ```bash
  ollama pull llama3.2
  ```
- **Start Service:** Start the server daemon:
  ```bash
  ollama serve
  ```
- **YAML config:** Set `provider: "ollama"`, `ollama_base_url: "http://localhost:11434"`, and `ollama_model: "llama3.2"` in `config.yaml`. No API keys are required in `.env`.

---

## Configuration Reference

The application behavior is controlled by `config.yaml`. Here is the full file template:

```yaml
dataset:
  path: "Bank_Loan_DS_Project.xlsx"      # Location of the Excel dataset
  sheet_name: 0                          # Excel worksheet index or name

eligibility:
  approved_tiers: ["P1", "P2"]           # Customer tiers matching pre-approved offers
  declined_tier: "P3"                    # Customer tiers skipped during evaluation

cooldown:
  days: 7                                # Cooldown duration in days to lock repeat notifications

scheduler:
  interval_hours: 1                      # Time interval in hours for background tasks
  timezone: "UTC"                        # Chrono zone standard context

llm:
  provider: "gemini"                     # Provider selection: gemini, groq, or ollama
  gemini_model: "gemini-1.5-flash"       # Gemini model choice
  groq_model: "llama3-8b-8192"           # Groq model choice
  ollama_base_url: "http://localhost:11434" # Host address for Ollama local service
  ollama_model: "llama3.2"               # Local Ollama model choice
  max_tokens: 1000                       # Limit response output lengths
  api_key_env: "GEMINI_API_KEY"          # Configured key variable to read from environment

notification:
  id_format: "NOTIF-{date}-{prospect_id:05d}" # ID generation template format
  offer_validity_days: 30                # Days pre-approved deals remain valid
  default_language: "English"            # Fallback language selection

output:
  log_csv: "output/notifications_log.csv" # Audit CSV target path
  sqlite_db: "db/notifications.db"       # Persistent SQLite ledger target path
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | Returns system metadata details (application name and version). |
| **GET** | `/health` | Returns active connection and health checks for configured LLMs. |
| **POST** | `/pipeline/run` | Triggers the pipeline batch processor run immediately in a worker thread. |
| **GET** | `/stats` | Fetches cumulative totals (sent, skipped, and total registry records). |
| **GET** | `/logs` | Returns the trailing log statements written to `logs/agent.log`. |
| **GET** | `/docs` | Interactive Swagger API documentation UI interface. |

---

## Docker Deployment

Deploy using Docker Compose to handle folder creations and database path persistence automatically:

1. **Build and run containers in detached mode:**
   ```bash
   docker-compose up -d --build
   ```

2. **Inspect pipeline execution logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop application containers:**
   ```bash
   docker-compose down
   ```

> [!NOTE]
> The database registry (`db/`), CSV exports (`output/`), and log files (`logs/`) are mounted locally as volumes in the workspace root, preserving pipeline history and files even if containers are destroyed.

---

## Running Tests

Perform automated checks locally using the following suites:

- **Run all unit tests:**
  ```bash
  pytest tests/ -v
  ```
- **Ruff check linting rules:**
  ```bash
  ruff check agent/ --ignore E501
  ```
- **Format styling checking:**
  ```bash
  black --check agent/
  ```
- **Type verification checking:**
  ```bash
  mypy agent/ --ignore-missing-imports
  ```

### Test Scope Coverage:
- `test_loader.py`: Verifies spreadsheet load reliability, validation errors on empty schema fields, and SQLite watermark records.
- `test_eligibility.py`: Asserts correct tier sorting (P1/P2 approved, P3 skipped), float parsing rules, and trade line outliers.
- `test_dedup.py`: Asserts cooldown rules, date comparisons, and statistics aggregations.
- `test_explainer.py`: Checks credit score additions, tier tag formatting, and semicolon splits.

---

## Changing Cooldown Period

To modify the communications deduplication lock duration, modify the `days` field inside the `cooldown` section in `config.yaml`:

```yaml
# Before (7-day wait window):
cooldown:
  days: 7

# After (14-day wait window):
cooldown:
  days: 14
```

Active records stored in the SQLite ledger (`db/notifications.db`) will instantly inherit the updated value. Future iterations will check transactions relative to `sent_at` timestamp metrics against the updated day delta count.

---

## Adding a New Language

Adding regional languages to ALENA is straightforward:

1. **Register the Language:** Append the target language name into the `self.supported_languages` list in `agent/multilang.py`:
   ```python
   self.supported_languages = [
       "English", "Hindi", "Tamil", "Bengali",
       "Telugu", "Marathi", "Gujarati", "Kannada", "Punjabi" # Added
   ]
   ```
2. **Add Preferences to Dataset:** Inside the customer spreadsheet database, assign `"Punjabi"` (or the normalized target name) under the `preferred_language` column for target prospects.
3. **Automatic Generation:** During pipeline execution, ALENA detects the preference and updates the LLM prompt. The generator dynamically requests responses written in that language.

---

## Notification ID Format

ALENA creates tracing notification keys following this structure:
`NOTIF-{YYYYMMDD}-{PROSPECTID zero-padded to 5 digits}`

### Examples:
- `NOTIF-20260727-00001` (Prospect ID 1 processed on July 27, 2026)
- `NOTIF-20260727-38343` (Prospect ID 38343 processed on July 27, 2026)

This format allows customer support or auditors to map any message back to the specific row in the credit underwriting database.

---

## License

This project is licensed under the terms of the MIT License.
