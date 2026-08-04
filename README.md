# FinRisk-ALENA: Unified Credit Underwriting & Smart Notification Agent

FinRisk-ALENA is a unified credit decisioning and customer engagement system. It combines an ML-powered underwriting pipeline with an LLM-driven multilingual notification agent.

---

## System Architecture & Data Flow

```
[Raw Customer Data] (featured_dataset.csv or Excel)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  FinRisk ML Pipeline                                   │
│  - Gradient Boosting (Risk Tier: P1–P4)               │
│  - Random Forest Regressor (Loan Sizing)               │
│  - Business Policy Caps & FOIR Calculations            │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼ [Standardized Eligibility JSON]
                    │ (prospect_id, tier, loan_amount, interest_rate, emi, repayment_summary)
                    │
┌───────────────────▼────────────────────────────────────┐
│  ALENA Notification Engine                             │
│  - Watermarking (Prevents reprocessing / duplicates)    │
│  - 7-Day Cooldown Validation (SQLite)                  │
│  - Multilingual Translation & LLM Explainer Reasoning   │
│  - Twilio WhatsApp Dispatch (Multilingual Message)     │
└───────────────────┬────────────────────────────────────┘
                    │
                    ├──────────────────────┐
                    │                      │
                    ▼                      ▼
           [SQLite Registry]        [Excel Sync]
           (notifications.db)   (Bank_Loan_DS_Project.xlsx)
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  Flask Management Dashboard                            │
│  - Live Portfolios & ML Metrics                       │
│  - Single Customer Detail & Payoff Schedule            │
│  - Live Predict & Batch Upload Scoring                 │
│  - Notifications Status & Audit Logs Tab               │
└────────────────────────────────────────────────────────┘
```

---

## Configuration Reference (`config.yaml`)

Manage system behavior by editing the `config.yaml` file:

```yaml
dataset:
  source: "finrisk_pipeline"  # "finrisk_pipeline" (run ML models) or "excel" (read precomputed data)
  excel_path: "Bank_Loan_DS_Project.xlsx"
  excel_sheet_name: 0
  csv_path: "data/processed/featured_dataset.csv"

eligibility:
  approved_tiers: ["P1", "P2"] # Tiers eligible for WhatsApp notification dispatch
  declined_tier: "P3"          # Declined risk tier

cooldown:
  days: 7                     # Notification suppression window length

scheduler:
  interval_hours: 1           # Interval between pipeline sweeps
  max_sends_per_run: 6        # Batch throttling limit

llm:
  provider: "gemini"          # "gemini" | "groq" | "ollama"
  gemini_model: "gemini-1.5-flash"
  groq_model: "llama-3.1-8b-instant"
  ollama_base_url: "http://localhost:11434"
  ollama_model: "llama3.2"
  max_tokens: 1000

notification:
  id_format: "NOTIF-{date}-{prospect_id:05d}"
  offer_validity_days: 30     # Days the pre-approved offer remains open
  default_language: "English"  # Fallback target language
```

---

## Setup & Running Instructions

### Prerequisites
Create a `.env` file in the project root containing your API credentials:
```env
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=whatsapp:+14155238886

# Provide the key corresponding to your configured LLM provider:
GEMINI_API_KEY=your_gemini_key
# GROQ_API_KEY=your_groq_key
```

### Option 1: Docker Compose (Recommended)
Build and start both the FastAPI backend and the Flask dashboard in single-step orchestration:
```bash
docker-compose up --build
```
- **Flask Management Dashboard**: `http://localhost:5000`
- **FastAPI API & Docs**: `http://localhost:8000/docs`

### Option 2: Local Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the background worker & FastAPI server:
   ```bash
   python main.py
   ```
3. In a separate terminal, run the Flask dashboard app:
   ```bash
   python dashboard/app.py
   ```

---

## API Endpoints

### FastAPI Backend (Port 8000)
- `GET /health`: Liveness and LLM connectivity status report.
- `POST /pipeline/run`: Synchronously trigger a notification run sweep.
- `GET /stats`: Retrieve SQLite execution statistics (total sent, skipped, cooldowns).
- `GET /logs`: Fetch trailing agent log statements.
- `GET /docs`: Re-route to interactive Swagger UI documentation.

### Flask Management Dashboard (Port 5000)
- `GET /`: Overview panel showing portfolio KPIs and EDA charts.
- `GET /predict`: Single applicant live-prediction scoring form.
- `GET /batch`: Drag-and-drop CSV batch upload pipeline interface.
- `GET /customers`: Searchable database of all approved customers.
- `GET /notifications`: Clean visual audit tracking of sent/skipped notifications.
