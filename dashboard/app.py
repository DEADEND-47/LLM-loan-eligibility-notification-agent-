"""
Flask dashboard for the AI-Powered Credit Risk & Smart Loan Recommendation System.

Run from the project root:
    python dashboard/app.py
then open http://localhost:5000
"""
import os
import tempfile
import pandas as pd
import sqlite3
import yaml
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    abort,
    Response,
)

try:
    from . import pipeline as pl
except ImportError:
    import pipeline as pl

app = Flask(__name__)

# Batch outputs go to the OS temp dir -- it is writable everywhere (the project
# folder is read-only on serverless hosts like Vercel, which would 500 the upload).
_TMP = tempfile.gettempdir()
BATCH_RESULT_PATH = os.path.join(_TMP, "finrisk_last_batch_results.csv")
BATCH_INPUTS_PATH = os.path.join(_TMP, "finrisk_last_batch_inputs.csv")


@app.context_processor
def inject_asset_version():
    """Cache-busting token so browsers reload static assets whenever they change."""
    v = 0
    for fname in ("style.css", "payoff.js"):
        try:
            v = max(v, int(os.path.getmtime(os.path.join(app.static_folder, fname))))
        except OSError:
            pass
    return {"asset_version": v}


@app.route("/favicon.ico")
@app.route("/favicon.png")
def favicon():
    return Response(status=204)


# Home / Overview page -- shows the portfolio KPIs, tier split and EDA charts.
@app.route("/")
def index():
    return render_template("index.html", stats=pl.dashboard_stats(),
                           tier_rules=pl.TIER_RULES.reset_index().to_dict("records"))


# Customers page -- searchable, sortable table of all approved applicants.
@app.route("/customers")
def customers():
    query = request.args.get("q", "").strip()      # optional Customer-ID search
    sort = request.args.get("sort", "id")          # how to order the table
    rows = pl.search_customers(query=query, sort=sort, limit=100)
    return render_template("customers.html", rows=rows, query=query, sort=sort)


# Notifications page -- displays notification logs from SQLite
def get_db_path():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg["output"]["sqlite_db"]
    except Exception:
        return "db/notifications.db"


@app.route("/notifications")
def notifications():
    db_path = get_db_path()
    rows = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prospect_id, notification_id, tier, sent_at, language, status, channel, processing_status, processing_remark, loan_amount
                FROM notifications
                ORDER BY sent_at DESC
            """)
            db_rows = cursor.fetchall()
            for r in db_rows:
                rows.append(dict(r))
            conn.close()
        except Exception as e:
            app.logger.error(f"Error querying notifications SQLite DB: {e}")
            
    status_mapping = {
        "sent": "Sent",
        "skipped_cooldown": "Skipped (Cooldown)",
        "skipped_declined": "Skipped (Declined)",
        "skipped_missing_data": "Skipped (Missing Data)",
        "skipped_zero_loan": "Skipped - Zero Loan Amount",
        "skipped_zero_emi": "Skipped - Zero EMI",
        "skipped_foir_exceeded": "Skipped - FOIR Exceeded",
        "skipped_income_below_floor": "Skipped - Income Below Floor",
        "api_error": "Failed (LLM Error)",
        "whatsapp_error": "Failed (WhatsApp Error)",
        "unexpected_error": "Failed (Unexpected Error)"
    }
    
    formatted_rows = []
    for r in rows:
        formatted_rows.append({
            "prospect_id": r["prospect_id"],
            "name": f"Customer #{r['prospect_id']}",
            "notification_id": r["notification_id"] or "N/A",
            "tier": r["tier"],
            "loan_amount": r.get("loan_amount") or 0.0,
            "status": status_mapping.get(r["status"], r["status"]),
            "status_raw": r["status"],
            "language": r["language"],
            "channel": r.get("channel") or "WhatsApp",
            "timestamp": r["sent_at"],
            "remark": r.get("processing_remark") or ""
        })
        
    return render_template("notifications.html", rows=formatted_rows, active="notifications")


# Single customer's full detail (loan, EMI, repayment plan). 404 if the id doesn't exist.
@app.route("/customer/<int:customer_id>")
def customer_detail(customer_id):
    data = pl.get_customer(customer_id)
    if data is None:
        abort(404)
    # scale schedule bars relative to the largest yearly payment
    max_total = max((r["Total_Paid_Year"] for r in data["schedule"]), default=1) or 1
    return render_template("customer_detail.html", data=data, max_total=max_total)


# Live Prediction page. GET shows the empty form; POST scores the submitted applicant.
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = comparison = error = None
    values = {f[0]: f[4] for f in pl.FORM_FIELDS}  # defaults
    clf_name = request.form.get("clf_name") or pl.DEFAULT_CLF_NAME
    reg_name = request.form.get("reg_name") or pl.DEFAULT_REG_NAME
    if request.method == "POST":
        try:
            # read every number the user typed, then run the full scoring
            for name, *_ in pl.FORM_FIELDS:
                values[name] = float(request.form.get(name, ""))
            result = pl.score_applicant(values, clf_name=clf_name, reg_name=reg_name)
            comparison = pl.compare_classifiers(values)   # how every model would decide
        except (ValueError, TypeError):
            error = "Please enter valid numbers in every field."   # bad/empty input
    max_total = 1
    if result and result["schedule"]:
        max_total = max(r["total_paid"] for r in result["schedule"]) or 1
    return render_template("predict.html", fields=pl.FORM_FIELDS, values=values,
                           result=result, comparison=comparison, error=error,
                           max_total=max_total, choices=pl.model_choices(),
                           clf_name=clf_name, reg_name=reg_name)


# Model page -- shows how the 4 classifiers / 2 regressors compared + feature importance.
@app.route("/model")
def model():
    return render_template("model.html", report=pl.model_report())


# Insights page -- approval-rate curves and the bank's risk exposure by tier.
@app.route("/insights")
def insights():
    return render_template("insights.html", data=pl.insights_data())


# Batch Scoring page. POST reads an uploaded CSV/Excel and scores every row at once.
@app.route("/batch", methods=["GET", "POST"])
def batch():
    results = summary = error = warning = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Please choose a CSV or Excel file to upload."
        else:
            try:
                if file.filename.lower().endswith((".xlsx", ".xls")):
                    df_in = pd.read_excel(file)
                else:
                    df_in = pd.read_csv(file)
                out = pl.score_batch(df_in)
                res, missing = out["results"], out["missing"]
                out["inputs"].to_csv(BATCH_INPUTS_PATH, index=False)
                res.to_csv(BATCH_RESULT_PATH, index=False)
                approved = res["Decision"] == "Approved"
                summary = {
                    "total": int(len(res)),
                    "approved": int(approved.sum()),
                    "rejected": int((~approved).sum()),
                    "avg_loan": int(res.loc[approved, "Recommended_Loan"].mean()) if approved.any() else 0,
                    "total_book": int(res.loc[approved, "Recommended_Loan"].sum()),
                }
                results = res.head(200).to_dict("records")
                # warn if columns didn't match (why every row would look identical)
                if missing:
                    key = [m for m in ("Credit_Score", "NETMONTHLYINCOME") if m in missing]
                    warning = (f"{len(missing)} expected column(s) were not found in your file and "
                               f"were filled with the dataset median: {', '.join(missing)}.")
                    if key:
                        warning = ("Your file is missing the key column(s) "
                                   f"{', '.join(key)}, so applicants may show identical results. "
                                   "Rename your columns to match the template (or download the "
                                   "blank template below). " + warning)
            except Exception as exc:
                error = f"Could not process that file: {exc}"
    return render_template("batch.html", results=results, summary=summary,
                           error=error, warning=warning, cols=pl.BATCH_TEMPLATE_COLS)


# "View" a single applicant from the last uploaded batch -- re-scores that row and
# shows the full detail page (same as Live Prediction, but for a batch row).
@app.route("/batch/applicant/<int:row>")
def batch_applicant(row):
    if not os.path.exists(BATCH_INPUTS_PATH):
        abort(404)
    inputs_df = pd.read_csv(BATCH_INPUTS_PATH)
    if row < 1 or row > len(inputs_df):
        abort(404)
    inputs = inputs_df.iloc[row - 1].to_dict()   # rows are shown 1-based in the UI
    result = pl.score_applicant(inputs)
    max_total = max((r["total_paid"] for r in result["schedule"]), default=1) or 1
    return render_template("batch_applicant.html", result=result, row=row,
                           inputs=inputs, fields=pl.FORM_FIELDS, max_total=max_total)


# Lets the user download a blank CSV with the right column headers to fill in.
@app.route("/batch/template")
def batch_template():
    csv = pl.batch_template_df().to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=applicants_template.csv"})


# Download the full scored results of the last batch upload as a CSV.
@app.route("/batch/download")
def batch_download():
    if not os.path.exists(BATCH_RESULT_PATH):
        abort(404)
    return send_from_directory(os.path.dirname(BATCH_RESULT_PATH),
                               os.path.basename(BATCH_RESULT_PATH),
                               as_attachment=True, download_name="scored_applicants.csv")


# Serves the EDA / evaluation chart PNGs from the figures/ folder to the web pages.
@app.route("/figure/<path:name>")
def figure(name):
    if not name.endswith(".png"):     # only allow .png -- basic safety check
        abort(404)
    return send_from_directory(pl.FIGURES_DIR, name)


# A small helper the templates use to print money the Indian way (e.g. Rs.1,23,456).
@app.template_filter("inr")
def inr(value):
    """Format a number in the Indian numbering system with a Rs. prefix."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return value
    neg = value < 0
    value = abs(int(round(value)))
    s = str(value)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts) + "," + last3
    return ("-Rs." if neg else "Rs.") + s


@app.template_filter("inr_compact")
def inr_compact(value):
    """Compact Indian currency for headline stats: crore (Cr) / lakh (L)."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return value
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}Rs.{v / 1e7:,.2f} Cr"
    if v >= 1e5:
        return f"{sign}Rs.{v / 1e5:,.2f} L"
    return inr(value)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)