import csv
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

from agent.config import load_config, setup_directories
from agent.loader import DataLoader
from agent.eligibility import EligibilityChecker
from agent.dedup import DedupManager
from agent.explainer import ExplainabilityLayer
from agent.multilang import LanguageSelector
from agent.generator import MessageGenerator
from agent.llm.factory import create_llm_provider
from agent.llm.base import LLMProviderError
from agent.whatsapp_sender import send_whatsapp

CSV_HEADERS = [
    "notification_id",
    "prospect_id",
    "tier",
    "loan_amount",
    "interest_rate",
    "tenure_years",
    "monthly_emi",
    "total_payable",
    "language",
    "generated_at",
    "message_preview",
    "status",
    "contact_no"
]

class AgentScheduler:
    """Orchestrates the entire batch notification pipeline processing cycle."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the AgentScheduler.

        Args:
            config_path (str): Filepath to the global configuration YAML file.
        """
        config = load_config(config_path)
        setup_directories(config)
        
        self.config = config
        self.dataset_config = config["dataset"]
        self.eligibility_config = config["eligibility"]
        self.cooldown_config = config["cooldown"]
        self.scheduler_config = config["scheduler"]
        self.llm_config = config["llm"]
        self.notification_config = config["notification"]
        self.output_config = config["output"]
        self.max_sends_per_run = int(self.scheduler_config.get("max_sends_per_run", 999999))

        # Resolve paths
        self.csv_path = self.output_config["log_csv"]
        self.db_path = self.output_config["sqlite_db"]

        # Initialize helper modules
        self.loader = DataLoader(config)
        self.eligibility = EligibilityChecker(config)
        self.dedup = DedupManager(
            db_path=self.db_path,
            cooldown_days=int(self.cooldown_config["days"])
        )
        self.loader.recover_db_from_excel()
        self.explainer = ExplainabilityLayer()
        self.multilang = LanguageSelector(
            default_lang=self.notification_config["default_language"]
        )
        self.llm = create_llm_provider(config)
        self.generator = MessageGenerator(
            llm=self.llm,
            offer_validity_days=int(self.notification_config["offer_validity_days"])
        )
        self.logger = logging.getLogger(__name__)

    def _ensure_csv(self) -> None:
        """Verify the CSV audit log exists, creating it with headers if necessary."""
        if not os.path.exists(self.csv_path):
            try:
                # Ensure parent dir exists
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                self.logger.info(f"Created new CSV audit log at: {self.csv_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize CSV log: {str(e)}")

    def _write_csv_row(self, row_dict: dict) -> None:
        """Append a processed transaction row into the CSV audit log.

        Args:
            row_dict (dict): A dictionary structured with CSV_HEADERS keys.
        """
        try:
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row_dict)
        except Exception as e:
            self.logger.error(f"Failed writing row to CSV log: {str(e)}")

    def _build_notification_id(self, prospect_id: int) -> str:
        """Create a traceable notification string key from prospect metadata parameters.

        Args:
            prospect_id (int): Customer prospect ID.

        Returns:
            str: Traceable tracking notification ID.
        """
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        fmt = self.notification_config.get("id_format", "NOTIF-{date}-{prospect_id:05d}")
        return fmt.format(date=date_str, prospect_id=prospect_id)

    def run_pipeline(self) -> dict:
        """Run the batch notification processing cycle, loading new rows and generating alerts.

        Returns:
            dict: Summary metrics dictionary of processed outcomes.
        """
        self.logger.info("Pipeline started — checking for new records")
        sent = 0
        skipped = 0
        errors = 0
        self._ensure_csv()

        try:
            df = self.loader.load_data()
        except Exception as e:
            self.logger.error(f"Pipeline aborting. Failed to load Excel data: {str(e)}")
            return {"sent": 0, "skipped": 0, "errors": 0, "total_processed": 0, "max_prospect_id": 0}

        new_records_df = self.loader.get_new_records(df)
        if new_records_df.empty:
            self.logger.info(
                f"No new records found. Next run in {self.scheduler_config['interval_hours']} hours."
            )
            return {"sent": 0, "skipped": 0, "errors": 0, "total_processed": 0, "max_prospect_id": 0}

        max_processed_id = 0
        
        # Get next sequence number for unique Notification IDs
        next_seq = self.dedup.get_next_sequence_number()

        # Process row by row
        for idx, pandas_row in new_records_df.iterrows():
            if sent >= self.max_sends_per_run:
                self.logger.info(
                    f"Reached maximum send limit of {self.max_sends_per_run} notifications. "
                    f"Stopping pipeline execution for this run."
                )
                break
            row = pandas_row.to_dict()
            prospect_id = int(row["PROSPECTID"])
            self.logger.info(f"Processing PROSPECTID {prospect_id}")

            try:
                # Requirement 3: Check database first. If already successfully notified, skip them.
                if self.dedup.has_been_notified(prospect_id):
                    self.logger.info(f"PROSPECTID {prospect_id} already notified successfully. Skipping.")
                    skipped += 1
                    max_processed_id = max(max_processed_id, prospect_id)
                    # Update watermark immediately (Requirement 6: crash resilience)
                    self.loader.update_watermark(prospect_id)
                    continue

                # Check eligibility
                eligibility_dict = self.eligibility.check(row)
                if eligibility_dict is None:
                    # Declined prospect (P3)
                    self.dedup.record_notification(
                        prospect_id=prospect_id,
                        notification_id="",
                        tier=str(row.get("Approved_Flag", "P3")),
                        language="N/A",
                        status="skipped_declined",
                        channel="N/A"
                    )
                    self.logger.warning(f"Skipped PROSPECTID {prospect_id} — declined (P3)")
                    skipped += 1
                    max_processed_id = max(max_processed_id, prospect_id)
                    # Update watermark immediately (Requirement 6: crash resilience)
                    self.loader.update_watermark(prospect_id)
                    continue

                # Check cooldown
                if self.dedup.is_in_cooldown(prospect_id):
                    self.dedup.record_notification(
                        prospect_id=prospect_id,
                        notification_id="",
                        tier=eligibility_dict["tier"],
                        language="N/A",
                        status="skipped_cooldown",
                        channel="N/A"
                    )
                    self.logger.warning(f"Skipped PROSPECTID {prospect_id} — in cooldown")
                    skipped += 1
                    max_processed_id = max(max_processed_id, prospect_id)
                    # Update watermark immediately (Requirement 6: crash resilience)
                    self.loader.update_watermark(prospect_id)
                    continue

                # Language detection
                language = self.multilang.detect_language(prospect_id, row)

                # Build identifiers (using global sequence number) and format reasons
                notification_id = self._build_notification_id(next_seq)
                enriched_reason = self.explainer.enrich_reason(
                    reason_str=eligibility_dict["reason"],
                    credit_score=eligibility_dict["credit_score"],
                    tier=eligibility_dict["tier"]
                )

                # Generate messaging draft via the selected provider
                try:
                    message_content = self.generator.generate(
                        eligibility_dict=eligibility_dict,
                        notification_id=notification_id,
                        language=language,
                        enriched_reason=enriched_reason
                    )
                    send_whatsapp(message_content)
                except LLMProviderError as l_err:
                    self.logger.critical(
                        f"LLM Provider generation failed for PROSPECTID {prospect_id}: {str(l_err)}"
                    )
                    self.dedup.record_notification(
                        prospect_id=prospect_id,
                        notification_id=notification_id,
                        tier=eligibility_dict["tier"],
                        language=language,
                        status="api_error",
                        channel="WhatsApp"
                    )
                    errors += 1
                    max_processed_id = max(max_processed_id, prospect_id)
                    # Update watermark immediately (Requirement 6: crash resilience)
                    self.loader.update_watermark(prospect_id)
                    continue

                # Log results to CSV audit
                csv_row = {
                    "notification_id": notification_id,
                    "prospect_id": prospect_id,
                    "tier": eligibility_dict["tier"],
                    "loan_amount": eligibility_dict["loan_amount"],
                    "interest_rate": eligibility_dict["interest_rate"],
                    "tenure_years": eligibility_dict["tenure_years"],
                    "monthly_emi": eligibility_dict["monthly_emi"],
                    "total_payable": eligibility_dict["total_payable"],
                    "language": language,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "message_preview": self.generator.generate_preview(message_content),
                    "status": "sent",
                    "contact_no": eligibility_dict.get("contact_no", "")
                }
                self._write_csv_row(csv_row)

                # Record in dedup
                self.dedup.record_notification(
                    prospect_id=prospect_id,
                    notification_id=notification_id,
                    tier=eligibility_dict["tier"],
                    language=language,
                    status="sent",
                    channel="WhatsApp"
                )

                sent += 1
                next_seq += 1  # Increment sequence number for next sent message
                max_processed_id = max(max_processed_id, prospect_id)
                
                # Update watermark immediately (Requirement 6: crash resilience)
                self.loader.update_watermark(prospect_id)
                self.logger.info(f"Notification generated: {notification_id} | Lang: {language}")

            except Exception as row_err:
                self.logger.error(
                    f"Unexpected error processing PROSPECTID {prospect_id}. Traceback:\n"
                    f"{traceback.format_exc()}"
                )
                errors += 1
                max_processed_id = max(max_processed_id, prospect_id)
                # Update watermark immediately (Requirement 6: crash resilience)
                self.loader.update_watermark(prospect_id)
                continue

        # Step 5: Update watermark
        if max_processed_id > 0:
            self.loader.update_watermark(max_processed_id)

        # Step 6: Synchronize database states back to Excel file
        try:
            self.loader.sync_db_to_excel()
        except Exception as sync_err:
            self.logger.error(f"Excel sync failed: {str(sync_err)}")

        self.logger.info(f"Pipeline complete — Sent: {sent}, Skipped: {skipped}, Errors: {errors}")

        return {
            "sent": sent,
            "skipped": skipped,
            "errors": errors,
            "total_processed": sent + skipped + errors,
            "max_prospect_id": max_processed_id
        }
