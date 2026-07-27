import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

class DedupManager:
    """Manages SQLite notification history to prevent duplicate messaging during cooldown periods."""

    def __init__(self, db_path: str, cooldown_days: int) -> None:
        """Initialize DedupManager.

        Args:
            db_path (str): Filepath to the SQLite database.
            cooldown_days (int): Interval days a prospect is locked from receiving communications.
        """
        self.db_path = db_path
        self.cooldown_days = cooldown_days
        self.logger = logging.getLogger(__name__)
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to the SQLite database.

        Returns:
            sqlite3.Connection: SQLite connection object.
        """
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_table(self) -> None:
        """Ensure the notifications registry table exists in SQLite and has correct columns."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    prospect_id INTEGER PRIMARY KEY,
                    notification_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT 'WhatsApp'
                )
                """
            )
            
            # Check if channel column exists (migration support)
            cursor.execute("PRAGMA table_info(notifications)")
            columns = [col[1] for col in cursor.fetchall()]
            if "channel" not in columns:
                self.logger.info("Migrating database: adding 'channel' column to notifications table")
                cursor.execute("ALTER TABLE notifications ADD COLUMN channel TEXT NOT NULL DEFAULT 'WhatsApp'")
            
            conn.commit()
        finally:
            conn.close()

    def is_in_cooldown(self, prospect_id: int) -> bool:
        """Determine if a customer should be skipped due to a pending cooldown window.

        Args:
            prospect_id (int): Customer prospect ID.

        Returns:
            bool: True if inside the cooldown window, False otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sent_at, status FROM notifications WHERE prospect_id = ?",
                (prospect_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            
            sent_at_str, status = row
            if status == "skipped_declined":
                return False
            
            # Parse using fromisoformat (assuming ISO representation from UTC now)
            # Remove any trailing 'Z' or offset if present, or let fromisoformat handle it
            sent_at = datetime.fromisoformat(sent_at_str)
            
            # Make sure sent_at is naive UTC to match naive comparison
            if sent_at.tzinfo is not None:
                sent_at = sent_at.astimezone(timezone.utc).replace(tzinfo=None)
            
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            delta = now - sent_at
            
            return delta.days < self.cooldown_days
        except Exception as e:
            self.logger.error(f"Error checking cooldown for PROSPECTID {prospect_id}: {str(e)}")
            return False
        finally:
            conn.close()

    def record_notification(
        self,
        prospect_id: int,
        notification_id: str,
        tier: str,
        language: str,
        status: str,
        channel: str = "WhatsApp"
    ) -> None:
        """Log a communication attempt (sent or skipped) to the SQLite registry database.

        Args:
            prospect_id (int): Customer prospect ID.
            notification_id (str): Notification ID string.
            tier (str): Credit risk tier (P1, P2, P3).
            language (str): Target language of the message.
            status (str): Outcome status (sent, skipped_cooldown, skipped_declined).
            channel (str): Dispatch channel used (e.g. WhatsApp, N/A).
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            sent_at_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            cursor.execute(
                """
                INSERT OR REPLACE INTO notifications (prospect_id, notification_id, tier, sent_at, language, status, channel)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (prospect_id, notification_id, tier, sent_at_iso, language, status, channel)
            )
            conn.commit()
            self.logger.debug(f"Recorded {status} via {channel} for PROSPECTID {prospect_id}")
        except Exception as e:
            self.logger.error(f"Failed to record notification for PROSPECTID {prospect_id}: {str(e)}")
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Fetch cumulative execution metrics from the notifications table.

        Returns:
            dict: Counts of sent, skipped, and total processed records.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(1) FROM notifications WHERE status = 'sent'")
            sent_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(1) FROM notifications WHERE status = 'skipped_cooldown'")
            cooldown_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(1) FROM notifications WHERE status = 'skipped_declined'")
            declined_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(1) FROM notifications")
            total_records = cursor.fetchone()[0]
            
            return {
                "total_sent": sent_count,
                "skipped_cooldown": cooldown_count,
                "skipped_declined": declined_count,
                "total_records": total_records
            }
        except Exception as e:
            self.logger.error(f"Error fetching stats from DB: {str(e)}")
            return {
                "total_sent": 0,
                "skipped_cooldown": 0,
                "skipped_declined": 0,
                "total_records": 0
            }
        finally:
            conn.close()

    def has_been_notified(self, prospect_id: int) -> bool:
        """Check if a customer has already been successfully notified (status == 'sent').

        Args:
            prospect_id (int): Customer prospect ID.

        Returns:
            bool: True if notified successfully, False otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(1) FROM notifications WHERE prospect_id = ? AND status = 'sent'",
                (prospect_id,)
            )
            return cursor.fetchone()[0] > 0
        except Exception as e:
            self.logger.error(f"Error checking notification status for PROSPECTID {prospect_id}: {str(e)}")
            return False
        finally:
            conn.close()

    def get_next_sequence_number(self) -> int:
        """Calculate the next sequence number starting from the maximum sequence value in the database.

        Returns:
            int: The next incremental sequence number.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT notification_id FROM notifications")
            rows = cursor.fetchall()
            max_seq = 0
            for row in rows:
                notif_id = row[0]
                if notif_id and notif_id.startswith("NOTIF-"):
                    parts = notif_id.split("-")
                    if len(parts) >= 3:
                        try:
                            seq = int(parts[-1])
                            if seq > max_seq:
                                max_seq = seq
                        except ValueError:
                            pass
            return max_seq + 1
        except Exception as e:
            self.logger.error(f"Error calculating next sequence number from DB: {str(e)}")
            return 1
        finally:
            conn.close()
