import os
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from agent.dedup import DedupManager

@pytest.fixture
def temp_db(tmp_path) -> str:
    """Fixture returning path string to a temporary SQLite database file."""
    db_file = tmp_path / "test_notifications.db"
    return str(db_file)

@pytest.fixture
def dedup(temp_db) -> DedupManager:
    """Fixture initializing DedupManager with a 7-day cooldown configuration."""
    return DedupManager(temp_db, cooldown_days=7)

def test_not_in_cooldown_new_prospect(dedup):
    """Verify that a brand new prospect ID is not blocked by cooldown."""
    assert dedup.is_in_cooldown(999) is False

def test_record_and_check_cooldown(dedup):
    """Verify that recording a sent notification puts the prospect into cooldown."""
    dedup.record_notification(1, "NOTIF-001", "P1", "English", "sent")
    assert dedup.is_in_cooldown(1) is True

def test_declined_not_in_cooldown(dedup):
    """Verify that skipped_declined notifications do not trigger a cooldown."""
    dedup.record_notification(2, "", "P3", "N/A", "skipped_declined")
    assert dedup.is_in_cooldown(2) is False

def test_cooldown_expired(dedup, temp_db):
    """Verify that prospects whose notification is older than cooldown_days are free to message."""
    prospect_id = 10
    sent_at_past = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=8)).isoformat()
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO notifications (prospect_id, notification_id, tier, sent_at, language, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (prospect_id, "NOTIF-OLD", "P1", sent_at_past, "English", "sent")
    )
    conn.commit()
    conn.close()

    assert dedup.is_in_cooldown(prospect_id) is False

def test_get_stats_empty(dedup):
    """Verify database metrics return zeros when notifications registry is empty."""
    stats = dedup.get_stats()
    assert stats["total_sent"] == 0
    assert stats["skipped_cooldown"] == 0
    assert stats["skipped_declined"] == 0
    assert stats["total_records"] == 0

def test_get_stats_after_records(dedup):
    """Verify registry metric counts match stored records accurately."""
    dedup.record_notification(1, "NOTIF-001", "P1", "English", "sent")
    dedup.record_notification(2, "NOTIF-002", "P2", "Hindi", "sent")
    dedup.record_notification(3, "", "P1", "N/A", "skipped_cooldown")
    dedup.record_notification(4, "", "P3", "N/A", "skipped_declined")

    stats = dedup.get_stats()
    assert stats["total_sent"] == 2
    assert stats["skipped_cooldown"] == 1
    assert stats["skipped_declined"] == 1
    assert stats["total_records"] == 4

def test_has_been_notified(dedup):
    """Verify that we can check if a prospect was already successfully notified."""
    assert dedup.has_been_notified(5) is False
    dedup.record_notification(5, "NOTIF-20260727-00005", "P1", "English", "sent", "WhatsApp")
    assert dedup.has_been_notified(5) is True
    
    # Skipped records should not count as successfully notified
    dedup.record_notification(6, "", "P3", "N/A", "skipped_declined", "N/A")
    assert dedup.has_been_notified(6) is False

def test_get_next_sequence_number(dedup):
    """Verify sequence calculation auto-increments based on largest existing ID in database."""
    assert dedup.get_next_sequence_number() == 1
    
    dedup.record_notification(1, "NOTIF-20260727-00045", "P1", "English", "sent")
    assert dedup.get_next_sequence_number() == 46
    
    dedup.record_notification(2, "NOTIF-20260727-00002", "P2", "English", "sent")
    # Still 46 since 45 is the maximum
    assert dedup.get_next_sequence_number() == 46
