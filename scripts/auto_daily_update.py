#!/usr/bin/env python3
"""
Simple Automated Daily Market Data Update Script

This script automatically runs the fetch_data.py and load_to_db.py scripts.

Usage:
    python scripts/auto_daily_update.py

Scheduling:
    Windows: Use Task Scheduler to run daily
    Linux/Mac: Use cron to run daily
"""

import subprocess
import sys
import smtplib
import json
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Get project root directory
project_root = Path(__file__).parent.parent
fetch_script = project_root / "scripts" / "fetch_data.py"
load_script = project_root / "scripts" / "load_to_db.py"
config_file = project_root / "config" / "email_config.json"


def load_email_config():
    """Load email configuration from file."""
    if not config_file.exists():
        return None

    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading email config: {e}")
        return None


def send_email_notification(success: bool, message: str, details: str = ""):
    """Send email notification about update status."""
    config = load_email_config()

    if not config or not config.get('enabled', False):
        return

    # Check if we should send this type of notification
    if success and not config.get('send_on_success', False):
        return
    if not success and not config.get('send_on_failure', True):
        return

    try:
        # Prepare email
        subject = f"Market Data Update {'Success' if success else 'Failed'} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        body = f"""
Market Data Update Report
========================

Status: {'SUCCESS' if success else 'FAILED'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

{details if details else ''}

---
Automated Market Data Update System
"""

        # Create email
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = ', '.join(config['recipient_emails'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])

        for recipient in config['recipient_emails']:
            server.sendmail(config['sender_email'], recipient, msg.as_string())

        server.quit()
        print(f"Email notification sent to {', '.join(config['recipient_emails'])}")

    except Exception as e:
        print(f"Failed to send email notification: {e}")


def create_email_config():
    """Create a default email configuration file."""
    default_config = {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "your-email@gmail.com",
        "sender_password": "your-app-password",
        "recipient_emails": ["recipient@example.com"],
        "send_on_success": True,
        "send_on_failure": True
    }

    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"Email configuration template created at: {config_file}")
    print("Please edit the file with your email settings and set 'enabled': true")


def run_script(script_path):
    """Run a Python script and handle errors."""
    try:
        print(f"Running {script_path.name}...")
        result = subprocess.run([sys.executable, str(script_path)],
                              capture_output=True, text=True, cwd=project_root)

        # Print output
        if result.stdout:
            print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print(f"Error running {script_path.name}:")
            if result.stderr:
                print(result.stderr)
            return False

        print(f"{script_path.name} completed successfully")
        return True

    except Exception as e:
        print(f"Failed to run {script_path.name}: {e}")
        return False

def main():
    """Main function to run fetch and load scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Automated Market Data Update")
    parser.add_argument('--create-email-config', action='store_true',
                       help='Create email configuration template')
    args = parser.parse_args()

    # Create email config if requested
    if args.create_email_config:
        create_email_config()
        return 0

    start_time = datetime.now()
    print("Starting automated daily update...")

    details = []

    # Step 1: Run fetch script
    print("Step 1: Fetching market data...")
    if not run_script(fetch_script):
        error_msg = "Daily update failed during fetch step"
        print(error_msg)
        send_email_notification(False, error_msg, "The data fetching process encountered an error.")
        return 1

    details.append("✓ Data fetch completed successfully")

    # Step 2: Run load script
    print("Step 2: Loading data to database...")
    if not run_script(load_script):
        error_msg = "Daily update failed during load step"
        print(error_msg)
        send_email_notification(False, error_msg, "The database loading process encountered an error.")
        return 1

    details.append("✓ Database load completed successfully")

    # Success!
    duration = datetime.now() - start_time
    success_msg = f"Daily update completed successfully in {duration}"
    print(success_msg)

    details_text = "\n".join(details) + f"\n\nTotal duration: {duration}"
    send_email_notification(True, success_msg, details_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())

