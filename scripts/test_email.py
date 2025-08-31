#!/usr/bin/env python3
"""
Test script for email notifications.
Use this to test your email configuration before scheduling the automated updates.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.auto_daily_update import send_email_notification, load_email_config

def main():
    """Test email notification functionality."""
    print("Testing email notification system...")
    
    # Check if email is configured
    config = load_email_config()
    
    if not config:
        print("❌ No email configuration found.")
        print("Run: python scripts/auto_daily_update.py --create-email-config")
        return 1
    
    if not config.get('enabled', False):
        print("❌ Email notifications are disabled.")
        print("Edit config/email_config.json and set 'enabled': true")
        return 1
    
    print(f"📧 Email configuration found:")
    print(f"   Server: {config['smtp_server']}:{config['smtp_port']}")
    print(f"   From: {config['sender_email']}")
    print(f"   To: {', '.join(config['recipient_emails'])}")
    print(f"   Send on success: {config['send_on_success']}")
    print(f"   Send on failure: {config['send_on_failure']}")
    
    # Send test emails
    print("\n🧪 Sending test emails...")
    
    try:
        # Test success notification
        if config['send_on_success']:
            print("Sending success test email...")
            send_email_notification(
                True, 
                "Email test successful!", 
                "This is a test email to verify your email configuration is working correctly."
            )
        
        # Test failure notification  
        if config['send_on_failure']:
            print("Sending failure test email...")
            send_email_notification(
                False, 
                "Email test - simulated failure", 
                "This is a test email simulating a failure notification."
            )
        
        print("✅ Test emails sent successfully!")
        print("Check your inbox (and spam folder) for the test emails.")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to send test emails: {e}")
        print("\nTroubleshooting tips:")
        print("- For Gmail, use an app password (not your regular password)")
        print("- Enable 2-factor authentication first")
        print("- Check SMTP server and port settings")
        print("- Verify your email and password are correct")
        return 1

if __name__ == "__main__":
    sys.exit(main())
