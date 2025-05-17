"""
Email service for Luna AI
Handles sending emails for user registration, password reset, etc.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template
from pydantic import EmailStr

from app.config import settings

logger = logging.getLogger(__name__)

# Skip email configuration if settings are not available
SMTP_ENABLED = settings.SMTP_ENABLED and all([
    settings.MAIL_USERNAME, 
    settings.MAIL_PASSWORD, 
    settings.MAIL_FROM
])

# Only import and configure if SMTP is enabled
if SMTP_ENABLED:
    from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates/email")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    email_config = ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME or "",
        MAIL_PASSWORD=settings.MAIL_PASSWORD or "",
        MAIL_FROM=settings.MAIL_FROM or "",
        MAIL_PORT=settings.MAIL_PORT or 587,
        MAIL_SERVER=settings.MAIL_SERVER or "",
        MAIL_FROM_NAME=settings.MAIL_FROM_NAME or "Luna AI",
        MAIL_STARTTLS=settings.MAIL_USE_TLS,
        MAIL_SSL_TLS=settings.MAIL_USE_SSL,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
        TEMPLATE_FOLDER=templates_dir,
    )
    
    fast_mail = FastMail(email_config)
else:
    logger.warning("Email functionality disabled due to incomplete configuration")
    fast_mail = None

async def send_email(
    recipients: List[EmailStr],
    subject: str,
    body: str,
    html_content: Optional[str] = None,
) -> bool:
    """
    Send an email to recipients
    
    Args:
        recipients: List of email addresses
        subject: Email subject
        body: Email body (plain text)
        html_content: HTML content (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    if not SMTP_ENABLED or fast_mail is None:
        logger.warning("Email sending is disabled - cannot send email")
        return False
    
    try:
        message = MessageSchema(
            subject=subject,
            recipients=recipients,
            body=body,
            html=html_content or "",
        )
        
        await fast_mail.send_message(message)
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

async def send_reset_password_email(
    email: str, username: str, token: str, base_url: Optional[str] = None
) -> bool:
    """
    Send password reset email
    
    Args:
        email: User email
        username: User username
        token: Reset token
        base_url: Base URL for reset link (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    site_url = base_url or settings.FRONTEND_URL
    reset_link = f"{site_url}/reset-password?token={token}"
    
    # Simple template for the email content
    email_template = """
    Hello {{ username }},
    
    You have requested a password reset for your Luna AI account.
    
    Please click the link below to reset your password:
    {{ reset_link }}
    
    This link will expire in 24 hours.
    
    If you did not request a password reset, please ignore this email.
    
    Best regards,
    Luna AI Team
    """
    
    # Render the template
    template = Template(email_template)
    email_content = template.render(username=username, reset_link=reset_link)
    
    # HTML version
    html_template = """
    <html>
    <body>
        <h2>Luna AI Password Reset</h2>
        <p>Hello {{ username }},</p>
        <p>You have requested a password reset for your Luna AI account.</p>
        <p>Please click the link below to reset your password:</p>
        <p><a href="{{ reset_link }}">Reset Password</a></p>
        <p>This link will expire in 24 hours.</p>
        <p>If you did not request a password reset, please ignore this email.</p>
        <p>Best regards,<br>Luna AI Team</p>
    </body>
    </html>
    """
    
    html_content = Template(html_template).render(username=username, reset_link=reset_link)
    
    # Send the email
    return await send_email(
        recipients=[email],
        subject="Luna AI - Password Reset",
        body=email_content,
        html_content=html_content,
    )

async def send_welcome_email(email: str, username: str) -> bool:
    """
    Send welcome email to new users
    
    Args:
        email: User email
        username: User username
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Simple template for the email content
    email_template = """
    Welcome to Luna AI, {{ username }}!
    
    Thank you for joining Luna AI, your advanced video analysis platform.
    
    You can now upload videos and get AI-powered insights, transcriptions, and analysis.
    
    Best regards,
    Luna AI Team
    """
    
    # Render the template
    template = Template(email_template)
    email_content = template.render(username=username)
    
    # HTML version
    html_template = """
    <html>
    <body>
        <h2>Welcome to Luna AI</h2>
        <p>Hello {{ username }}!</p>
        <p>Thank you for joining Luna AI, your advanced video analysis platform.</p>
        <p>You can now upload videos and get AI-powered insights, transcriptions, and analysis.</p>
        <p>Best regards,<br>Luna AI Team</p>
    </body>
    </html>
    """
    
    html_content = Template(html_template).render(username=username)
    
    # Send the email
    return await send_email(
        recipients=[email],
        subject="Welcome to Luna AI",
        body=email_content,
        html_content=html_content,
    )