"""
Email utilities for Luna AI
Handles sending emails for user notifications

Merged from:
- app/utils/email.py
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from app.config import settings

# Configure logging
logger = logging.getLogger("email")

async def send_email(
    email_to: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None
) -> bool:
    """
    Send an email using SMTP
    
    Args:
        email_to: Recipient email address
        subject: Email subject
        html_content: HTML content of the email
        text_content: Optional plain text content
        
    Returns:
        True if email was sent successfully, False otherwise
    """
    # Try FastAPI-Mail if available
    try:
        from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
        
        # Default email if none is provided
        DEFAULT_EMAIL_FROM = "noreply@luna-ai.example.com"
        
        # Configure FastMail
        conf = ConnectionConfig(
            MAIL_USERNAME=settings.SMTP_USER or "",
            MAIL_PASSWORD=settings.SMTP_PASSWORD or "",
            MAIL_FROM=settings.EMAILS_FROM_EMAIL or DEFAULT_EMAIL_FROM,
            MAIL_PORT=settings.SMTP_PORT or 587,
            MAIL_SERVER=settings.SMTP_HOST or "localhost",
            MAIL_STARTTLS=settings.SMTP_TLS,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=bool(settings.SMTP_USER and settings.SMTP_PASSWORD),
            VALIDATE_CERTS=True
        )
        
        message = MessageSchema(
            subject=subject,
            recipients=[email_to],
            body=html_content,
            subtype=MessageType.html
        )
        
        fm = FastMail(conf)
        await fm.send_message(message)
        logger.info(f"Email sent to {email_to} using FastAPI-Mail")
        return True
        
    except ImportError:
        logger.info("FastAPI-Mail not available, using standard SMTP")
        
        # Fallback to standard SMTP
        if not settings.SMTP_ENABLED:
            logger.warning("Email sending is disabled. Would have sent email to %s", email_to)
            return False
            
        # Create message container
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.EMAILS_FROM_EMAIL
        msg["To"] = email_to
        
        # Create the text part
        if text_content:
            part1 = MIMEText(text_content, "plain")
            msg.attach(part1)
        
        # Create the HTML part
        part2 = MIMEText(html_content, "html")
        msg.attach(part2)
        
        try:
            # Connect to SMTP server
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
            if settings.SMTP_TLS:
                server.starttls()
            
            # Login if credentials provided
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            
            # Send email
            server.sendmail(settings.EMAILS_FROM_EMAIL, email_to, msg.as_string())
            server.quit()
            logger.info(f"Email sent successfully to {email_to}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {email_to}: {str(e)}")
            return False

async def send_reset_password_email(email: str, token: str, user_name: Optional[str] = None) -> bool:
    """
    Send password reset email
    
    Args:
        email: User's email address
        token: Password reset token
        user_name: Optional user's name for personalization
        
    Returns:
        True if email was sent successfully, False otherwise
    """
    # Create the reset URL
    frontend_url = settings.FRONTEND_URL or "http://localhost:5173"  # Default for development
    reset_url = f"{frontend_url}/reset-password/{token}"
    
    # Use provided name or generic greeting
    greeting = f"Hello {user_name}," if user_name else "Hello,"
    
    # Create email content
    subject = "Luna AI - Password Reset"
    
    # HTML version
    html_content = f"""
    <html>
        <body>
            <h2>Password Reset</h2>
            <p>{greeting}</p>
            <p>You've requested a password reset for your Luna AI account.</p>
            <p>Please click the link below to set a new password:</p>
            <p><a href="{reset_url}">Reset Password</a></p>
            <p>If you didn't request this, please ignore this email.</p>
            <p>This link will expire in 24 hours.</p>
            <p>Best regards,<br>Luna AI Team</p>
        </body>
    </html>
    """
    
    # Plain text version
    text_content = f"""
    Hello {user_name or ''},
    
    You've requested a password reset for your Luna AI account.
    
    Please visit the following link to reset your password:
    {reset_url}
    
    If you didn't request this, please ignore this email.
    This link will expire in 24 hours.
    
    Best regards,
    Luna AI Team
    """
    
    return await send_email(email, subject, html_content, text_content)