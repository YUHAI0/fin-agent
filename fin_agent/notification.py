import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import logging
from fin_agent.config import Config

logger = logging.getLogger(__name__)

class NotificationManager:
    @staticmethod
    def send_email(subject, content, receiver=None):
        """
        Send an email notification.
        
        Args:
            subject (str): Email subject
            content (str): Email body content
            receiver (str, optional): Receiver email. If None, sends to self (EMAIL_SENDER).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not Config.is_email_configured():
            logger.warning("Email configuration is missing. Cannot send notification.")
            return False
            
        smtp_server = Config.EMAIL_SMTP_SERVER
        smtp_port = Config.EMAIL_SMTP_PORT
        sender = Config.EMAIL_SENDER
        password = Config.EMAIL_PASSWORD
        
        if not receiver:
            receiver = sender
            
        try:
            message = MIMEMultipart()
            message['From'] = Header(sender, 'utf-8')
            message['To'] = Header(receiver, 'utf-8')
            message['Subject'] = Header(subject, 'utf-8')
            
            message.attach(MIMEText(content, 'plain', 'utf-8'))
            
            # Connect to server
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                
            server.login(sender, password)
            server.sendmail(sender, [receiver], message.as_string())
            server.quit()
            
            logger.info(f"Email sent to {receiver}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

