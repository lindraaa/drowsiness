import smtplib
from email.mime.text import MIMEText

# Email configuration
sender_email = "chancruz739@gmail"  # Replace with your Gmail address
receiver_email = "chancruz739@gmail"  # Set it to your own Gmail address
password = "Ailestrike24"  # Use the App Password generated in your Google Account settings

# Create message
subject = "Test Email"
body = "This is a test email."
message = MIMEText(body)
message["Subject"] = subject
message["From"] = sender_email
message["To"] = receiver_email

# Connect to SMTP server
with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

print("Email sent successfully.")
