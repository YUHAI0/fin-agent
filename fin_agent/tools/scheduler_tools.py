from fin_agent.scheduler import TaskScheduler
from fin_agent.config import Config

scheduler = TaskScheduler()

def add_price_alert(ts_code, operator, threshold, email=None):
    """
    Add a price monitoring alert.
    :param ts_code: Stock code (e.g., 000001.SZ)
    :param operator: Comparison operator ('>', '>=', '<', '<=')
    :param threshold: Price threshold
    :param email: (Optional) Email to receive notification.
    """
    try:
        # Check if email is configured
        if not Config.is_email_configured():
            print("Email configuration is missing. You need to configure it to receive alerts.")
            # Trigger interactive setup
            Config.setup_email()
            
            # Re-check
            if not Config.is_email_configured():
                return "Error: Email configuration failed or cancelled. Cannot add alert."
        
        task_id = scheduler.add_price_alert(ts_code, operator, threshold, email)
        return f"Success: Price alert added. Task ID: {task_id}. You will be notified when {ts_code} price is {operator} {threshold}."
    except Exception as e:
        return f"Error adding alert: {str(e)}"

def list_alerts():
    """
    List all active alerts.
    """
    tasks = scheduler.list_tasks()
    if not tasks:
        return "No active alerts."
    
    result = "Active Alerts:\n"
    for t in tasks:
        status = "Active" if t.get('enabled', True) else "Fired/Disabled"
        if t['type'] == 'price_alert':
            result += f"- [{status}] {t['ts_code']} Price {t['operator']} {t['threshold']} (ID: {t['id']})\n"
        else:
            result += f"- [{status}] Unknown Task Type (ID: {t['id']})\n"
    return result

def remove_alert(task_id):
    """
    Remove an alert by ID.
    """
    if scheduler.remove_task(task_id):
        return f"Success: Alert {task_id} removed."
    else:
        return f"Error: Alert {task_id} not found."

def update_alert(task_id, ts_code=None, operator=None, threshold=None):
    """
    Update an existing alert.
    """
    if scheduler.update_price_alert(task_id, ts_code, operator, threshold):
        return f"Success: Alert {task_id} updated and re-enabled."
    else:
        return f"Error: Alert {task_id} not found."

def reset_email_config():
    """
    Reset or update email configuration interactively.
    """
    try:
        print("Initiating email configuration reset...")
        Config.setup_email()
        return "Email configuration wizard finished. New settings are applied."
    except Exception as e:
        return f"Error resetting email config: {str(e)}"

SCHEDULER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add_price_alert",
            "description": "Add a scheduled task to monitor stock price and send email notification when condition is met.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "operator": {
                        "type": "string",
                        "enum": [">", ">=", "<", "<="],
                        "description": "Comparison operator."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Price threshold to trigger alert."
                    },
                    "email": {
                        "type": "string",
                        "description": "Optional email address. If not provided, uses the default configured sender."
                    }
                },
                "required": ["ts_code", "operator", "threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_alerts",
            "description": "List all configured price alerts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_alert",
            "description": "Remove a configured alert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to remove."
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_alert",
            "description": "Update an existing price alert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to update."
                    },
                    "ts_code": {
                        "type": "string",
                        "description": "New stock code (optional)."
                    },
                    "operator": {
                        "type": "string",
                        "enum": [">", ">=", "<", "<="],
                        "description": "New comparison operator (optional)."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "New price threshold (optional)."
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_email_config",
            "description": "Reset or update the email configuration interactively. Use this when the user wants to change email notification settings.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

