import time
import schedule
import threading
import json
import os
import logging
from pathlib import Path
from fin_agent.config import Config
from fin_agent.notification import NotificationManager

logger = logging.getLogger(__name__)

class TaskScheduler:
    _instance = None
    _started = False
    _last_mtime = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskScheduler, cls).__new__(cls)
            cls._instance.tasks = {}
            cls._instance.task_file = os.path.join(Config.get_config_dir(), "tasks.json")
            cls._instance.verbose = False
            cls._instance.load_tasks()
        return cls._instance

    def load_tasks(self):
        if not os.path.exists(self.task_file):
            self.tasks = {}
            return

        try:
            mtime = os.path.getmtime(self.task_file)
            if mtime > self._last_mtime:
                with open(self.task_file, 'r', encoding='utf-8') as f:
                    self.tasks = json.load(f)
                self._last_mtime = mtime
                # logger.debug(f"Tasks reloaded from file (mtime: {mtime})")
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def save_tasks(self):
        try:
            with open(self.task_file, 'w', encoding='utf-8') as f:
                json.dump(self.tasks, f, indent=4, ensure_ascii=False)
            # Update mtime after write to avoid reloading own changes
            self._last_mtime = os.path.getmtime(self.task_file)
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def add_price_alert(self, ts_code, operator, threshold, email=None):
        self.load_tasks() # Ensure we have latest
        task_id = f"price_alert_{ts_code}_{int(time.time())}"
        task = {
            "id": task_id,
            "type": "price_alert",
            "ts_code": ts_code,
            "operator": operator,
            "threshold": float(threshold),
            "email": email or Config.EMAIL_RECEIVER or Config.EMAIL_SENDER,
            "enabled": True,
            "created_at": time.time()
        }
        self.tasks[task_id] = task
        self.save_tasks()
        return task_id

    def update_price_alert(self, task_id, ts_code=None, operator=None, threshold=None):
        self.load_tasks()
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        if ts_code:
            task['ts_code'] = ts_code
        if operator:
            task['operator'] = operator
        if threshold is not None:
            task['threshold'] = float(threshold)
            
        # If updating, re-enable it if it was disabled/fired
        task['enabled'] = True
        
        self.save_tasks()
        return True

    def list_tasks(self):
        self.load_tasks()
        return list(self.tasks.values())

    def remove_task(self, task_id):
        self.load_tasks()
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.save_tasks()
            return True
        return False

    def check_conditions(self):
        self.load_tasks()
        
        if getattr(self, 'verbose', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker heartbeat: Checking {len(self.tasks)} tasks...")
        
        for task_id, task in list(self.tasks.items()):
            if not task.get("enabled", True):
                continue
            
            if task["type"] == "price_alert":
                self._check_price_alert(task)

    def _check_price_alert(self, task):
        from fin_agent.tools.tushare_tools import get_realtime_price
        
        ts_code = task["ts_code"]
        operator = task["operator"]
        threshold = task["threshold"]
        email = task["email"]
        
        try:
            # Note: get_realtime_price returns a JSON string or error message
            result = get_realtime_price(ts_code)
            
            if isinstance(result, str) and ("Error" in result or "No realtime data" in result):
                # Silent fail for transient network issues, but maybe log debug
                # logger.debug(f"Failed to check price for {ts_code}: {result}")
                return

            # Parse JSON
            try:
                data = json.loads(result)
            except:
                return
                
            if not data:
                return
            
            # Tushare realtime returns list of records
            record = data[0]
            current_price = float(record.get('price', 0))
            stock_name = record.get('name', ts_code)

            if getattr(self, 'verbose', False):
                print(f"  [Check] {stock_name} ({ts_code}): Current={current_price}, Condition: {operator} {threshold}")

            if current_price == 0:
                # Sometimes pre-market price is 0 or invalid
                return
            
            triggered = False
            if operator == ">" and current_price > threshold:
                triggered = True
            elif operator == ">=" and current_price >= threshold:
                triggered = True
            elif operator == "<" and current_price < threshold:
                triggered = True
            elif operator == "<=" and current_price <= threshold:
                triggered = True
                
            if triggered:
                subject = f"Price Alert: {ts_code} {operator} {threshold}"
                stock_name = record.get('name', ts_code)
                content = (
                    f"Price Alert Triggered!\n\n"
                    f"Stock: {stock_name} ({ts_code})\n"
                    f"Current Price: {current_price}\n"
                    f"Condition: Price {operator} {threshold}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                
                print(f"\n[Scheduler] Triggering task {task['id']}: {subject}")
                success = NotificationManager.send_email(subject, content, email)
                
                if success:
                    print(f"[Scheduler] Email sent to {email}")
                else:
                    print(f"[Scheduler] Failed to send email to {email}")

                # Disable task after firing (one-time alert)
                task["enabled"] = False
                self.save_tasks()
                    
        except Exception as e:
            logger.error(f"Error checking task {task['id']}: {e}")

    def run_scheduler(self):
        # Schedule the check every 1 minute
        # For stricter timing, we could do every 10 seconds, but Tushare has rate limits.
        # 1 minute is safe.
        schedule.every(1).minutes.do(self.check_conditions)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)

    def start(self):
        if not self._started:
            t = threading.Thread(target=self.run_scheduler, daemon=True)
            t.start()
            self._started = True
            # print("Background scheduler started.") 

    def run_forever(self):
        """Run the scheduler in blocking mode."""
        print(f"Starting scheduler worker... (Press Ctrl+C to stop)")
        print(f"Task file: {self.task_file}")
        
        self.verbose = True
        self.run_scheduler() 

