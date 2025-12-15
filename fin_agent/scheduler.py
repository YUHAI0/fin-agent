import time
import schedule
import threading
import json
import os
import logging
from pathlib import Path
from fin_agent.config import Config
from fin_agent.notification import NotificationManager


import errno

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
            cls._instance.pid_file = os.path.join(Config.get_config_dir(), "scheduler.pid")
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
        self.load_tasks()
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
        # In interactive mode (not verbose), yield to worker if one is running
        if not self.verbose and self._is_worker_running():
            return

        self.load_tasks()
        
        if self.verbose:
            enabled_count = sum(1 for t in self.tasks.values() if t.get('enabled', True))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking {len(self.tasks)} tasks ({enabled_count} enabled)...")

        if not self.tasks:
            return

        for task_id, task in self.tasks.items():
            if not task.get('enabled', True):
                if self.verbose:
                    print(f"  [Task {task_id}] Skipped (Disabled)")
                continue
                
            if task['type'] == 'price_alert':
                self._check_price_alert(task)

    def _check_price_alert(self, task):
        from fin_agent.tools.tushare_tools import get_realtime_price
        
        try:
            ts_code = task['ts_code']
            operator = task['operator']
            threshold = task['threshold']
            email = task['email']
            
            # Fetch Price
            # Note: Tushare limits. We should probably cache this if many tasks watch same stock.
            # But for simple version, fetch one by one.
            price_info = get_realtime_price(ts_code)
            
            if not price_info:
                logger.warning(f"Could not get price for {ts_code}")
                return
                
            # Parse price (get_realtime_price returns a formatted string or list, 
            # let's assume it returns a dict-like or we parse the tool output)
            # Actually tool returns a string. We need internal function to get raw data for robustness.
            # For now, let's use tushare API directly via tools helper if available, 
            # OR parse the string. 
            # Better: use get_realtime_price but we need the float value.
            
            # Re-implementing simple fetch here to avoid parsing the AI-friendly string output of tools
            import tushare as ts
            pro = ts.pro_api(Config.TUSHARE_TOKEN)
            
            # Use Tushare's standard realtime API if available, otherwise fallback to legacy
            # NOTE: tushare.pro.realtime_quote is NOT a standard PRO interface.
            # Tushare PRO uses 'get_realtime_quotes' from legacy 'ts' package for free realtime data (Sina source).
            # The 'pro' object typically handles historical data.
            #
            # The error "请指定正确的接口名" usually means we called a non-existent method on pro_api.
            # pro.realtime_quote() is likely invalid.
            #
            # Let's use the legacy method which works for realtime snapshot.
            
            # df = pro.realtime_quote(ts_code=ts_code) # This was causing the error
            
            # Using legacy ts.get_realtime_quotes
            # It expects code like '000001', '600519', no suffix usually, but let's try handling suffix.
            code_parts = ts_code.split('.')
            code_no_suffix = code_parts[0]
            
            df = ts.get_realtime_quotes(code_no_suffix)
            
            if df is None or df.empty:
                 logger.warning(f"No data for {ts_code}")
                 return

            # Legacy df columns are: name, open, pre_close, price, high, low, bid, ask, volume, amount, date, time, code
            current_price = float(df.iloc[0]['price'])
            
            if self.verbose:
                print(f"  [Task {task['id']}] {ts_code}: {current_price} (Target: {operator} {threshold})")

            # Special case for "0" price (suspension or error)
            if current_price == 0:
                return
            
            record = df.iloc[0].to_dict()
            
            # Compare
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
            if "403" in str(e) and "Forbidden" in str(e):
                 msg = f"Tushare API 403 Forbidden. Please check your token validity and permissions."
                 if self.verbose:
                     print(f"  [Task {task['id']}] Error: {msg}")
                 else:
                     # In interactive mode, print to stderr or just log
                     logger.error(f"Error checking task {task['id']}: {msg}")
            else:
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

    def _is_worker_running(self):
        """Check if a worker process is running using PID file."""
        if not os.path.exists(self.pid_file):
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return False
                pid = int(content)
            
            # Check if process exists
            try:
                os.kill(pid, 0) # Signal 0 doesn't kill, just checks existence
                return True
            except OSError as e:
                # Handle specific errors
                if e.errno == errno.ESRCH: # No such process
                    logger.info(f"Found stale PID file (PID {pid}), removing...")
                    try:
                        os.remove(self.pid_file)
                    except OSError:
                        pass # Ignore if already gone
                    return False
                elif e.errno == errno.EPERM: # Permission denied (Process exists)
                    return True
                else:
                    # Other error, assume process exists to be safe
                    logger.warning(f"Error checking PID {pid}: {e}")
                    return True
        except Exception as e:
            logger.error(f"Error checking worker status: {e}")
            return False

    def start(self):
        """
        Start the scheduler in a background thread.
        Will NOT start if a worker process is detected via PID file.
        """
        if self._started:
            return

        if self._is_worker_running():
            print(f"[{time.strftime('%H:%M:%S')}] Detected active Worker process. Interactive scheduler disabled to avoid duplicates.")
            return

        t = threading.Thread(target=self.run_scheduler, daemon=True)
        t.start()
        self._started = True
        # print("Background scheduler started.") 

    def run_forever(self):
        """Run the scheduler in blocking mode (Worker Mode)."""
        print(f"Starting scheduler worker... (Press Ctrl+C to stop)")
        print(f"Task file: {self.task_file}")
        
        # Write PID file
        pid = os.getpid()
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
            
        try:
            self.verbose = True
            self.run_scheduler()
        finally:
            # Clean up PID file on exit
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
