import time
import datetime
import threading
import json
import os
import logging
import uuid
from pathlib import Path
from fin_agent.config import Config
from fin_agent.notification import NotificationManager


import errno
import logging
import platform

logger = logging.getLogger(__name__)

# 全局唯一的调度线程与停止信号。init_agent() 会被重复调用，
# 必须在模块级去重，实例级的 _started 无法阻止重复启动。
_SCHEDULER_THREAD = None
_SCHEDULER_STOP = threading.Event()
_SCHEDULER_LOCK = threading.Lock()

# 交易日历缓存：{"date": "YYYYMMDD", "open_days": set()}
_CALENDAR_CACHE = {"date": None, "open_days": set()}

# A 股交易时段，含集合竞价与收盘缓冲
_TRADING_WINDOWS = (
    (datetime.time(9, 15), datetime.time(11, 30)),
    (datetime.time(12, 55), datetime.time(15, 5)),
)


def _is_trading_day(today_str):
    """借助数据源的交易日历判断是否交易日，按天缓存；失败则退化为工作日判断。"""
    if _CALENDAR_CACHE["date"] == today_str:
        return today_str in _CALENDAR_CACHE["open_days"]

    try:
        from fin_agent.datasources import get_provider
        df = get_provider().get_trade_calendar(today_str, today_str)
        open_days = set(df["cal_date"].astype(str).tolist())
        _CALENDAR_CACHE["date"] = today_str
        _CALENDAR_CACHE["open_days"] = open_days
        return today_str in open_days
    except Exception as e:
        logger.warning(f"Trade calendar unavailable, fallback to weekday check: {e}")
        return datetime.datetime.strptime(today_str, "%Y%m%d").weekday() < 5


def is_trading_now():
    """当前是否处于 A 股交易时段。"""
    now = datetime.datetime.now()
    if not _is_trading_day(now.strftime("%Y%m%d")):
        return False
    current = now.time()
    return any(start <= current <= end for start, end in _TRADING_WINDOWS)


def stop_scheduler():
    """通知调度线程退出。"""
    _SCHEDULER_STOP.set()


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
        # 秒级 time.time() 在同秒连建两条时会撞 ID 并覆盖；用 uuid 保证唯一
        task_id = f"price_alert_{ts_code}_{uuid.uuid4().hex[:8]}"
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

    def list_tasks_enriched(self):
        """列表附带股票名称与现价涨跌，供前端提醒弹窗展示（不写回 tasks.json）。"""
        tasks = [dict(t) for t in self.list_tasks()]
        codes = []
        seen = set()
        for t in tasks:
            code = t.get("ts_code")
            if t.get("type") == "price_alert" and code and code not in seen:
                seen.add(code)
                codes.append(code)
        if not codes:
            return tasks

        try:
            from fin_agent.datasources import get_provider

            df = get_provider().get_realtime_price(codes)
        except Exception as e:
            logger.warning(f"Failed to enrich alert quotes: {e}")
            return tasks

        quotes = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                code = row.get("ts_code")
                if not code:
                    continue
                try:
                    price = float(row.get("price") or 0)
                    change = float(row.get("change") or 0)
                    pct = float(row.get("pct_chg") or 0)
                except (TypeError, ValueError):
                    price, change, pct = 0.0, 0.0, 0.0
                quotes[str(code)] = {
                    "stock_name": row.get("name") or None,
                    "current_price": price,
                    "change": change,
                    "pct_chg": pct,
                }

        for t in tasks:
            q = quotes.get(t.get("ts_code"))
            if q:
                t.update(q)
        return tasks

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
        from fin_agent.datasources import get_provider

        try:
            ts_code = task['ts_code']
            operator = task['operator']
            threshold = task['threshold']
            email = task['email']

            df = get_provider().get_realtime_price(ts_code)
            if df is None or df.empty:
                logger.warning(f"No data for {ts_code}")
                return

            record = df.iloc[0].to_dict()
            current_price = float(record.get('price') or 0)

            if self.verbose:
                print(f"  [Task {task['id']}] {ts_code}: {current_price} (Target: {operator} {threshold})")

            # Special case for "0" price (suspension or error)
            if current_price == 0:
                return

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
                stock_name = record.get('name') or ts_code
                # 优化邮件标题，使其看起来更正规，减少被识别为垃圾邮件的可能
                subject = f"[Fin-Agent] 股价提醒: {stock_name} ({ts_code}) 触发条件"
                
                # 优化纯文本内容
                content = (
                    f"股价提醒通知\n"
                    f"================================\n"
                    f"股票名称: {stock_name}\n"
                    f"股票代码: {ts_code}\n"
                    f"当前价格: {current_price}\n"
                    f"触发条件: 价格 {operator} {threshold}\n"
                    f"触发时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"================================\n"
                    f"此邮件由 Fin-Agent 自动发送。"
                )
                
                # 添加HTML内容，提升邮件质量
                price_color = "#d9534f" if operator.startswith('>') else "#5cb85c" # 涨红跌绿(或根据涨跌逻辑调整)
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>股价提醒</title>
                    <style>
                        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f4f4f4; }}
                        .container {{ max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden; }}
                        .header {{ background-color: #0056b3; color: #ffffff; padding: 20px; text-align: center; }}
                        .header h2 {{ margin: 0; font-size: 24px; }}
                        .content {{ padding: 30px; }}
                        .stock-info {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #0056b3; }}
                        .info-row {{ display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                        .info-row:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
                        .label {{ font-weight: bold; color: #666; }}
                        .value {{ font-weight: 500; color: #333; }}
                        .price {{ font-size: 18px; font-weight: bold; color: {price_color}; }}
                        .footer {{ background-color: #eee; color: #888; padding: 15px; text-align: center; font-size: 12px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h2>股价提醒通知</h2>
                        </div>
                        <div class="content">
                            <p>您好，您关注的股票已触发提醒条件：</p>
                            <div class="stock-info">
                                <div class="info-row">
                                    <span class="label">股票名称</span>
                                    <span class="value">{stock_name}</span>
                                </div>
                                <div class="info-row">
                                    <span class="label">股票代码</span>
                                    <span class="value">{ts_code}</span>
                                </div>
                                <div class="info-row">
                                    <span class="label">当前价格</span>
                                    <span class="value price">{current_price}</span>
                                </div>
                                <div class="info-row">
                                    <span class="label">触发条件</span>
                                    <span class="value">价格 {operator} {threshold}</span>
                                </div>
                            </div>
                            <p>触发时间：{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                        <div class="footer">
                            <p>此邮件由 Fin-Agent 智能助手自动发送，请勿直接回复。</p>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                print(f"\n[Scheduler] Triggering task {task['id']}: {subject}")
                success = NotificationManager.send_email(subject, content, email, html_content=html_content)
                
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

    def run_scheduler(self, cycle=None):
        """轮询循环。间隔与交易时段开关每轮从配置读取，改动下一周期即生效。"""
        last_heartbeat = 0
        timeout = 0

        while not _SCHEDULER_STOP.wait(timeout):
            try:
                if self.verbose:
                    current_time = time.time()
                    if current_time - last_heartbeat > 5:
                        try:
                            if not os.path.exists(self.pid_file):
                                with open(self.pid_file, 'w') as f:
                                    f.write(str(os.getpid()))
                                logger.warning("Restored missing PID file.")
                            else:
                                os.utime(self.pid_file, None)
                            last_heartbeat = current_time
                        except Exception as e:
                            logger.error(f"Failed to update PID file heartbeat: {e}")

                Config.load()
                if not Config.ALERT_TRADING_HOURS_ONLY or is_trading_now():
                    self.check_conditions()
                else:
                    logger.debug("Outside trading hours, skip this cycle.")

                timeout = max(int(Config.ALERT_POLL_INTERVAL_MINUTES), 1) * 60
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                timeout = 60

    def _is_worker_running(self):
        """Check if a worker process is running using PID file heartbeat."""
        if not os.path.exists(self.pid_file):
            return False
            
        try:
            # Check if file was updated recently (heartbeat)
            # This avoids using os.kill which can be problematic on Windows
            mtime = os.path.getmtime(self.pid_file)
            age = time.time() - mtime
            
            # If heartbeat is within 20 seconds (worker updates every 5s), it's alive.
            if age < 20:
                return True
                
            # If file is older, it might be stale.
            # We assume it's NOT running to avoid blocking the interactive scheduler forever
            # in case of a crash.
            
            # Note: This means if the user is running an OLD version of the worker
            # that doesn't update mtime, this will return False, and we will start
            # a duplicate scheduler. This is a safe degradation compared to killing the process.
            
            # If it's stale (older than 20s), we should check if the process actually exists
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check process existence
                if platform.system() == "Windows":
                    # Windows
                    # OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                    # or just try tasklist / psutil. 
                    # Simpler: os.kill(pid, 0) works on Windows to check existence (permissions aside)
                    # But Python's os.kill on Windows does TerminateProcess if signal is not 0?
                    # No, os.kill(pid, 0) is supported on Windows since Python 2.7 to check validity.
                    os.kill(pid, 0)
                else:
                    # Unix
                    os.kill(pid, 0)
                    
                # If we get here, process exists but hasn't updated heartbeat.
                # Maybe it's stuck. We still treat it as "running" to be safe?
                # Or if it is VERY old, we assume dead? 
                # Let's trust the heartbeat. If heartbeat failed, worker might be frozen.
                # But here we just want to know if we should clean up the PID file.
                return False # It exists but is frozen/stale heartbeat. 
                             # Actually, if it exists, we shouldn't delete the PID file blindly.
                             
            except OSError:
                # Process does not exist
                logger.warning(f"Found stale PID file (PID {pid} not running). Removing.")
                try:
                    # Force remove just in case
                    if os.path.exists(self.pid_file):
                         os.remove(self.pid_file)
                         logger.warning(f"Removed stale PID file: {self.pid_file}")
                except Exception as e:
                    logger.error(f"Failed to remove stale PID file: {e}")
                return False
            except Exception:
                # Reading PID failed, maybe file is empty or corrupted
                try:
                    if os.path.exists(self.pid_file):
                         os.remove(self.pid_file)
                except:
                    pass
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking worker status: {e}")
            return False

    def start(self):
        """启动后台调度线程。全局至多一个，重复调用直接返回。"""
        global _SCHEDULER_THREAD

        with _SCHEDULER_LOCK:
            if _SCHEDULER_THREAD is not None and _SCHEDULER_THREAD.is_alive():
                return

            if self._is_worker_running():
                return

            _SCHEDULER_STOP.clear()
            _SCHEDULER_THREAD = threading.Thread(target=self.run_scheduler, daemon=True)
            _SCHEDULER_THREAD.start()
            self._started = True

    def run_forever(self, cycle=10):
        """Run the scheduler in blocking mode (Worker Mode)."""
        print(f"Starting scheduler worker (interval: {cycle}m)... (Press Ctrl+C to stop)")
        print(f"Task file: {self.task_file}")
        
        # Write PID file
        pid = os.getpid()
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
            
        try:
            self.verbose = True
            self.run_scheduler(cycle=cycle)
        except KeyboardInterrupt:
            print("\nWorker stopped by user.")
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            print(f"\nWorker crashed: {e}")
        finally:
            # Clean up PID file on exit
            if os.path.exists(self.pid_file):
                try:
                    # Check if it's still our PID before deleting
                    with open(self.pid_file, 'r') as f:
                        content = f.read().strip()
                    if content == str(pid):
                        os.remove(self.pid_file)
                except Exception:
                    pass
