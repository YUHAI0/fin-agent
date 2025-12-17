import sys
import argparse
import subprocess
import re
import os
import platform
import tempfile
import csv
import io
import signal
from importlib.metadata import version, PackageNotFoundError
import colorama
from colorama import Fore, Style
from rich.console import Console
from fin_agent.utils import FinMarkdown
from fin_agent.agent.core import FinAgent
from fin_agent.config import Config
from fin_agent.scheduler import TaskScheduler

# Initialize colorama
colorama.init()

def get_version():
    try:
        return version("fin-agent")
    except PackageNotFoundError:
        return "unknown (dev)"

def parse_version(v_str):
    """
    Parse a version string into a tuple of integers.
    Example: '0.2.1' -> (0, 2, 1)
             '0.2.1rc1' -> (0, 2, 1)
    """
    if v_str == "unknown (dev)":
        return (0, 0, 0)
    parts = []
    for part in v_str.split('.'):
        if part.isdigit():
            parts.append(int(part))
        else:
            # Handle suffixes like rc1, b1 etc by just taking the leading digits
            match = re.match(r'(\d+)', part)
            if match:
                parts.append(int(match.group(1)))
            else:
                parts.append(0)
    return tuple(parts)

def post_upgrade_hook():
    """
    Hook to be run AFTER the package has been upgraded.
    This runs in the context of the NEW version.
    """
    try:
        # Try to read old version from file first (more reliable)
        version_file = os.path.join(Config.get_config_dir(), ".upgrade_old_version")
        old_version_str = "0.0.0"
        
        if os.path.exists(version_file):
            try:
                with open(version_file, "r") as f:
                    old_version_str = f.read().strip()
                # Clean up
                os.remove(version_file)
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Failed to read version file: {e}{Style.RESET_ALL}")
        
        # Fallback to env var
        if old_version_str == "0.0.0":
            old_version_str = os.environ.get("FIN_AGENT_OLD_VERSION", "0.0.0")

        new_version_str = get_version()
        print(f"{Fore.CYAN}Running post-upgrade hook... (v{old_version_str} -> v{new_version_str}){Style.RESET_ALL}")
        
        curr_tuple = parse_version(old_version_str)
        new_tuple = parse_version(new_version_str)
        
        # Migration Logic (Now defined in the NEW version)
        # 1. old < 0.2.1 AND new >= 0.2.1 (Config location change)
        target_v2_1 = (0, 2, 1)
        
        need_clear = False
        
        if curr_tuple < target_v2_1 and new_tuple >= target_v2_1:
            need_clear = True
        # 2. For 0.3.x series, due to frequent config structure changes, 
        #    we force token clear on ANY upgrade within/to this series.
        elif new_tuple[0] == 0 and new_tuple[1] == 3 and curr_tuple < new_tuple:
            need_clear = True
            
        if need_clear:
             print(f"{Fore.YELLOW}Major configuration update detected.{Style.RESET_ALL}")
             print(f"{Fore.YELLOW}Clearing old configuration to support new features...{Style.RESET_ALL}")
             Config.clear()
             print(f"{Fore.GREEN}Configuration cleared. Please restart the agent to re-configure.{Style.RESET_ALL}")
        else:
             print(f"{Fore.GREEN}Upgrade complete. No configuration reset needed.{Style.RESET_ALL}")
             
    except Exception as e:
        print(f"{Fore.RED}Error in post-upgrade hook: {e}{Style.RESET_ALL}")

def check_and_kill_processes():
    """
    Check for other running fin-agent processes and ask user to kill them.
    Returns True if safe to proceed (processes killed or none found), False if user cancelled.
    """
    current_pid = os.getpid()
    pids = []
    
    try:
        if platform.system() == "Windows":
            # Windows implementation using PowerShell (WMIC is deprecated/missing on some systems)
            # We look for *fin-agent* (exe/script) or *fin_agent* (module)
            ps_cmd = (
                'Get-CimInstance Win32_Process | '
                'Where-Object { $_.CommandLine -like "*fin-agent*" -or $_.CommandLine -like "*fin_agent*" } | '
                'Select-Object ProcessId, CommandLine | '
                'ConvertTo-Csv -NoTypeInformation'
            )
            
            try:
                # Use shell=False and list of args for safety and to avoid cmd.exe parsing issues
                cmd = ['powershell', '-NoProfile', '-Command', ps_cmd]
                output = subprocess.check_output(cmd, text=True)
                
                # Clean up empty lines for CSV reader
                output_lines = [line.strip() for line in output.splitlines() if line.strip()]
                if output_lines:
                    reader = csv.DictReader(output_lines)
                    for row in reader:
                        try:
                            pid = int(row.get('ProcessId', row.get('Node', '0')))
                            cmdline = row.get('CommandLine', '')
                            
                            # Filter out the PowerShell query itself if captured
                            if "Get-CimInstance" in cmdline:
                                continue
                                
                            if pid != current_pid and pid != 0:
                                # Exclude the upgrade command itself (e.g. wrapper process)
                                if "upgrade" in cmdline.lower():
                                    continue
                                pids.append((pid, cmdline))
                        except ValueError:
                            continue
            except subprocess.CalledProcessError:
                pass # No processes found or error
            except FileNotFoundError:
                 # Powershell might not be in PATH (unlikely on modern Windows)
                 pass
                
        else:
            # Unix/Linux/Mac implementation using pgrep
            try:
                # pgrep -f -l "fin-agent"
                # -f matches against full command line
                # -l lists PID and process name (command line)
                cmd = ["pgrep", "-f", "-l", "fin-agent"]
                output = subprocess.check_output(cmd).decode('utf-8')
                for line in output.splitlines():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 1:
                        pid = int(parts[0])
                        # If pgrep -l returns name, cmdline might be parts[1] if available
                        cmdline = parts[1] if len(parts) > 1 else "fin-agent"
                        
                        # Exclude build/test processes or the upgrade command itself if pgrep matches broadly
                        # But current_pid check handles the main one.
                        if pid != current_pid:
                            if "upgrade" in cmdline.lower():
                                continue
                            pids.append((pid, cmdline))
            except subprocess.CalledProcessError:
                pass # pgrep returns non-zero if no process found
                
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not check for running processes: {e}{Style.RESET_ALL}")
        # Proceed anyway if we can't check
        return True

    if not pids:
        return True
        
    print(f"\n{Fore.YELLOW}Found the following running fin-agent processes:{Style.RESET_ALL}")
    for pid, cmd in pids:
        # Truncate long command lines
        display_cmd = (cmd[:90] + '...') if len(cmd) > 90 else cmd
        print(f"  PID: {pid:<6} {display_cmd}")
        
    print(f"\n{Fore.RED}These processes should be stopped before upgrading.{Style.RESET_ALL}")
    choice = input(f"Do you want to terminate them now? (y/n): ").strip().lower()
    
    if choice == 'y':
        for pid, _ in pids:
            try:
                print(f"Terminating PID {pid}...")
                if platform.system() == "Windows":
                     subprocess.call(['taskkill', '/F', '/PID', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                     os.kill(pid, signal.SIGTERM)
            except Exception as e:
                print(f"{Fore.RED}Failed to kill PID {pid}: {e}{Style.RESET_ALL}")
        
        # Give them a moment to die
        import time
        time.sleep(1)
        return True
    else:
        print(f"{Fore.RED}Upgrade cancelled by user.{Style.RESET_ALL}")
        return False

def upgrade_package():
    package_name = "fin-agent"
    
    # Check for running instances
    if not check_and_kill_processes():
        return

    try:
        current_v_str = version(package_name)
    except PackageNotFoundError:
        print(f"{Fore.YELLOW}Package '{package_name}' not found (installed in development mode?). Cannot auto-upgrade.{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}Current version: {current_v_str}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Upgrading {package_name} from PyPI...{Style.RESET_ALL}")

    # Store old version to file for the post-upgrade hook to read
    # This is safer than env vars in some environments
    try:
        config_dir = Config.get_config_dir()
        os.makedirs(config_dir, exist_ok=True)
        version_file = os.path.join(config_dir, ".upgrade_old_version")
        with open(version_file, "w") as f:
            f.write(current_v_str)
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not save version info to file: {e}{Style.RESET_ALL}")

    if platform.system() == "Windows":
        # Windows-specific upgrade mechanism:
        # We cannot update the running executable (fin-agent.exe), so we spawn a separate
        # process to wait for this one to exit, then run pip, then run the post-upgrade hook.
        
        print(f"{Fore.YELLOW}Windows detected. Launching separate updater process...{Style.RESET_ALL}")
        
        updater_code = f"""
import os
import sys
import time
import subprocess

print("Waiting for fin-agent process to exit...")
time.sleep(3)

try:
    print("Starting upgrade for {package_name}...")
    # Add --no-cache-dir to avoid installing stale cached versions
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "{package_name}"])
    
    print("Upgrade successful. Running post-upgrade hook...")
    
    # Run post-upgrade hook
    env = os.environ.copy()
    env["FIN_AGENT_POST_UPGRADE"] = "1"
    # Old version is read from file by the hook
    
    # We use -m fin_agent.main to run the potentially new entry point
    subprocess.check_call([sys.executable, "-m", "fin_agent.main"], env=env)
    
    print("\\nUpgrade complete! You can now close this window and run 'fin-agent'.")
    
except subprocess.CalledProcessError as e:
    print(f"Update failed with error code {{e.returncode}}.")
    print("Please try running 'python -m pip install -U fin-agent' manually.")
except Exception as e:
    print(f"An unexpected error occurred: {{e}}")

input("\\nPress Enter to exit...")
"""
        try:
            # Create a temporary file for the updater script
            fd, path = tempfile.mkstemp(suffix=".py", text=True)
            with os.fdopen(fd, 'w') as f:
                f.write(updater_code)
                
            # Spawn the updater in a new console window
            # CREATE_NEW_CONSOLE = 0x00000010
            subprocess.Popen([sys.executable, path], creationflags=0x00000010)
            
            print(f"{Fore.GREEN}Updater launched. This window will now close.{Style.RESET_ALL}")
            sys.exit(0)
            
        except Exception as e:
            print(f"{Fore.RED}Failed to launch updater: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to in-process upgrade (may fail on Windows)...{Style.RESET_ALL}")

    try:
        # Use python -m pip to ensure we're upgrading the package for the current python interpreter
        # Add --no-cache-dir to avoid installing stale cached versions
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", package_name])
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}Upgrade failed. Please check your network connection or permissions.{Style.RESET_ALL}")
        return

    # Call the new version's post-upgrade hook
    # We use sys.executable and -m fin_agent.main to ensure we invoke the module from the same python environment
    print(f"{Fore.GREEN}Package installed successfully.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Invoking post-upgrade hook...{Style.RESET_ALL}")
    
    try:
        # Use subprocess to run the NEW code using env vars to pass state
        env = os.environ.copy()
        env["FIN_AGENT_POST_UPGRADE"] = "1"
        env["FIN_AGENT_OLD_VERSION"] = current_v_str
        
        # We need to run the installed package, not the local file if we are in source dir
        # Using -m fin_agent.main should pick up the installed package if in sys.path
        # But if current dir is in sys.path (which it often is), it might pick up local code.
        # To ensure we run the INSTALLED updated version, we should try to force it or rely on pip install behavior.
        # However, subprocess.check_call([sys.executable, ...]) basically runs a new python.
        
        subprocess.check_call([sys.executable, "-m", "fin_agent.main"], env=env)
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}Post-upgrade hook failed.{Style.RESET_ALL}")


def run_chat_loop(agent):
    print(f"{Fore.GREEN}Agent initialized successfully.{Style.RESET_ALL}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/clear' to start a new conversation.")
    print("Type '/save' to save current session manually.")
    print("Type '/load' to load the last session.")
    
    # Check if a previous session exists and ask? Or just auto-load if arg provided?
    # For now, let's just keep it simple with commands.
    
    console = Console()
    
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print(f"{Fore.YELLOW}Saving session...{Style.RESET_ALL}")
                save_msg = agent.save_session()
                print(f"{Fore.CYAN}{save_msg}{Style.RESET_ALL}")
                print("Goodbye!")
                break
                
            if user_input.lower() == '/clear':
                agent.clear_history()
                print(f"{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}")
                continue
                
            if user_input.lower() == '/save':
                msg = agent.save_session()
                print(f"{Fore.CYAN}{msg}{Style.RESET_ALL}")
                continue
                
            if user_input.lower() == '/load':
                msg = agent.load_session()
                print(f"{Fore.CYAN}{msg}{Style.RESET_ALL}")
                continue
                
            response = agent.run(user_input)
            if response: # Only print if there's a response (might be empty if interrupted)
                print(f"\n{Fore.CYAN}Agent: {Style.RESET_ALL}")
                console.print(FinMarkdown(response))
            
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.YELLOW}Saving session...{Style.RESET_ALL}")
            agent.save_session()
            print("\nGoodbye!")
            # Use os._exit(0) to immediately terminate without triggering atexit handlers
            # which can cause "Exception ignored in atexit callback" stack traces with colorama on Windows
            os._exit(0)
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(
        description="Fin-Agent: A financial analysis AI agent powered by LLMs (DeepSeek/OpenAI) and Tushare data.",
        epilog="Examples:\n  fin-agent                 # Start interactive mode\n  fin-agent --clear-token   # Clear configuration\n  fin-agent --upgrade       # Upgrade to latest version",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version number and exit")
    parser.add_argument("--clear-token", action="store_true", help="Clear the existing configuration token and exit.")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade fin-agent to the latest version.")
    parser.add_argument("--worker", action="store_true", help="Run in worker mode (scheduler only, no chat interface).")
    parser.add_argument("--cycle", type=int, default=10, help="Scheduler interval in minutes (default: 10, only for worker mode).")
    parser.add_argument("--backend-scheduler", action="store_true", help="Run scheduler in background during interactive mode.")
    
    args = parser.parse_args()

    # Check for internal post-upgrade flag via environment variable
    if os.environ.get("FIN_AGENT_POST_UPGRADE"):
        post_upgrade_hook()
        return

    if args.version:
        print(f"fin-agent version {get_version()}")
        sys.exit(0)

    if args.clear_token:
        print(f"{Fore.YELLOW}Clearing configuration...{Style.RESET_ALL}")
        Config.clear()
        print(f"{Fore.GREEN}Configuration cleared successfully.{Style.RESET_ALL}")
        return

    if args.upgrade:
        upgrade_package()
        return

    # Worker Mode
    if args.worker:
        print(f"{Fore.GREEN}Starting Fin-Agent Worker (v{get_version()})...{Style.RESET_ALL}")
        try:
            # We need to load config first
            Config.validate()
        except ValueError:
            print(f"{Fore.RED}Configuration missing. Please run 'fin-agent' first to configure.{Style.RESET_ALL}")
            return

        scheduler = TaskScheduler()
        scheduler.run_forever(cycle=args.cycle)
        return

    print(f"{Fore.GREEN}Welcome to Fin-Agent (v{get_version()})!{Style.RESET_ALL}")
    print("Initializing...")

    agent = None
    try:
        agent = FinAgent()
    except ValueError as e:
        error_msg = str(e)
        if "Missing environment variables" in error_msg:
             print(f"{Fore.YELLOW}Configuration missing or incomplete. Starting setup...{Style.RESET_ALL}")
        else:
             print(f"{Fore.RED}Configuration Error: {error_msg}{Style.RESET_ALL}")
             print(f"{Fore.YELLOW}Running setup...{Style.RESET_ALL}")
        try:
            Config.setup()
            # Retry initialization
            agent = FinAgent()
        except Exception as setup_error:
             print(f"{Fore.RED}Setup failed: {str(setup_error)}{Style.RESET_ALL}")
             return

    if agent:
        # Start Scheduler if requested
        if args.backend_scheduler:
            try:
                scheduler = TaskScheduler()
                scheduler.start()
                print(f"{Fore.GREEN}Background scheduler started.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Failed to start scheduler: {e}{Style.RESET_ALL}")
            
        run_chat_loop(agent)

if __name__ == "__main__":
    main()
