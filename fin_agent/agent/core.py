import json
import inspect
import os
from datetime import datetime
from types import SimpleNamespace
from colorama import Fore, Style
from fin_agent.config import Config
from fin_agent.llm.factory import LLMFactory
from fin_agent.tools.tushare_tools import TOOLS_SCHEMA, execute_tool_call
from fin_agent.tools.profile_tools import profile_manager
from fin_agent.utils import FinMarkdown
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

class FinAgent:
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.history = []
        self._init_history()

    def _get_system_content(self):
        # Get user profile summary
        user_profile_summary = profile_manager.get_profile_summary()

        return (
            "You are a financial assistant powered by LLM and Tushare.\\n"
            "You can help users analyze stocks, check prices, and provide recommendations.\\n\\n"
            "### CRITICAL PROTOCOL ###\\n"
            "1. **CHECK TIME FIRST**: For any request involving market data, 'today', 'latest', or time-sensitive info, you MUST first call 'get_current_time' to establish the correct date. This is mandatory.\\n"
            "2. **USE TOOLS**: All market data MUST be obtained via Tushare tools. Do not use internal knowledge.\\n\\n"
            "### TOOL USAGE ###\\n"
            "For 'latest' or 'current' price queries, use 'get_realtime_price'. "
            "For trends and analysis, use 'get_daily_price' to get historical context. "
            "For valuation (PE, PB) or market cap, use 'get_daily_basic'. "
            "For financial performance (Revenue, Profit), use 'get_income_statement'. "
            "For market index (Shanghai Composite, etc.), use 'get_index_daily'. "
            "For technical analysis (MACD, RSI, KDJ, BOLL), use 'get_technical_indicators'. "
            "To check for technical patterns (Golden Cross, Overbought, etc.), use 'get_technical_patterns'. "
            "For funds flow, limit up/down, or concepts, use the corresponding tools. "
            "For portfolio queries (e.g. 'my portfolio', '我的持仓', 'holdings'), ALWAYS use 'get_portfolio_status'. "
            "For portfolio management (add/remove position), use 'add_portfolio_position' or 'remove_portfolio_position'. "
            "When setting price alerts with percentages (e.g. 'alert if rises 5%'), you MUST first fetch the current price (or relevant base price), calculate the target absolute price, and then set the alert with that absolute value. "
            "For configuration (email, tushare token, llm settings), use 'reset_email_config' or 'reset_core_config'. These tools are INTERACTIVE, so simply call them when requested; do NOT ask the user for details in the chat."
            "When analyzing, EXPLICITLY mention the date of the data you are using. "
            "Calculate percentage changes and describe the trend (e.g., upward, downward, volatile) based on the data. "
            "When you have enough information, answer the user's question directly.\\n\\n"
            "### USER CONTEXT & MEMORY ###\\n"
            "You have access to a long-term memory of the user's investment preferences. "
            "Use the 'update_user_profile' tool to SAVE new preferences when the user explicitly states them or when you infer them (e.g., 'I prefer low risk', 'I only buy tech stocks'). "
            "The current user profile is:\\n"
            f"{user_profile_summary}\\n\\n"
            "Tailor your responses and recommendations based on this profile. "
            "If the user asks for recommendations without specifying criteria, refer to their profile (e.g. 'Based on your preference for low risk...')."
        )

    def _init_history(self):
        """Initialize history with system prompt."""
        self.history = [
            {"role": "system", "content": self._get_system_content()}
        ]

    def _to_dict(self, message):
        """Helper to convert message object to dictionary."""
        if isinstance(message, dict):
            return message
        if hasattr(message, 'model_dump'):
            return message.model_dump()
        if hasattr(message, 'to_dict'):
            return message.to_dict()
        # Fallback for SimpleNamespace or other objects
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None)
        }

    def save_session(self, filename="last_session.json"):
        """Save current session history to file."""
        config_dir = Config.get_config_dir()
        filepath = os.path.join(config_dir, "sessions", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            return f"Session saved to {filepath}"
        except Exception as e:
            return f"Error saving session: {e}"

    def load_session(self, filename="last_session.json"):
        """Load session history from file."""
        config_dir = Config.get_config_dir()
        filepath = os.path.join(config_dir, "sessions", filename)
        
        if not os.path.exists(filepath):
            return "No saved session found."
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            # Ensure we update the system prompt part of the loaded history to reflect latest code/profile?
            # Or trust the saved one? Usually, we want the LATEST profile in system prompt.
            # Let's update the first message if it is 'system'
            if self.history and self.history[0].get('role') == 'system':
                self.history[0]['content'] = self._get_system_content()
                
            return f"Session loaded from {filepath}"
        except Exception as e:
            return f"Error loading session: {e}"

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        self._init_history()

    def run(self, user_input):
        """
        Run the agent with user input.
        """
        # Update system prompt to ensure latest profile is used
        if self.history and self.history[0].get('role') == 'system':
             self.history[0]['content'] = self._get_system_content()
        else:
             # Should not happen if initialized correctly, but safety check
             self.history.insert(0, {"role": "system", "content": self._get_system_content()})

        # Append user input
        self.history.append({"role": "user", "content": user_input})

        # Initialize console for rich output
        console = Console()

        step = 0
        try:
            while True:
                step += 1
                
                # Call LLM
                try:
                    # Determine stream mode from Config
                    stream_mode = Config.LLM_STREAM
                    
                    response = self.llm.chat(self.history, tools=TOOLS_SCHEMA, tool_choice="auto", stream=stream_mode)
                    
                    message = None
                    
                    if stream_mode and inspect.isgenerator(response):
                        # Handle Streaming Response
                        print(f"{Fore.CYAN}Agent: {Style.RESET_ALL}")
                        
                        full_content = ""
                        stream_interrupted = False
                        
                        # Buffer for handling <think> tags
                        buffer = ""
                        thinking_state = False
                        
                        # State for Markdown Streaming
                        md_buffer = ""
                        in_code_block = False
                        
                        # Live component for the *current, incomplete* chunk
                        # We use vertical_overflow="visible" but since we clear it often, it won't grow infinitely.
                        live_md = Live(FinMarkdown(""), console=console, auto_refresh=True, refresh_per_second=10, vertical_overflow="visible")
                        live_md.start()

                        def stop_md():
                            if live_md:
                                live_md.stop()

                        def update_md(text):
                            nonlocal md_buffer, in_code_block
                            
                            # Append new text
                            md_buffer += text
                            
                            # We want to "commit" (print permanently) any complete blocks to avoid Live scrolling issues.
                            # A complete block is defined as text ending with a newline, UNLESS we are in a code block.
                            
                            # Naive check for code block toggle
                            # We check the *newly added* text for fences, or scan the buffer?
                            # Safest is to scan the buffer line by line if we are not in code block.
                            
                            if not in_code_block:
                                # Split by newline to find commit points
                                if '\n' in md_buffer:
                                    lines = md_buffer.split('\n')
                                    
                                    # We keep the last line as it might be incomplete
                                    # Unless the text ends with \n, then even the last line is complete?
                                    # split('a\n', '\n') -> ['a', '']
                                    
                                    # Process all complete lines
                                    # If md_buffer ends with \n, all lines in split result (except last empty) are complete.
                                    # If md_buffer does NOT end with \n, the last element is incomplete.
                                    
                                    complete_lines = lines[:-1]
                                    remainder = lines[-1]
                                    
                                    # If we have complete lines, check if they start a code block
                                    for i, line in enumerate(complete_lines):
                                        # Check for code fence
                                        # We assume a fence is "```" at start of line (ignoring whitespace)
                                        if line.strip().startswith("```"):
                                            # We found a code block start!
                                            # We cannot commit this line alone, and we are now in code block mode.
                                            # We must stop committing lines and keep the rest in buffer.
                                            
                                            # Commit everything BEFORE this line
                                            to_commit = "\n".join(complete_lines[:i])
                                            if to_commit:
                                                # Clear Live temporarily to print above
                                                live_md.update(FinMarkdown(""))
                                                console.print(FinMarkdown(to_commit))
                                            
                                            # The rest (from fence onwards) stays in buffer
                                            # We rejoin the rest of complete_lines + remainder
                                            md_buffer = "\n".join(complete_lines[i:] + [remainder])
                                            in_code_block = True
                                            break
                                    else:
                                        # No code block start found in complete lines
                                        # Commit all complete lines
                                        to_commit = "\n".join(complete_lines)
                                        if to_commit:
                                            live_md.update(FinMarkdown(""))
                                            console.print(FinMarkdown(to_commit))
                                        
                                        md_buffer = remainder

                            else:
                                # We are IN a code block. We wait for it to close.
                                # Check if we have the closing fence in the buffer.
                                if "```" in text or "```" in md_buffer: 
                                    # Naive check: does the buffer contain a line starting with ``` that closes it?
                                    # We need to be careful not to match the opening fence if we just added it.
                                    # But we only enter this block if we already processed the opening fence (it's in md_buffer).
                                    
                                    # Let's verify line by line
                                    lines = md_buffer.split('\n')
                                    # We skip the first line if it's the opening fence? 
                                    # Actually, just look for a closing fence.
                                    
                                    # Find closing fence index
                                    closing_index = -1
                                    
                                    # We need to skip the opening fence. 
                                    # Since we just entered in_code_block, the opening fence is at the start of md_buffer.
                                    # But wait, md_buffer accumulates.
                                    
                                    for i, line in enumerate(lines):
                                        if i == 0 and line.strip().startswith("```"):
                                            continue # Skip opening
                                        
                                        if line.strip().startswith("```"):
                                            closing_index = i
                                            break
                                    
                                    if closing_index != -1:
                                        # Found closing fence!
                                        # Commit the whole block
                                        to_commit = "\n".join(lines[:closing_index+1])
                                        remainder = "\n".join(lines[closing_index+1:])
                                        
                                        live_md.update(FinMarkdown(""))
                                        console.print(FinMarkdown(to_commit))
                                        
                                        md_buffer = remainder
                                        in_code_block = False

                            # Update Live with whatever is left in buffer
                            # Note: FinMarkdown might look partial, but that's what we want for streaming
                            live_md.update(FinMarkdown(md_buffer))

                        try:
                            for chunk in response:
                                if chunk['type'] == 'content':
                                    content = chunk['content']
                                    full_content += content # Store full raw content
                                    
                                    buffer += content
                                    
                                    while True:
                                        if not thinking_state:
                                            if "<think>" in buffer:
                                                # Split content before <think>
                                                pre, buffer = buffer.split("<think>", 1)
                                                if pre:
                                                    update_md(pre)
                                                
                                                # Stop markdown before switching to thinking
                                                stop_md()

                                                # Switch to thinking style
                                                print(f"{Style.DIM}{Fore.YELLOW}", end="", flush=True)
                                                thinking_state = True
                                            else:
                                                # Optimization: avoid printing potential partial tag
                                                # <think> length is 7
                                                if len(buffer) < 7:
                                                    break
                                                
                                                # Print safe part
                                                to_print = buffer[:-6]
                                                buffer = buffer[-6:]
                                                update_md(to_print)
                                                break
                                        
                                        else: # thinking_state is True
                                            if "</think>" in buffer:
                                                # Split content before </think>
                                                pre, buffer = buffer.split("</think>", 1)
                                                if pre:
                                                    print(pre, end="", flush=True)
                                                
                                                # Switch back to normal style
                                                print(Style.RESET_ALL, end="", flush=True)
                                                thinking_state = False
                                                
                                                # Restart Live Markdown
                                                live_md = Live(FinMarkdown(""), console=console, auto_refresh=True, refresh_per_second=10, vertical_overflow="visible")
                                                live_md.start()
                                                
                                                # Consume potential newline after </think> if it starts the remaining buffer
                                                if buffer.startswith("\n"):
                                                    buffer = buffer[1:]
                                                elif buffer.startswith("\r\n"): # Handle Windows newline
                                                    buffer = buffer[2:]
                                            else:
                                                # </think> length is 8
                                                if len(buffer) < 8:
                                                    break
                                                
                                                to_print = buffer[:-7]
                                                buffer = buffer[-7:]
                                                print(to_print, end="", flush=True)
                                                break
                                    
                                elif chunk['type'] == 'response':
                                    message = chunk['response']
                            
                            # Print remaining buffer
                            if buffer:
                                if thinking_state:
                                    print(buffer, end="", flush=True)
                                    print(Style.RESET_ALL, end="", flush=True)
                                else:
                                    update_md(buffer)
                            
                            # Final flush of md_buffer
                            if md_buffer:
                                live_md.update(FinMarkdown("")) # Clear Live
                                console.print(FinMarkdown(md_buffer)) # Print Final
                            
                            stop_md()
                            
                            # Ensure reset if still thinking (unlikely for well-formed output)
                            if thinking_state:
                                print(Style.RESET_ALL, end="", flush=True)
                                
                        except KeyboardInterrupt:
                            stream_interrupted = True
                            stop_md()
                            print(f"{Style.RESET_ALL}\n{Fore.YELLOW}[Output interrupted]{Style.RESET_ALL}")
                        
                        if stream_interrupted:
                             # Save partial content if any
                             if full_content:
                                 message = SimpleNamespace(role="assistant", content=full_content, tool_calls=None)
                                 self.history.append(message)
                             return ""

                        # If we printed content, add a newline at the end
                        # if full_content and not thinking_state and not full_content.endswith("\n"):
                        #    print() 
 
                            
                    else:
                        # Handle Normal Response
                        message = response

                except Exception as e:
                    return f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}"

                # If no tool calls, this is the final answer
                if not message.tool_calls:
                    answer = message.content
                    self.history.append(self._to_dict(message)) # Keep history
                    
                    if stream_mode:
                        return "" # Already printed
                    else:
                        return answer

                # Handle tool calls
                self.history.append(self._to_dict(message)) # Add assistant's message with tool_calls to history

                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    call_id = tool_call.id
                    
                    print(f"{Fore.CYAN}Calling Tool: {function_name} with args: {arguments}{Style.RESET_ALL}")
                    
                    # Execute tool
                    tool_result = execute_tool_call(function_name, arguments)
                    
                    # Truncate result if too long for display, but keep full for LLM (context window permitting)
                    display_result = tool_result[:200] + "..." if len(str(tool_result)) > 200 else tool_result
                    print(f"{Fore.BLUE}Tool Result: {display_result}{Style.RESET_ALL}")

                    # Append tool result to history
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(tool_result)
                    })

        except KeyboardInterrupt:
            # Catch global interruptions (e.g. during tool execution or thinking)
            print(f"\n{Fore.YELLOW}[Interrupted by user]{Style.RESET_ALL}")
            return ""
