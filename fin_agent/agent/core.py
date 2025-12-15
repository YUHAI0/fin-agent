import json
import inspect
from types import SimpleNamespace
from colorama import Fore, Style
from fin_agent.config import Config
from fin_agent.llm.factory import LLMFactory
from fin_agent.tools.tushare_tools import TOOLS_SCHEMA, execute_tool_call
from fin_agent.utils import FinMarkdown
from rich.console import Console
from rich.live import Live

class FinAgent:
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.history = []

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

    def run(self, user_input):
        """
        Run the agent with user input.
        """
        # Initialize conversation with user input
        self.history = [
            {"role": "system", "content": "You are a financial assistant powered by LLM and Tushare. "
                                          "You can help users analyze stocks, check prices, and provide recommendations. "
                                          "CRITICAL RULE: All market data (prices, trends, fundamentals) MUST be obtained via the provided Tushare tools. "
                                          "DO NOT use your internal knowledge for any market data, as it may be outdated. "
                                          "Always fetch the latest data using the tools before answering. "
                                          "First step: Check the current time using 'get_current_time' to establish the temporal context if the user asks about time-sensitive data (e.g. 'today', 'recent'). "
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
                                          "When you have enough information, answer the user's question directly."},
            {"role": "user", "content": user_input}
        ]

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
                        
                        # Markdown Live State
                        live_md = None
                        md_buffer = ""

                        def stop_md():
                            nonlocal live_md, md_buffer
                            if live_md:
                                live_md.stop()
                                live_md = None
                                md_buffer = ""

                        def update_md(text):
                            nonlocal live_md, md_buffer
                            md_buffer += text
                            if live_md is None:
                                # refresh_per_second controls how often the screen is updated
                                # Lowering it slightly might help with flickering, but keeping it smooth
                                live_md = Live(FinMarkdown(md_buffer), auto_refresh=True, refresh_per_second=4, vertical_overflow="visible")
                                live_md.start()
                            else:
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
                        if full_content and not thinking_state and not full_content.endswith("\n"):
                            print() 
 
                            
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
                        return "" # Already printed (via Console().print above) or we return it? 
                        # The original code returned answer if not stream_mode.
                        # But main.py likely prints the return value.
                        # Let's check main.py usage.
                        # Original: return answer
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
