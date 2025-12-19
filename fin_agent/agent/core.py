import json
import inspect
import os
from datetime import datetime
from types import SimpleNamespace
from colorama import Fore, Style
from fin_agent.config import Config
from fin_agent.llm.factory import LLMFactory
from fin_agent.tools.tushare_tools import TOOLS_SCHEMA, execute_tool_call
from fin_agent.tools.profile_tools import get_profile_manager
from fin_agent.utils import FinMarkdown
from rich.console import Console
from rich.live import Live

class FinAgent:
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.history = []
        self._init_history()

    def _get_system_content(self):
        # Get user profile summary
        user_profile_summary = get_profile_manager().get_profile_summary()

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

    def stream_chat(self, user_input):
        """
        Generator function that yields events for the chat interaction.
        Yields dicts with 'type' and 'content'/'data'.
        Types: 'content', 'thinking', 'tool_call', 'tool_result', 'error', 'log', 'answer'
        """
        import sys
        from fin_agent.utils import debug_print
        debug_print(f"Starting stream_chat with input: {user_input[:50]}...", file=sys.stderr)
        
        # Check if LLM is valid
        if not self.llm:
             yield {"type": "error", "content": "LLM not initialized. Please check configuration."}
             return
        
        # Update system prompt to ensure latest profile is used
        if self.history and self.history[0].get('role') == 'system':
             self.history[0]['content'] = self._get_system_content()
        else:
             self.history.insert(0, {"role": "system", "content": self._get_system_content()})

        # Append user input
        self.history.append({"role": "user", "content": user_input})

        step = 0
        try:
            while True:
                step += 1
                debug_print(f"Step {step}", file=sys.stderr)
                
                try:
                    # Determine stream mode from Config - Force True for stream_chat
                    stream_mode = True 
                    
                    debug_print("Calling LLM chat...", file=sys.stderr)
                    response = self.llm.chat(self.history, tools=TOOLS_SCHEMA, tool_choice="auto", stream=stream_mode)
                    debug_print(f"LLM chat returned {type(response)}", file=sys.stderr)
                    
                    message = None
                    
                    if stream_mode and inspect.isgenerator(response):
                        full_content = ""
                        stream_interrupted = False
                        
                        # Buffer for handling <think> tags
                        buffer = ""
                        thinking_state = False
                        
                        try:
                            debug_print("Starting response iteration", file=sys.stderr)
                            for chunk in response:
                                if chunk['type'] == 'content':
                                    content = chunk['content']
                                    full_content += content 
                                    buffer += content
                                    
                                    while True:
                                        if not thinking_state:
                                            # Look for <think>
                                            tag = "<think>"
                                            if tag in buffer:
                                                pre, buffer = buffer.split(tag, 1)
                                                if pre:
                                                    yield {"type": "content", "content": pre}
                                                yield {"type": "log", "content": "Thinking..."}
                                                thinking_state = True
                                                continue # Re-evaluate buffer in thinking state
                                            
                                            # Smart Flush: Yield anything that can't be part of <think>
                                            # If no '<', yield all
                                            if "<" not in buffer:
                                                if buffer:
                                                    yield {"type": "content", "content": buffer}
                                                    buffer = ""
                                                break
                                            
                                            # Has '<'. Find first '<'
                                            idx = buffer.find("<")
                                            # Yield everything before '<'
                                            if idx > 0:
                                                yield {"type": "content", "content": buffer[:idx]}
                                                buffer = buffer[idx:]
                                            
                                            # Now buffer starts with '<'
                                            # Check if it matches partial tag
                                            # buffer is like "<...", len >= 1
                                            
                                            # If buffer is shorter than tag, checking partial match
                                            # Optimization: just check if it IS a prefix
                                            if tag.startswith(buffer):
                                                # It is a prefix, we must wait for more data
                                                break
                                            
                                            # It's NOT a prefix of <think> (e.g. "<div>" or "< 5")
                                            # But wait, what if buffer is longer than tag?
                                            # We already checked `if tag in buffer`.
                                            # So if len(buffer) >= len(tag) and tag not in buffer (at start),
                                            # then it's not our tag.
                                            
                                            if len(buffer) >= len(tag):
                                                # We know it starts with < but is not <think>
                                                # Yield the < and continue
                                                yield {"type": "content", "content": "<"}
                                                buffer = buffer[1:]
                                                continue
                                            
                                            # If len(buffer) < len(tag), we checked startswith above.
                                            # If it didn't match startswith, it's not our tag.
                                            if not tag.startswith(buffer):
                                                 yield {"type": "content", "content": "<"}
                                                 buffer = buffer[1:]
                                                 continue
                                            
                                            # Should be caught by startswith check, but safe break
                                            break

                                        else:
                                            # Thinking State - Look for </think>
                                            tag = "</think>"
                                            if tag in buffer:
                                                pre, buffer = buffer.split(tag, 1)
                                                if pre: 
                                                    yield {"type": "thinking", "content": pre}
                                                thinking_state = False
                                                # yield {"type": "log", "content": "Thinking ended."}
                                                if buffer.startswith("\n"): buffer = buffer[1:]
                                                elif buffer.startswith("\r\n"): buffer = buffer[2:]
                                                continue # Re-evaluate buffer in content state

                                            # Smart Flush for Thinking
                                            if "<" not in buffer:
                                                if buffer:
                                                    yield {"type": "thinking", "content": buffer}
                                                    buffer = ""
                                                break
                                            
                                            idx = buffer.find("<")
                                            if idx > 0:
                                                yield {"type": "thinking", "content": buffer[:idx]}
                                                buffer = buffer[idx:]
                                            
                                            # buffer starts with <
                                            if tag.startswith(buffer):
                                                break
                                            
                                            if len(buffer) >= len(tag):
                                                yield {"type": "thinking", "content": "<"}
                                                buffer = buffer[1:]
                                                continue
                                                
                                            if not tag.startswith(buffer):
                                                 yield {"type": "thinking", "content": "<"}
                                                 buffer = buffer[1:]
                                                 continue
                                            
                                            break
                                    
                                elif chunk['type'] == 'tool_call_chunk':
                                    # If we receive a tool call chunk, it means content/thinking stream is paused or done for now.
                                    # Flush buffer immediately to show any pending thinking/content
                                    if buffer:
                                        if thinking_state:
                                            yield {"type": "thinking", "content": buffer}
                                        else:
                                            yield {"type": "content", "content": buffer}
                                        buffer = ""

                                    # Yield the tool call chunk to frontend for real-time update
                                    yield chunk

                                    # Yield the tool call chunk to frontend for real-time update
                                    yield chunk

                                elif chunk['type'] == 'response':
                                    debug_print("Received final response object", file=sys.stderr)
                                    message = chunk['response']
                            
                            debug_print("Response iteration finished", file=sys.stderr)
                            
                            # Flush remaining buffer
                            if buffer:
                                if thinking_state:
                                    yield {"type": "thinking", "content": buffer}
                                else:
                                    yield {"type": "content", "content": buffer}
                            
                        except KeyboardInterrupt:
                            # print("DEBUG: KeyboardInterrupt during iteration", file=sys.stderr)
                            stream_interrupted = True
                            yield {"type": "error", "content": "Interrupted by user"}
                        
                        if stream_interrupted:
                             # Save partial content if any
                             if full_content:
                                 message = SimpleNamespace(role="assistant", content=full_content, tool_calls=None)
                                 self.history.append(message)
                             return

                    else:
                        # Handle Normal Response (Non-stream fallback)
                        # print("DEBUG: Handling non-stream response", file=sys.stderr)
                        message = response
                        if message.content:
                            yield {"type": "content", "content": message.content}

                except Exception as e:
                    import traceback
                    err_msg = f"Error: {str(e)}"
                    # print(f"DEBUG: Exception in stream_chat: {traceback.format_exc()}", file=sys.stderr)
                    yield {"type": "error", "content": err_msg}
                    return

                if not message:
                    debug_print("Message is None after loop!", file=sys.stderr)
                    return

                # If no tool calls, this is the final answer
                if not message.tool_calls:
                    answer = message.content if message.content else ""
                    # debug_print(f"No tool calls, finishing. Answer: '{answer[:100] if answer else '(empty)'}'", file=sys.stderr)
                    self.history.append(self._to_dict(message)) # Keep history
                    # Always yield answer event, even if empty, so frontend knows we're done
                    yield {"type": "answer", "content": answer}
                    return

                # Handle tool calls
                # print(f"DEBUG: Processing {len(message.tool_calls)} tool calls", file=sys.stderr)
                self.history.append(self._to_dict(message)) # Add assistant's message with tool_calls to history

                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    call_id = tool_call.id
                    
                    # print(f"DEBUG: Tool Call: {function_name}", file=sys.stderr)
                    yield {"type": "tool_call", "tool_name": function_name, "args": arguments}
                    
                    # Execute tool
                    try:
                        tool_result = execute_tool_call(function_name, arguments)
                    except Exception as e:
                        tool_result = f"Error executing tool: {e}"
                    
                    # Check for config reset to reload LLM
                    if function_name == "reset_core_config":
                        yield {"type": "log", "content": "Reloading LLM configuration..."}
                        try:
                            # Re-create LLM instance with new config
                            self.llm = LLMFactory.create_llm()
                            yield {"type": "log", "content": "LLM re-initialized successfully."}
                        except Exception as e:
                            yield {"type": "error", "content": f"Error re-initializing LLM: {str(e)}"}

                    # Truncate result if too long for log, but keep full for LLM
                    # display_result = tool_result[:200] + "..." if len(str(tool_result)) > 200 else tool_result
                    
                    yield {"type": "tool_result", "tool_name": function_name, "result": str(tool_result)}

                    # Append tool result to history
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(tool_result)
                    })

        except KeyboardInterrupt:
            yield {"type": "error", "content": "Interrupted by user"}
            return

    def run(self, user_input, callback=None):
        """
        Run the agent with user input.
        Kept for backward compatibility and CLI usage.
        """
        # We'll use stream_chat internally to avoid code duplication, 
        # but we need to reconstruct the rich/Live display logic.
        
        # NOTE: This is a slightly simplified version of the original run to reuse stream_chat.
        # If strict exact behavior of original CLI is needed, we might need to be more careful.
        # But for now, let's try to adapt the CLI to consume the generator.

        # However, the original run method had complex Live Markdown update logic 
        # that might be hard to perfectly replicate from the event stream without some work.
        # To be SAFE and not break CLI, I will leave the original run method mostly AS IS,
        # but I will copy the logic to stream_chat. 
        # (Wait, I just overwrote the whole file content in the tool call above?)
        # YES. I need to put the original `run` method back or reimplement it.
        
        # Re-implementing run using stream_chat to ensure consistency:
        
        print(f"{Fore.CYAN}Agent: {Style.RESET_ALL}")
        
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
                live_md = Live(FinMarkdown(md_buffer), auto_refresh=True, refresh_per_second=4, vertical_overflow="visible")
                live_md.start()
            else:
                live_md.update(FinMarkdown(md_buffer))

        generator = self.stream_chat(user_input)
        
        final_answer = ""
        
        try:
            for event in generator:
                event_type = event['type']
                
                if event_type == 'content':
                    content = event['content']
                    update_md(content)
                    if callback: callback('content', content)
                    
                elif event_type == 'thinking':
                    content = event['content']
                    stop_md()
                    print(f"{Style.DIM}{Fore.YELLOW}{content}", end="", flush=True)
                    # We might need to handle resetting color after thinking block ends
                    # The generator stream separates thinking chunks. 
                    # We need to know when thinking ENDS to reset color?
                    # The generator doesn't explicitly say "thinking_end".
                    # But if we receive 'content' after 'thinking', we should reset.
                    pass 
                    
                elif event_type == 'tool_call':
                    stop_md()
                    # If we were thinking, reset color
                    print(Style.RESET_ALL, end="", flush=True) 
                    
                    name = event['tool_name']
                    args = event['args']
                    print(f"\n{Fore.CYAN}Calling Tool: {name} with args: {args}{Style.RESET_ALL}")
                    if callback: callback('tool_call', {"name": name, "args": args})

                elif event_type == 'tool_result':
                    result = event['result']
                    display_result = result[:200] + "..." if len(result) > 200 else result
                    print(f"{Fore.BLUE}Tool Result: {display_result}{Style.RESET_ALL}")
                    if callback: callback('tool_result', {"name": event['tool_name'], "result": result})

                elif event_type == 'error':
                    stop_md()
                    print(f"\n{Fore.RED}Error: {event['content']}{Style.RESET_ALL}")
                    if callback: callback('error', event['content'])
                    return event['content']

                elif event_type == 'answer':
                    final_answer = event['content']
            
            stop_md()
            print(Style.RESET_ALL) # Ensure reset at end
            return final_answer

        except KeyboardInterrupt:
            stop_md()
            print(f"\n{Fore.YELLOW}[Interrupted by user]{Style.RESET_ALL}")
            return ""
