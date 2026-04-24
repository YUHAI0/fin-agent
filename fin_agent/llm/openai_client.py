from openai import OpenAI
from fin_agent.llm.base import LLMBase


def _split_concatenated_json_tool_args(arg_str: str):
    """
    部分 OpenAI 兼容接口在并行工具调用时会把多个 JSON 对象拼成一串（如 {...}{...}），
    若仍只生成一条 tool_call，会导致后端只执行一次、只发一条 tool_result。
    按最外层花括号切分为多条参数字符串，供拆成多条 ToolCall。
    """
    s = (arg_str or "").strip()
    if not s:
        return []
    depth = 0
    start = None
    chunks = []
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    chunks.append(s[start : i + 1].strip())
                    start = None
    if not chunks:
        return [s]
    return chunks


class OpenAICompatibleClient(LLMBase):
    def __init__(self, api_key, base_url, model, default_headers=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers
        )
        self.model = model

    def chat(self, messages, tools=None, tool_choice=None, stream=False):
        """
        Send a chat completion request to an OpenAI-compatible API.
        """
        # Ensure messages are serializable
        sanitized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                sanitized_messages.append(msg)
            elif hasattr(msg, 'model_dump'): # Pydantic/OpenAI v1 objects
                # Check if model_dump returns dict directly or needs arguments
                try:
                    sanitized_messages.append(msg.model_dump())
                except:
                    # Fallback for some objects that might not work with empty model_dump
                    sanitized_messages.append(msg)
            elif hasattr(msg, 'to_dict'): # Our custom objects or legacy objects
                sanitized_messages.append(msg.to_dict())
            else:
                # Fallback for OpenAI objects from library if they don't have model_dump exposed directly
                # or if it's a simple object. 
                # Try to use __dict__ if available, but be careful
                try:
                    # For OpenAI ChatCompletionMessage, it might be safer to let the library handle it
                    # IF the library supports it. But our custom Message class definitely needs conversion.
                    # If we are here, it's an unknown object.
                    sanitized_messages.append(msg) 
                except:
                    sanitized_messages.append(msg)

        params = {
            "model": self.model,
            "messages": sanitized_messages,
            "stream": stream
        }
        
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        try:
            response = self.client.chat.completions.create(**params)
            
            if stream:
                return self._handle_stream(response)
            else:
                return response.choices[0].message
                
        except Exception as e:
            # print(f"Error calling LLM API: {e}") # Let caller handle logging
            raise e
            
    def _handle_stream(self, response_stream):
        """
        Handle streaming response from OpenAI.
        Yields content chunks for display, and finally returns a constructed message object.
        """
        import sys
        collected_content = []
        collected_tool_calls = {} # index -> tool_call_data
        
        # Simulated Message Object structure to match OpenAI's object for compatibility
        class Message:
            def __init__(self, role, content, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
            
            def model_dump(self):
                return {
                    "role": self.role,
                    "content": self.content,
                    "tool_calls": [tc.model_dump() for tc in self.tool_calls] if self.tool_calls else None
                }
            
            def to_dict(self):
                return self.model_dump()

        class ToolCall:
            def __init__(self, id, type, function):
                self.id = id
                self.type = type
                self.function = function
            
            def model_dump(self):
                return {
                    "id": self.id,
                    "type": self.type,
                    "function": self.function.model_dump()
                }

        class Function:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments
            
            def model_dump(self):
                return {
                    "name": self.name,
                    "arguments": self.arguments
                }

        try:
            chunk_count = 0
            for chunk in response_stream:
                chunk_count += 1
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Debug: log what we receive
                has_content = hasattr(delta, 'content') and delta.content
                has_tool_calls = hasattr(delta, 'tool_calls') and delta.tool_calls
                # if chunk_count <= 3 or has_content or has_tool_calls:
                #     from fin_agent.utils import debug_print
                #     debug_print(f"[Stream] Chunk #{chunk_count}: has_content={has_content}, has_tool_calls={has_tool_calls}", file=sys.stderr)
                
                # Handle Content
                if delta.content:
                    content_chunk = delta.content
                    collected_content.append(content_chunk)
                    yield {"type": "content", "content": content_chunk}
                
                # Handle Tool Calls (streaming)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        index = tc.index
                        
                        # Yield a tool_call_chunk to allow real-time display
                        yield {
                            "type": "tool_call_chunk",
                            "index": index,
                            "id": tc.id if tc.id else None,
                            "name": tc.function.name if tc.function and tc.function.name else None,
                            "arguments": tc.function.arguments if tc.function and tc.function.arguments else None
                        }

                        if index not in collected_tool_calls:
                            collected_tool_calls[index] = {
                                "id": "", "type": "function", "name": "", "arguments": ""
                            }
                        
                        if tc.id:
                            collected_tool_calls[index]["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[index]["name"] += tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[index]["arguments"] += tc.function.arguments

            # Construct final message
            final_content = "".join(collected_content) if collected_content else None
            
            final_tool_calls = []
            if collected_tool_calls:
                # Sort by index；兼容 index 为 None（避免 int 与 None 比较报错）
                def _tool_call_sort_key(idx):
                    if idx is None:
                        return (1, 0)
                    if isinstance(idx, int):
                        return (0, idx)
                    try:
                        return (0, int(idx))
                    except (TypeError, ValueError):
                        return (1, 0)

                sorted_indices = sorted(collected_tool_calls.keys(), key=_tool_call_sort_key)
                for index in sorted_indices:
                    tc_data = collected_tool_calls[index]
                    arg_str = tc_data.get("arguments") or ""
                    name = (tc_data.get("name") or "").strip()
                    base_id = (tc_data.get("id") or "").strip() or f"call_{index}"
                    parts = _split_concatenated_json_tool_args(arg_str)
                    if len(parts) <= 1:
                        final_tool_calls.append(
                            ToolCall(
                                id=base_id,
                                type="function",
                                function=Function(
                                    name=name,
                                    arguments=arg_str,
                                ),
                            )
                        )
                    else:
                        # 少数网关会把函数名也重复拼接，与参数段数对齐时还原为单段
                        eff_name = name
                        if eff_name and len(eff_name) % len(parts) == 0:
                            step = len(eff_name) // len(parts)
                            seg = eff_name[:step]
                            if seg * len(parts) == eff_name:
                                eff_name = seg
                        for j, part in enumerate(parts):
                            tid = f"{base_id}:{j}" if base_id else f"call_{index}_{j}"
                            final_tool_calls.append(
                                ToolCall(
                                    id=tid,
                                    type="function",
                                    function=Function(
                                        name=eff_name,
                                        arguments=part,
                                    ),
                                )
                            )
            
            final_message = Message(
                role="assistant",
                content=final_content,
                tool_calls=final_tool_calls if final_tool_calls else None
            )
            
            # Yield the final message object
            yield {"type": "response", "response": final_message}

        except Exception as e:
            import traceback
            from fin_agent.utils import debug_print
            debug_print(f"Exception in _handle_stream: {traceback.format_exc()}", file=sys.stderr)
            raise e
