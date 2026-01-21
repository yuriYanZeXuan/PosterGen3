"""
Gemini to OpenAI API Proxy (local)

This file is adapted from `Paper2Slides/gemini_proxy.py`.

It starts a FastAPI server (default port: 51958) that exposes OpenAI-compatible
`/v1/chat/completions` and proxies requests to a Gemini native endpoint.

Supports:
- Chat completions (OpenAI format)
- Multimodal content via `image_url` data-URL (data:image/...;base64,...)
- Tool calling / Function calling (OpenAI <-> Gemini)
"""

import argparse
import json
import time
import requests
import uvicorn
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# Default upstream endpoint (same as Paper2Slides)
DEFAULT_GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"


def convert_openai_tools_to_gemini(tools: List[Dict]) -> List[Dict]:
    """
    OpenAI tools format:
      [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
    Gemini tools format:
      [{"functionDeclarations":[{"name":"...","description":"...","parameters":{...}}]}]
    """
    if not tools:
        return []

    function_declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            func_decl = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                func_decl["parameters"] = func["parameters"]
            function_declarations.append(func_decl)

    if function_declarations:
        return [{"functionDeclarations": function_declarations}]
    return []


def convert_openai_functions_to_gemini(functions: List[Dict]) -> List[Dict]:
    """
    Legacy OpenAI functions format:
      [{"name":"...","description":"...","parameters":{...}}]
    """
    if not functions:
        return []

    function_declarations = []
    for func in functions:
        func_decl = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        if "parameters" in func:
            func_decl["parameters"] = func["parameters"]
        function_declarations.append(func_decl)

    if function_declarations:
        return [{"functionDeclarations": function_declarations}]
    return []


def _load_api_key_from_request_or_default(req: Request) -> str:
    """
    We intentionally do NOT require environment variables.
    Priority:
    - request header `api-key`
    - request header `authorization` (Bearer ...)
    - server default (`app.state.default_api_key`, may be empty)
    """
    header_key = req.headers.get("api-key")
    if header_key:
        return header_key.strip()

    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return (getattr(app.state, "default_api_key", "") or "").strip()


class GeminiClient:
    """Gemini client for the upstream native endpoint."""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def generate_content(
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        functions: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[str, Optional[List[Dict]]]:
        # 1) Convert OpenAI messages -> Gemini contents + systemInstruction
        contents: List[Dict[str, Any]] = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                continue

            # Tool/function result messages
            if role in ("tool", "function"):
                tool_call_id = msg.get("tool_call_id") or msg.get("name", "")
                name = msg.get("name", tool_call_id)
                result_content = content or ""
                try:
                    response_data = json.loads(result_content) if isinstance(result_content, str) else result_content
                except json.JSONDecodeError:
                    response_data = {"result": result_content}

                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": name,
                                    "response": response_data,
                                }
                            }
                        ],
                    }
                )
                continue

            # Assistant messages with tool calls
            if role == "assistant":
                parts: List[Dict[str, Any]] = []

                if content:
                    if isinstance(content, str):
                        parts.append({"text": content})
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text")})

                tool_calls = msg.get("tool_calls", [])
                for i, tc in enumerate(tool_calls):
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                        except json.JSONDecodeError:
                            args_dict = {}
                        parts.append({"functionCall": {"name": func_name, "args": args_dict}})

                function_call = msg.get("function_call")
                if function_call:
                    func_name = function_call.get("name", "")
                    func_args = function_call.get("arguments", "{}")
                    try:
                        args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError:
                        args_dict = {}
                    parts.append({"functionCall": {"name": func_name, "args": args_dict}})

                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            # user/model turns
            gemini_role = "user" if role == "user" else "model"
            parts: List[Dict[str, Any]] = []

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text")})
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:") and "," in url:
                            try:
                                header, b64_data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append(
                                    {
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": b64_data,
                                        }
                                    }
                                )
                            except Exception:
                                # ignore malformed images
                                pass

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # 2) Payload
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.6),
                "maxOutputTokens": kwargs.get("max_tokens", 7000),
                "topP": kwargs.get("top_p", 1),
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # 3) Tools/functions
        gemini_tools = []
        if tools:
            gemini_tools = convert_openai_tools_to_gemini(tools)
        elif functions:
            gemini_tools = convert_openai_functions_to_gemini(functions)
        if gemini_tools:
            payload["tools"] = gemini_tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key

        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")

        res_json = response.json()
        if "error" in res_json:
            raise RuntimeError(f"API Error: {res_json.get('error')}")
        if "candidates" not in res_json or not res_json["candidates"]:
            raise RuntimeError(f"No candidates returned. Full response: {res_json}")

        candidate = res_json["candidates"][0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        tool_calls: List[Dict[str, Any]] = []
        for i, part in enumerate(content_parts):
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                func_name = func_call.get("name", "")
                func_args = func_call.get("args", {})
                tool_calls.append(
                    {
                        "id": f"call_{func_name}_{i}_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(func_args, ensure_ascii=False),
                        },
                    }
                )

        return text_content, tool_calls if tool_calls else None


def generate_stream_response(model: str, text_content: str, tool_calls: Optional[List[Dict]]):
    """SSE stream in OpenAI chunk format."""
    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    first_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    if text_content:
        content_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text_content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tool_chunk)}\n\n"

    finish_reason = "tool_calls" if tool_calls else "stop"
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: Request):
    try:
        data = await req.json()

        model = data.get("model", "gemini-2.5-pro")
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.6)
        max_tokens = data.get("max_tokens", 7000)
        stream = data.get("stream", False)

        tools = data.get("tools", [])
        functions = data.get("functions", [])

        api_key = _load_api_key_from_request_or_default(req)
        endpoint = getattr(app.state, "gemini_endpoint", DEFAULT_GEMINI_ENDPOINT)
        client = GeminiClient(api_key=api_key, endpoint=endpoint)

        import asyncio

        text_content, tool_calls = await asyncio.to_thread(
            client.generate_content,
            model=model,
            messages=messages,
            tools=tools,
            functions=functions,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if stream:
            return StreamingResponse(
                generate_stream_response(model, text_content, tool_calls),
                media_type="text/event-stream",
            )

        message = {"role": "assistant", "content": text_content if text_content else None}
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        resp_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(content=resp_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def main():
    parser = argparse.ArgumentParser(description="Gemini OpenAI-compatible proxy")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=51958)
    parser.add_argument("--endpoint", type=str, default=DEFAULT_GEMINI_ENDPOINT, help="Upstream Gemini endpoint")
    parser.add_argument("--api-key", type=str, default="", help="Default upstream api-key (optional)")
    args = parser.parse_args()

    app.state.gemini_endpoint = args.endpoint
    app.state.default_api_key = args.api_key

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

