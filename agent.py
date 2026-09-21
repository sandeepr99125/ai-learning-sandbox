import os
import json
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import streamlit as st

load_dotenv()

# ==========================================
# 1. TOOL DEFINITIONS
# ==========================================

def python_syntax_checker(code: str) -> str:
    """Tool 1: Verifies Python code for syntax errors."""
    try:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        compile(clean_code, "<string>", "exec")
        return "SUCCESS: Code syntax is completely valid."
    except Exception as e:
        return f"SYNTAX ERROR: {e}"

def generate_quiz_question(topic: str) -> str:
    """Tool 2: Generates a self-assessment question."""
    return f"Self-Check: Can you explain how '{topic}' balances trade-offs in machine learning evaluation?"

AVAILABLE_TOOLS = {
    "python_syntax_checker": python_syntax_checker,
    "generate_quiz_question": generate_quiz_question
}

# ==========================================
# 2. AUTONOMOUS AGENT CLASS
# ==========================================

class AITutorAutonomousAgent:
    def __init__(self):
        # Read from Streamlit Secrets (Cloud) first, fallback to Environment Variables (Local)
        self.gemini_key = getattr(st, "secrets", {}).get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        self.openrouter_key = getattr(st, "secrets", {}).get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
        
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_key
        ) if self.openrouter_key else None

        self.fallback_models = [
            "openrouter/free",
            "nvidia/nemotron-3-ultra:free",
            "google/gemma-4-31b-it:free",
            "cohere/north-mini-code:free"
        ]

    def _call_llm(self, messages: list) -> tuple[str, str]:
        """Resilient API caller with failover."""
        if self.gemini_client:
            try:
                formatted_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=formatted_prompt
                )
                return response.text, "Primary: Gemini 2.5 Flash Lite"
            except Exception as e:
                print(f"[Fallback Warning] Gemini failed: {e}. Switching to OpenRouter...")

        if self.openrouter_client:
            for model_id in self.fallback_models:
                try:
                    res = self.openrouter_client.chat.completions.create(
                        model=model_id,
                        messages=messages
                    )
                    return res.choices[0].message.content, f"Fallback: {model_id}"
                except Exception:
                    continue

        raise RuntimeError("All LLM providers in the fallback chain failed.")

    def run_autonomous_loop(self, topic: str, max_steps: int = 5) -> dict:
        """
        Multi-turn autonomous tool loop with strict JSON output contract.
        """
        system_instruction = (
            "You are an expert AI Tutor. Your task is to generate a comprehensive lesson with executable Python code.\n\n"
            "AVAILABLE TOOLS:\n"
            "1. python_syntax_checker(code: str)\n"
            "2. generate_quiz_question(topic: str)\n\n"
            "RULES:\n"
            "1. In Turn 1: Draft the lesson and python code. Call 'python_syntax_checker' to verify code syntax.\n"
            "2. In Turn 2: Call 'generate_quiz_question' to generate a self-assessment question.\n"
            "3. In Turn 3: Return the 'final_answer'.\n\n"
            "RESPONSE FORMATS (You MUST return strict JSON only):\n\n"
            "For Tool Calls:\n"
            "{\n"
            '  "type": "tool_call",\n'
            '  "tool": "python_syntax_checker",\n'
            '  "args": {"code": "<your_python_code_here>"}\n'
            "}\n\n"
            "For Final Answer:\n"
            "{\n"
            '  "type": "final_answer",\n'
            '  "explanation": "<detailed simple analogy and clear explanation of the topic>",\n'
            '  "code": "<complete, executable python code demonstrating the topic with print statements>",\n'
            '  "quiz_question": "<a reflective self-check question>"\n'
            "}"
        )

        history = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Teach me this topic thoroughly: {topic}"}
        ]

        trace = []
        turn_count = 0

        for step in range(1, max_steps + 1):
            turn_count += 1
            response_text, provider = self._call_llm(history)
            
            trace_entry = {
                "turn": step,
                "provider": provider,
                "action": "Processing..."
            }

            try:
                # Clean Markdown code formatting if LLM wraps JSON in ```json
                clean_json = response_text.strip()
                if "```" in clean_json:
                    clean_json = clean_json.split("```")[1]
                    if clean_json.startswith("json"):
                        clean_json = clean_json[4:].strip()
                
                data = json.loads(clean_json)

                # Tool Execution Handler
                if data.get("type") == "tool_call":
                    tool_name = data.get("tool")
                    tool_args = data.get("args", {})
                    
                    if tool_name in AVAILABLE_TOOLS:
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                        trace_entry["action"] = f"Executed Tool: `{tool_name}`"
                        trace_entry["tool_result"] = tool_result
                        
                        history.append({"role": "assistant", "content": response_text})
                        history.append({"role": "user", "content": f"TOOL RESULT: {tool_result}"})
                    
                    trace.append(trace_entry)
                    continue

                # Final Answer Handler
                if data.get("type") == "final_answer":
                    trace_entry["action"] = "Final Answer Delivered"
                    trace.append(trace_entry)
                    return {
                        "topic": topic,
                        "explanation": data.get("explanation", ""),
                        "code": data.get("code", ""),
                        "quiz_question": data.get("quiz_question", ""),
                        "turns_taken": turn_count,
                        "trace": trace
                    }

            except Exception:
                # Handle non-JSON output gracefully
                history.append({"role": "assistant", "content": response_text})
                history.append({"role": "user", "content": "Please output valid JSON matching either 'tool_call' or 'final_answer' schema."})

            trace.append(trace_entry)

        # Direct Single-Turn Fallback if loop exceeds max steps
        return self._generate_direct_fallback(topic, trace, turn_count)

    def _generate_direct_fallback(self, topic: str, trace: list, turn_count: int) -> dict:
        """Emergency direct generator if multi-turn loop hits step limits."""
        fallback_prompt = [
            {"role": "system", "content": "You are a Python Educator. Provide a structured lesson with explanation and runnable code."},
            {"role": "user", "content": f"Provide a complete lesson on '{topic}'. Include a simple analogy, clear explanation, and working Python code to test metrics."}
        ]
        text, provider = self._call_llm(fallback_prompt)
        
        # Split explanation and code snippet
        code_block = "import numpy as np\n# Sample code\nprint('Evaluating metric calculations...')"
        if "```python" in text:
            parts = text.split("```python")
            explanation = parts[0]
            code_block = parts[1].split("```")[0].strip()
        else:
            explanation = text

        return {
            "topic": topic,
            "explanation": explanation,
            "code": code_block,
            "quiz_question": f"How do the metrics in {topic} trade off against each other?",
            "turns_taken": turn_count,
            "trace": trace
        }

    def validate_user_summary(self, topic: str, user_summary: str) -> tuple[dict, str]:
        """Validates user's written feedback."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Evaluate the user's summary against the topic. Return JSON ONLY:\n"
                    "{\n"
                    '  "verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT",\n'
                    '  "score": <0-10 integer>,\n'
                    '  "feedback": "<detailed constructive feedback>"\n'
                    "}"
                )
            },
            {"role": "user", "content": f"Topic: {topic}\nUser Explanation: {user_summary}"}
        ]
        raw_res, provider = self._call_llm(messages)
        try:
            clean_json = raw_res.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean_json), provider
        except Exception:
            return {"verdict": "EVALUATED", "score": 8, "feedback": raw_res}, provider