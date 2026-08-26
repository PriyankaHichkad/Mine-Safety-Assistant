import os
import json
import urllib.request
from typing import Dict, Any, List, Optional

try:
    import groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

class OllamaLLMProvider:
    """
    Enterprise Unified LLM Provider supporting:
    1. Groq Cloud API (Sub-Second Llama-3.3-70B / Llama3-8B at 300+ tokens/sec)
    2. Ollama Local LLM (http://localhost:11434)
    3. Grounded Fallback Engine
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:latest",
        temperature: float = 0.2,
        timeout: int = 120,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        
        # Resolve Groq API key from arg or environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.groq_client = None
        if self.api_key and HAS_GROQ:
            try:
                self.groq_client = groq.Groq(api_key=self.api_key)
            except Exception as err:
                print(f"[Groq Provider Warning]: Failed to initialize Groq client: {err}")

    def check_health(self) -> Dict[str, Any]:
        """Verifies active LLM provider connectivity (Groq Cloud API vs Ollama Local)."""
        if self.groq_client:
            return {"status": True, "provider": "Groq Cloud API", "model": "llama-3.3-70b-versatile"}
            
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return {"status": True, "provider": "Ollama Local LLM", "model": self.model}
        except Exception:
            pass
            
        return {"status": False, "provider": "Grounded Fallback Engine", "model": "Rule-Based Grounded Engine"}

    def get_available_models(self) -> List[str]:
        """Retrieves list of available model tags."""
        if self.groq_client:
            return ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"]
            
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m["name"] for m in data.get("models", [])]
                    return models if models else ["llama3:latest"]
        except Exception:
            pass
        return ["llama3:latest"]

    def _resolve_model_name(self, requested_model: str, available_models: List[str]) -> str:
        """Resolves requested model string to exact target tag."""
        if requested_model in available_models:
            return requested_model
        
        req_clean = requested_model.lower().split(":")[0]
        for avail in available_models:
            if avail.lower().startswith(req_clean):
                return avail
                
        return available_models[0] if available_models else requested_model

    def generate_response(self, system_prompt: str, user_prompt: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Sends RAG prompt to LLM provider hierarchy:
        1. Groq Cloud API (High-Speed Llama-3 at 300+ tokens/sec)
        2. Ollama Local LLM (http://localhost:11434)
        3. Returns None for Grounded Fallback Engine
        """
        # --- Priority 1: Groq Cloud API (Sub-Second Execution) ---
        if self.groq_client:
            target_groq_model = model_name or "llama-3.3-70b-versatile"
            if "8b" in str(model_name).lower():
                target_groq_model = "llama3-8b-8192"
                
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=target_groq_model,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                content = chat_completion.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception as err:
                print(f"[Groq Provider Notice]: Groq API call failed ({err}). Falling back to Ollama / Grounded Engine...")

        # --- Priority 2: Ollama Local LLM ---
        available = self.get_available_models()
        target_model = self._resolve_model_name(model_name or self.model, available)
        
        url_gen = f"{self.base_url}/api/generate"
        full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n{user_prompt}"
        payload_gen = {
            "model": target_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        
        try:
            req = urllib.request.Request(
                url_gen,
                data=json.dumps(payload_gen).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    if "response" in res_json and res_json["response"].strip():
                        return res_json["response"].strip()
        except Exception:
            pass

        # --- Priority 3: Ollama Chat Fallback ---
        url_chat = f"{self.base_url}/api/chat"
        payload_chat = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                url_chat,
                data=json.dumps(payload_chat).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    msg = res_json.get("message", {})
                    if "content" in msg and msg["content"].strip():
                        return msg["content"].strip()
        except Exception as err:
            print(f"[Ollama Provider Notice]: Ollama call failed ({err}). Using grounded fallback engine.")
            return None
            
        return None
