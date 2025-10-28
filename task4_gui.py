#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EE5112 Project 1 – Task 4 GUI (Tkinter)
---------------------------------------
Features required by Task 4:
- ChatGPT-like multi-turn chat UI
- Adjustable model (Echo / GPT4All / llama.cpp)
- Clear dialogue box
- ENTER to send, Shift+ENTER for newline
- Non-blocking UI with streaming tokens where possible

Usage (Ubuntu / Conda, Python 3.9):
1) Ensure Tkinter is available (Ubuntu):
   sudo apt-get update && sudo apt-get install -y python3-tk

2) (Optional) Install one or both backends:
   # GPT4All
   pip install gpt4all
   # llama.cpp python bindings
   pip install llama-cpp-python

3) Run:
   python task4_gui.py

Tip: You can already test the GUI with the built-in 'Echo (test)' backend
without installing any LLM first. Then switch to GPT4All or llama.cpp later.
"""
import os
import sys
import time
import threading
from datetime import datetime
from functools import partial

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:
    print("Tkinter is required to run this GUI. On Ubuntu: sudo apt-get install python3-tk")
    raise

# -------- Backend Abstractions --------
class BaseBackend:
    """Abstract base for all backends."""
    name = "Base"
    def __init__(self, model_path=None, temperature=1.0):
        self.model_path = model_path
        self.temperature = float(temperature)

    def load(self):
        """Load heavy resources here. Should be idempotent."""
        pass

    def generate(self, messages, stop_event=None):
        """
        Yield text chunks (streaming if possible).
        messages: list of dicts like [{"role":"system/user/assistant", "content": "..."}]
        stop_event: threading.Event to allow cooperative cancel.
        """
        raise NotImplementedError


class EchoBackend(BaseBackend):
    """A dummy backend to verify the GUI quickly."""
    name = "Echo (test)"
    def generate(self, messages, stop_event=None):
        user_text = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_text = m["content"]
                break
        reply = f"(echo) You said: {user_text[:1200]}"
        # pretend to stream
        for ch in reply:
            if stop_event and stop_event.is_set():
                return
            yield ch
            time.sleep(0.002)


class GPT4AllBackend(BaseBackend):
    """GPT4All Python bindings.
    You can pass either a model name (e.g. 'orca-mini-3b.gguf') if it exists in GPT4All's models dir,
    or an absolute path to a .gguf file.
    """
    name = "GPT4All"
    def __init__(self, model_path=None, temperature=1.0):
        super().__init__(model_path, temperature)
        self._model = None
        self._last_err = None

    def load(self):
        if self._model is not None:
            return
        try:
            from gpt4all import GPT4All
        except Exception as e:
            self._last_err = f"Failed to import gpt4all: {e}\nTry: pip install gpt4all"
            raise
        # If model_path is None, GPT4All will ask to download a model. We prefer explicit path/name.
        model_id = self.model_path or "gpt4all-falcon-q4_0.gguf"  # safe default name
        self._model = GPT4All(model_id)

    def generate(self, messages, stop_event=None):
        if self._model is None:
            self.load()
        # simple chat prompt: concatenate history
        prompt_lines = []
        for m in messages:
            role = m["role"]
            content = m["content"].strip()
            if role == "system":
                prompt_lines.append(f"[System]: {content}")
            elif role == "user":
                prompt_lines.append(f"[User]: {content}")
            elif role == "assistant":
                prompt_lines.append(f"[Assistant]: {content}")
        prompt_lines.append("[Assistant]:")
        prompt = "\n".join(prompt_lines)

        # stream tokens using callback
        buf = []
        def _cb(token_id, token_str):
            if stop_event and stop_event.is_set():
                return False  # stop streaming
            buf.append(token_str)
            return True

        try:
            with self._model.chat_session():
                # Note: parameter names may vary slightly by version
                self._model.generate(
                    prompt,
                    temp=self.temperature,
                    max_tokens=512,
                    streaming=True,
                    callback=_cb
                )
        except Exception as e:
            err = f"[GPT4All error] {e}"
            for ch in err:
                yield ch
            return

        # Yield from buffer in small chunks
        for ch in "".join(buf):
            if stop_event and stop_event.is_set():
                return
            yield ch


class LlamaCppBackend(BaseBackend):
    """llama-cpp-python bindings."""
    name = "llama.cpp"
    def __init__(self, model_path=None, temperature=1.0, n_ctx=4096, n_threads=None):
        super().__init__(model_path, temperature)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(f"Failed to import llama-cpp-python: {e}\nTry: pip install llama-cpp-python")
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError("Please set a valid .gguf model path for llama.cpp.")
        # Initialize
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads or max(1, os.cpu_count() or 1),
            verbose=False,
        )

    def generate(self, messages, stop_event=None):
        self.load()
        # Build OpenAI-like messages
        msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role not in ("system", "user", "assistant"):
                role = "user"
            msgs.append({"role": role, "content": content})

        try:
            # Stream tokens via create_chat_completion (if supported)
            stream = self._llm.create_chat_completion(
                messages=msgs,
                temperature=self.temperature,
                max_tokens=512,
                stream=True
            )
            for part in stream:
                if stop_event and stop_event.is_set():
                    return
                delta = part.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    for ch in token:
                        if stop_event and stop_event.is_set():
                            return
                        yield ch
        except Exception as e:
            err = f"[llama.cpp error] {e}"
            for ch in err:
                yield ch


# -------- GUI --------
class ChatGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("EE5112 Task 4 – Local LLM GUI (Tkinter)")
        self.pack(fill="both", expand=True)

        # Conversation buffer as list of {"role": ..., "content": ...}
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # Backends registry and current backend
        self.backend_var = tk.StringVar(value="Echo (test)")
        self.available_backends = {
            "Echo (test)": EchoBackend,
            "GPT4All": GPT4AllBackend,
            "llama.cpp": LlamaCppBackend,
        }
        self.backend = None  # lazy init

        # Runtime controls
        self.temperature = tk.DoubleVar(value=0.9)
        self.model_path = tk.StringVar(value="")  # used for GPT4All (optional) and llama.cpp (required)
        self.n_ctx = tk.IntVar(value=4096)
        self.stream_enabled = tk.BooleanVar(value=True)

        self._infer_thread = None
        self._stop_event = threading.Event()

        self._build_widgets()

    # ---- UI helpers ----
    def _build_widgets(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Label(top, text="Backend:").pack(side="left")
        backend_box = ttk.Combobox(top, textvariable=self.backend_var, values=list(self.available_backends.keys()), width=14, state="readonly")
        backend_box.pack(side="left", padx=6)

        ttk.Label(top, text="Temperature:").pack(side="left", padx=(10,0))
        temp_scale = ttk.Scale(top, from_=0.1, to=1.5, variable=self.temperature, orient="horizontal", length=160)
        temp_scale.pack(side="left", padx=4)
        self.temp_label = ttk.Label(top, text=f"{self.temperature.get():.2f}")
        self.temp_label.pack(side="left", padx=(4,0))
        temp_scale.bind("<ButtonRelease-1>", lambda e: self.temp_label.config(text=f"{self.temperature.get():.2f}"))

        ttk.Checkbutton(top, text="Stream", variable=self.stream_enabled).pack(side="left", padx=10)

        # Model path row (needed for llama.cpp and optional for GPT4All)
        path_row = ttk.Frame(self)
        path_row.pack(side="top", fill="x", padx=8, pady=(0,6))
        ttk.Label(path_row, text="Model path (.gguf):").pack(side="left")
        path_entry = ttk.Entry(path_row, textvariable=self.model_path)
        path_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(path_row, text="Browse", command=self._browse_model_path).pack(side="left")
        ttk.Label(path_row, text="Ctx:").pack(side="left", padx=(10,0))
        ttk.Entry(path_row, width=6, textvariable=self.n_ctx).pack(side="left")

        # Chat history
        hist_frame = ttk.Frame(self)
        hist_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0,6))
        self.history = tk.Text(hist_frame, wrap="word", height=18, state="disabled")
        self.history.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(hist_frame, command=self.history.yview)
        scroll.pack(side="right", fill="y")
        self.history["yscrollcommand"] = scroll.set

        # Input area
        input_frame = ttk.Frame(self)
        input_frame.pack(side="top", fill="x", padx=8, pady=(0,8))
        self.input = tk.Text(input_frame, height=3, wrap="word")
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self._on_enter_send)
        self.input.bind("<Shift-Return>", lambda e: None)  # allow newline with Shift+Enter

        btns = ttk.Frame(input_frame)
        btns.pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Send (Enter)", command=self.send_message).pack(side="top", fill="x")
        ttk.Button(btns, text="Clear Chat", command=self.clear_chat).pack(side="top", fill="x", pady=4)
        ttk.Button(btns, text="Stop", command=self.stop_generation).pack(side="top", fill="x")

        # Footer
        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", padx=8, pady=(0,6))
        ttk.Button(footer, text="Save Log", command=self.save_log).pack(side="right")

        self._append_to_history("System", "GUI ready. Choose backend and start chatting.")

    def _browse_model_path(self):
        path = filedialog.askopenfilename(title="Select .gguf model", filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _append_to_history(self, speaker, text):
        self.history.config(state="normal")
        self.history.insert("end", f"{speaker}: {text}\n")
        self.history.see("end")
        self.history.config(state="disabled")

    def _on_enter_send(self, event):
        # Enter = send; Shift+Enter = newline
        if event.state & 0x0001:  # Shift key
            return
        self.send_message()
        return "break"

    # ---- Chat logic ----
    def _get_or_create_backend(self):
        key = self.backend_var.get()
        BackendCls = self.available_backends.get(key, EchoBackend)
        # If backend type changed, create a new one
        if self.backend is None or not isinstance(self.backend, BackendCls):
            if BackendCls is LlamaCppBackend:
                self.backend = BackendCls(model_path=self.model_path.get().strip(), temperature=self.temperature.get(), n_ctx=self.n_ctx.get())
            else:
                self.backend = BackendCls(model_path=self.model_path.get().strip(), temperature=self.temperature.get())
        else:
            # Update settings
            self.backend.temperature = self.temperature.get()
            if isinstance(self.backend, LlamaCppBackend):
                self.backend.model_path = self.model_path.get().strip()
                self.backend.n_ctx = self.n_ctx.get()
            else:
                self.backend.model_path = self.model_path.get().strip()
        return self.backend

    def send_message(self):
        if self._infer_thread and self._infer_thread.is_alive():
            messagebox.showinfo("Busy", "Generation is in progress. Click 'Stop' to cancel, or wait.")
            return
        user_text = self.input.get("1.0", "end").strip()
        if not user_text:
            return
        self.input.delete("1.0", "end")
        self.messages.append({"role": "user", "content": user_text})
        self._append_to_history("You", user_text)

        # Start generation in a thread
        self._stop_event.clear()
        self._append_to_history("Assistant", "")  # placeholder line, we'll append to it
        self._infer_thread = threading.Thread(target=self._generate_worker, daemon=True)
        self._infer_thread.start()

    def _generate_worker(self):
        backend = None
        try:
            backend = self._get_or_create_backend()
            # Pre-load heavy model once (can raise helpful errors)
            backend.load()
        except Exception as e:
            self._safe_append("\n[Backend error] " + str(e) + "\n")
            return

        # Build message history for the model
        messages_for_model = list(self.messages)

        # Collect streamed text into a buffer
        response_buf = []

        try:
            for token in backend.generate(messages_for_model, stop_event=self._stop_event):
                if not self.stream_enabled.get():
                    response_buf.append(token)
                else:
                    # update UI incrementally via main thread
                    self._safe_append(token)
            # If streaming disabled, output consolidated text once
            if not self.stream_enabled.get():
                text = "".join(response_buf)
                self._safe_append(text)
        except Exception as e:
            self._safe_append("\n[Generation error] " + str(e) + "\n")
            return

        # Finalize: store assistant response in history
        final_text = self._history_last_assistant_text()
        self.messages.append({"role": "assistant", "content": final_text})

    def _history_last_assistant_text(self):
        # Read the last line that starts with "Assistant:"
        # Safer approach: track last assistant insertion index; here we parse text widget content.
        content = self.history.get("1.0", "end")
        lines = [ln for ln in content.splitlines() if ln.startswith("Assistant:")]
        if not lines:
            return ""
        return lines[-1].replace("Assistant:", "", 1).strip()

    def _safe_append(self, text):
        # Schedule UI update on main thread
        self.history.after(0, self._append_token_to_last_assistant, text)

    def _append_token_to_last_assistant(self, text):
        self.history.config(state="normal")
        # Find the index of the last "Assistant:" line end, and insert after it.
        idx = self.history.search("Assistant:", "end-1c linestart", backwards=True, regexp=False)
        if not idx:
            # if not found (shouldn't happen), append anew
            self.history.insert("end", f"Assistant: {text}")
            self.history.see("end")
            self.history.config(state="disabled")
            return
        # Move to end of that line, insert text, keep view at end
        line_end = idx.split(".")[0] + ".end"
        self.history.insert(line_end, text)
        self.history.see("end")
        self.history.config(state="disabled")

    def clear_chat(self):
        if self._infer_thread and self._infer_thread.is_alive():
            if not messagebox.askyesno("Confirm", "Generation is in progress. Stop and clear?"):
                return
            self.stop_generation()
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.history.config(state="normal")
        self.history.delete("1.0", "end")
        self.history.config(state="disabled")
        self._append_to_history("System", "Chat cleared.")

    def stop_generation(self):
        if self._infer_thread and self._infer_thread.is_alive():
            self._stop_event.set()

    def save_log(self):
        content = self.history.get("1.0", "end")
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("logs", f"chat_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Saved", f"Chat saved to {path}")


def main():
    root = tk.Tk()
    # Optional: nicer default fonts on some platforms
    try:
        root.option_add('*Dialog.msg.font', 'TkDefaultFont 10')
    except Exception:
        pass
    app = ChatGUI(root)
    root.geometry("860x640")
    root.mainloop()


if __name__ == "__main__":
    main()
