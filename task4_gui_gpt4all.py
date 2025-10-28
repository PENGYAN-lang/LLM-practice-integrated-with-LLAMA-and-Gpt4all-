# task4_gui_gpt4all.py
import threading, queue, tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from gpt4all import GPT4All

# ===== 路径与模型名（与你已下载的一致）=====
MODELS_DIR = Path("/media/peng/Data/models")
MODEL_A = "llama.cpp.gguf"                 # A 位（已下载 608MB 的 TinyLlama）
MODEL_B = "replit-code-v1_5-3b-q4_0.gguf"  # B 位（可用硬链接/复制指向 A）
# =========================================

CTX_LEN = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.7

def format_prompt(history, user_msg):
    """简单通用的 Llama 风格提示词"""
    sys = "You are a helpful, concise assistant."
    text = f"<s>[INST] <<SYS>>{sys}<</SYS>>\n"
    for role, msg in history:
        if role == "user":
            text += f"{msg}\n[/INST] "
        else:
            text += f"{msg}</s>\n<s>[INST] "
    text += f"{user_msg} [/INST]"
    return text

class Gpt4AllWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._llm = None

    def load(self):
        if self._llm is None:
            path = MODELS_DIR / self.model_name
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            self._llm = GPT4All(
                model_name=self.model_name,
                model_path=str(MODELS_DIR),
                allow_download=False,  # 禁止联网下载
                device="cpu",
                verbose=False
            )
        return self

    def generate_stream(self, prompt: str):
        # 为了简洁，这里用一次性生成再“字符流”喂给 GUI（不卡界面）
        text = self._llm.generate(
            prompt,
            max_tokens=MAX_TOKENS,
            temp=TEMPERATURE,
            top_p=0.9
        )
        for ch in text:
            yield ch

class ChatGUI:
    def __init__(self, root):
        root.title("EE5112 Task 4 – Local GGUF Chat (GPT4All)")
        root.geometry("860x600")

        # 顶部：模型选择、加载、清空
        top = ttk.Frame(root, padding=6); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.models = {
            f"A: {MODEL_A}": Gpt4AllWrapper(MODELS_DIR and MODEL_A),
            f"B: {MODEL_B}": Gpt4AllWrapper(MODELS_DIR and MODEL_B),
        }
        self.model_var = tk.StringVar(value=list(self.models.keys())[0])
        self.model_box = ttk.Combobox(
            top, textvariable=self.model_var,
            values=list(self.models.keys()),
            state="readonly", width=60
        )
        self.model_box.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load/Reload", command=self.load_model).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Clear Chat", command=self.clear_chat).pack(side=tk.LEFT, padx=6)

        # 中部：对话显示区
        mid = ttk.Frame(root, padding=(6,0,6,6)); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.text = tk.Text(mid, wrap=tk.WORD, state=tk.DISABLED)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, command=self.text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text['yscrollcommand'] = sb.set

        # 底部：输入 + 发送
        bottom = ttk.Frame(root, padding=6); bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.entry = ttk.Entry(bottom)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        ttk.Button(bottom, text="Send", command=self.on_send).pack(side=tk.LEFT)

        # 状态栏
        self.status = tk.StringVar(value="Ready. Click 'Load/Reload' to load model.")
        ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor="w", padding=3)\
           .pack(side=tk.BOTTOM, fill=tk.X)

        # 运行时状态
        self.history, self.current = [], None
        self.queue = queue.Queue()
        root.after(50, self._drain_queue)

    def load_model(self):
        try:
            key = self.model_var.get()
            self.status.set(f"Loading {key} ...")
            self.current = self.models[key].load()
            self.status.set(f"Loaded: {key}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.status.set("Load failed.")

    def clear_chat(self):
        self.history.clear()
        self.text.config(state=tk.NORMAL); self.text.delete("1.0", tk.END); self.text.config(state=tk.DISABLED)
        self.status.set("Chat cleared.")

    def append_text(self, who, msg):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, f"{who}: {msg}\n")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def on_send(self):
        msg = self.entry.get().strip()
        if not msg: return
        if self.current is None:
            messagebox.showwarning("No model", "Please load a model first."); return
        self.entry.delete(0, tk.END)
        self.history.append(("user", msg))
        self.append_text("You", msg)
        self.append_text("Assistant", "")
        threading.Thread(target=self._worker, args=(msg,), daemon=True).start()
        self.status.set("Generating...")

    def _worker(self, user_msg):
        try:
            prompt = format_prompt(self.history[:-1], user_msg)
            for tok in self.current.generate_stream(prompt):
                self.queue.put(tok)
            self.queue.put("\n")
        except Exception as e:
            self.queue.put(f"\n[ERROR] {e}\n")
        finally:
            self.queue.put("[[__DONE__]]")

    def _drain_queue(self):
        done = False
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item == "[[__DONE__]]":
                done = True
            else:
                self.text.config(state=tk.NORMAL)
                self.text.insert(tk.END, item)
                self.text.see(tk.END)
                self.text.config(state=tk.DISABLED)
        if done:
            self.status.set("Ready.")
        self.text.after(50, self._drain_queue)

if __name__ == "__main__":
    root = tk.Tk()
    ChatGUI(root)
    root.mainloop()
