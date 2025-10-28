# gpt4all_probe.py
from gpt4all import GPT4All
from pathlib import Path

MODELS_DIR = Path("/media/peng/Data/models")
A = "llama.cpp.gguf"
B = "replit-code-v1_5-3b-q4_0.gguf"  # 你已有

def try_one(name: str):
    path = MODELS_DIR / name
    assert path.exists(), f"Not found: {path}"
    print(f"\nLoading via GPT4All: {path}")
    llm = GPT4All(model_name=name, model_path=str(MODELS_DIR), allow_download=False, verbose=True)
    with llm.chat_session(system_prompt="You are a helpful, concise assistant."):
        out = llm.generate("Say hello in one short sentence.", max_tokens=64, temp=0.2)
    print("Reply:", out.strip())

if __name__ == "__main__":
    try_one(B)      # 先测 Replit（更常见能直接通）
    # try_one(A)    # 若 B 成功，再把这一行解除注释测 A
    print("\nProbe done ✅")
