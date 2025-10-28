from gpt4all import GPT4All
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

def main():
    print(f"Loading the model: {MODEL_NAME} ...")
    model = GPT4All(model_name=MODEL_NAME, model_path=MODEL_DIR)
    print("Let's start to chat！(input 'quit' or 'exit' to end)\n")

    with model.chat_session():
        while True:
            user_input = input("> ")
            if user_input.strip().lower() in ["quit", "exit"]:
                print("chat end.")
                break
            response = model.generate(user_input,
                    max_tokens=300,
                    temp=0.7,
                    top_k=40,
                    top_p=0.9,
                    repeat_penalty=1.1)
            print(response)

if __name__ == "__main__":
    main()
