from llama_cpp import Llama
import textwrap

def main():
    
    # LLMの準備
    llm = Llama(model_path="..models/llama-2-7b-chat.Q2_K.gguf")

    while True:
        try:
            user_input = input("prompt: ")

            if user_input.lower() == 'exit':
                print("チャットを終了します。")
                break
            
            # プロンプトの準備
            prompt = f"""prompt: {user_input}\nAI: """

            # 推論の実行
            output = llm(
                prompt,
                temperature=0.1,
                stop=["prompt:", "AI:", "\n"],
                echo=False,
            )

            print(f"AI: {output["choices"][0]["text"]}")
        except KeyboardInterrupt:
            print("チャットを終了します。")
            break

if __name__ == "__main__":
    main()