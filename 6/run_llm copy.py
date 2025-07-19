from llama_cpp import Llama
import textwrap

def main():
    
    # LLMの準備
    llm = Llama(model_path="../models/rinna-youri-7b-chat-q2_K.gguf")

    # チャットログの初期化
    chat_log = []

    while True:
        try:
            user_input = input("prompt: ")

            # ユーザー発言を履歴に追加
            chat_log.append(f"prompt: {user_input}")

            # 会話履歴を1つのプロンプトにまとめる
            history = "\n".join(chat_log) + "\nAI:"

            if user_input.lower() == 'exit':
                print("チャットを終了します。")
                break
            
            # 推論の実行
            output = llm(
                history,
                max_tokens=150,
                temperature=0.1,
                stop=["prompt:", "AI:", "\n"],
                echo=False,
            )

            print(output)
            reply = output["choices"][0]["text"].strip()
            print(f"AI: {reply}")

            # AIの応答も履歴に追加
            chat_log.append(f"AI: {reply}")
        except KeyboardInterrupt:
            print("チャットを終了します。")
            break

if __name__ == "__main__":
    main()