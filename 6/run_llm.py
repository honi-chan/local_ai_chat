from llama_cpp import Llama
import textwrap

def main():
    # モデルファイルのパス（必要に応じて変更）
    model_path = "../models/rinna-youri-7b-chat-q2_K.gguf"

    # LLMの初期化
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,              # コンテキスト長
        n_threads=4,             # 並列スレッド数
        use_mlock=True,          # メモリ固定（高速化）
        use_mmap=True            # メモリマップ
    )

    # 会話履歴（システムプロンプト込み）
    history = []

    while True:
        try:
            user_input = input("prompt: ")

            if user_input.lower() == 'exit':
                print("チャットを終了します。")
                break
                        
            # プロンプトの構築
            prompt = ""
            for turn in history:
                prompt += f"{turn['role']}: {turn['content']}\n"
            prompt += f"prompt: {user_input}\nAI:"
                        
            # 推論の実行
            output = llm(
                prompt,
                max_tokens=150,
                stop=["prompt:", "AI:", "\n"],
                echo=False
            )

            # 応答の抽出
            response = output["choices"][0]["text"].strip()

            history.append({"role": "prompt", "content": user_input})
            history.append({"role": "AI", "content": response})

            print(f"AI: {response}")
        except KeyboardInterrupt:
            print("チャットを終了します。")
            break

if __name__ == "__main__":
    main()