import json
import os
import textwrap
from dotenv import load_dotenv
from openai import OpenAI

def get_ai_response(chat_log, openai_key):
    try:
        client = OpenAI(api_key=openai_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_log,
            max_tokens=150,                  # 応答の最大長
            n=1,                             # 生成する応答数
            stream=False,                   # ストリーミングは使わない
            stop=None                       # 停止トークンなし
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"APIリクエストエラー: {e}")


def main():
    print(textwrap.dedent('''\
        === OpenAI Chat CLI ===
        終了するには「exit」または Ctrl-C を入力してください。
    '''))

    load_dotenv()

    openai_key = os.environ['OPEN_AI_KEY']

    chat_log = []

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() == 'exit':
                print("チャットを終了します。")
                break
            
            chat_log.append({"role": "user", "content": user_input})

            ai_reply = get_ai_response(chat_log, openai_key)

            print(f"AI : {ai_reply}")
        except KeyboardInterrupt:
            print("チャットを終了します。")
            break

if __name__ == "__main__":
    main()