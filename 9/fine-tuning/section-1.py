from unsloth import FastLanguageModel
import torch
max_seq_length = 2048  # 任意の値を選択可能。RoPEスケーリングは内部で自動的にサポート
dtype = None  # 自動検出の場合はNone。Tesla T4、V100の場合はFloat16、Ampere以降の場合はBfloat16
load_in_4bit = True  # メモリ使用量を削減するために4ビット量子化を使用。Falseも可能

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",  # meta-llama/Llama-2-7b-hfのようなゲート付きモデルを使用する場合は、これを使用
)