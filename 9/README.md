### Install Packages in google colab

```shell
%%capture
# Unsloth、Xformers (Flash Attention)、その他のすべてのパッケージをインストール
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```


### Learn setting
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 0より大きい任意の数値を選択してください！8、16、32、64、128を推奨します
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # 任意の値をサポートしていますが、= 0が最適化されています
    bias="none",  # 任意の値をサポートしていますが、= "none"が最適化されています
    # [新機能] "unsloth"はVRAMの使用量を30%削減し、2倍のサイズのバッチサイズに対応します！
    use_gradient_checkpointing="unsloth",  # 非常に長いコンテキストの場合は、Trueまたは"unsloth"を使用します
    random_state=3407,
    use_rslora=False,  # ランク安定化LoRAをサポートしています
    loftq_config=None,  # LoftQもサポートしています
)
```

### Upload Learn Files

```shell
from google.colab import files
uploaded = files.upload()
```


### Json Load

```python
from datasets import load_dataset

# JSONLをロード
dataset = load_dataset("json", data_files="data.json")["train"]

# Alpaca形式に沿って整形
def formatting_prompts(example):
    instruction = example["instruction"]
    input_text  = example.get("input", "")
    output      = example["output"]
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return {"text": prompt + output}

dataset = dataset.map(formatting_prompts)
print(dataset)
```


```python
from unsloth import to_sharegpt
dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # より長い会話を処理するには、これを増やしてください
)
```


```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # 短いシーケンスの場合、トレーニングを5倍高速化できます
    args=TrainingArguments(
        report_to="none",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        # num_train_epochs = 1,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

```


```python
trainer_stats = trainer.train()
```


```python
FastLanguageModel.for_inference(model)  # ネイティブで2倍高速な推論を有効にします
messages = [  # 以下を変更してください！
    {"role": "user", "content": "こんにちは。今日の天気について教えてください。"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id
)

```

```python

model.save_pretrained("lora_model")  # ローカルに保存
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # オンラインに保存
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # オンラインに保存
```

```python
if False:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # トレーニングに使用したモデル
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # ネイティブで2倍高速な推論を有効にします
pass

messages = [  # 以下を変更してください！
    {
        "role": "user",
        "content": "こんにちは。今日の天気について教えてください。",
    },
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id
)
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 8ビットQ8_0に保存
if True:
    model.save_pretrained_gguf("model", tokenizer)
# https://huggingface.co/settings/tokens にアクセスしてトークンを取得してください。
# また、hfを自分のユーザー名に変更してください。
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# 16ビットGGUFに保存
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# q4_k_m GGUFに保存
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")

# 複数のGGUFオプションに保存 - 複数必要な場合ははるかに高速です！
if False:
    model.push_to_hub_gguf(
        "hf/model",  # hfを自分のユーザー名に変更してください。
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
        token="",  # https://huggingface.co/settings/tokens からトークンを取得してください
    )
```

```python
import subprocess
subprocess.Popen(["ollama", "serve"])
import time
time.sleep(3) # Ollamaがロードされるまで数秒間待ちます。
```

```python
print(tokenizer._ollama_modelfile)
``` 