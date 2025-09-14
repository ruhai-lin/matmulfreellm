import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from mmfreelm.models.hgrn_bit import HGRNBitConfig, HGRNBitForCausalLM

def main():
    # 1. 载入分词器
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 载入 TinyStories 并做简单 token 化
    block_size = 256
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=block_size)
    ds = load_dataset("roneneldan/TinyStories")
    tokenized = ds.map(tok_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 3. 构建约 15M 参数的模型配置
    config = HGRNBitConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,          # 控制模型规模的关键参数
        num_hidden_layers=8,
        num_heads=4,
        expand_ratio=1,
        hidden_ratio=4,
        max_position_embeddings=block_size,
    )
    model = HGRNBitForCausalLM(config)
    print("参数总量 (M):", model.num_parameters() / 1e6)

    # 4. 训练参数
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir="./mmfreelm-15m",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),  # 如果显卡支持，启用半精度
        logging_steps=50,
        eval_strategy="steps",
        save_steps=500,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        torch_compile=False,  # Triton + compile 容易首轮很慢/不稳，先关
    )

    # 5. 启动 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    trainer.train()

if __name__ == "__main__":
    main()
