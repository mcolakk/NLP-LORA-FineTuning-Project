# train.py
# ==============================================================================
# NLP PROJESİ - LORA FINE-TUNING EĞİTİM KODU (V4 CUSTOM)
# Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
# Ayarlar: Rank 16, Alpha 32, Dropout 0.1, LR 2e-5, Weight Decay 0.1
# ==============================================================================

import os
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ==========================================
# ⚙️ HİPERPARAMETRELER (V4 AYARLARI)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR_BASE = "./results"  # GitHub için yerel klasör ayarlandı

# Eğitim Ayarları (V4 - Stability & Anti-Overfit)
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5     # Düşük öğrenme hızı
WEIGHT_DECAY = 0.1       # Ağırlık çürümesi (Overfitting önleyici)
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
CONTEXT_LENGTH = 1024
PATIENCE = 3             # Early Stopping sabrı

# LoRA Ayarları (V4)
LORA_R = 16              # Rank
LORA_ALPHA = 32          # Alpha
LORA_DROPOUT = 0.1       # Dropout artırıldı

SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

# Hafıza Yönetimi
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 🛠️ EĞİTİM FONKSİYONU
# ==========================================
def train_model_v4(dataset_name, hf_dataset_path):
    # Temizlik
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = f"{OUTPUT_DIR_BASE}/{dataset_name}_Model_V4_Custom"
    print(f"\n{'='*50}")
    print(f"🚀 {dataset_name} EĞİTİMİ BAŞLIYOR (V4)...")
    print(f"📂 Kayıt Yeri: {output_dir}")

    # 1. Dataset Yükleme ve Ayrıştırma
    print("📥 Veri seti yükleniyor...")
    try:
        full_dataset = load_dataset(hf_dataset_path, split="train")

        # Senin kodundaki özel split mantığı
        if "split" in full_dataset.column_names:
            print("✅ 'split' sütunu bulundu. Ayrıştırılıyor...")
            train_dataset = full_dataset.filter(lambda x: x["split"] == "train")
            eval_dataset = full_dataset.filter(lambda x: x["split"] == "test")

            if len(eval_dataset) == 0:
                print("⚠️ Test etiketi boş, 'valid' etiketi deneniyor...")
                eval_dataset = full_dataset.filter(lambda x: x["split"] == "valid")
        else:
            # Split sütunu yoksa otomatik ayır
            print("⚠️ 'split' sütunu yok. Otomatik %10 validation ayrılıyor.")
            split_data = full_dataset.train_test_split(test_size=0.1)
            train_dataset = split_data["train"]
            eval_dataset = split_data["test"]

        print(f"✅ Hazır: Train ({len(train_dataset)}) | Val ({len(eval_dataset)})")

    except Exception as e:
        print(f"❌ Veri seti yükleme hatası: {e}")
        return

    # 2. Model & Tokenizer Yükleme
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Model yükleme (4-bit quantization config eklenebilir, burada standart yükleme var)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        use_cache=False
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. LoRA Yapılandırması
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Veri Formatlama (Chat Template)
    def format_chat(sample):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["solution"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(text, truncation=True, max_length=CONTEXT_LENGTH, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("🔄 Veriler işleniyor (Tokenization)...")
    train_dataset = train_dataset.map(format_chat, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_chat, remove_columns=eval_dataset.column_names)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True, fp16=False, 
        optim="adamw_torch_fused", 
        report_to="none"
    )

    # 6. Trainer Başlatma
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    # Eğitimi Başlat
    trainer.train()

    # Modeli Kaydet
    final_path = f"{output_dir}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ {dataset_name} Eğitimi Tamamlandı! Kayıt: {final_path}")

    # Bellek Temizliği
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# ▶️ ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    # Not: GitHub'da çalıştırmak isteyenler için dataset isimleri
    print("Eğitim Scripti Başlatılıyor...")
    
    # 1. DEEP Eğitimi
    train_model_v4("DEEP", "Naholav/CodeGen-Deep-5K")
    
    # 2. DIVERSE Eğitimi
    train_model_v4("DIVERSE", "Naholav/CodeGen-Diverse-5K")