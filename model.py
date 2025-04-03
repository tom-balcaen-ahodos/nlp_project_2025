import torch
from huggingface_hub import login
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

login(token = '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')

# bnb_config = BitsAndBytesConfig(
#   load_in_4bit=True,
#   bnb_4bit_quant_type="nf4",
#   bnb_4bit_use_double_quant=True,
#   bnb_4bit_compute_dtype=torch.float16
# )
repo_id = 'distilbert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(
  repo_id, device_map="cpu"
)

# model = prepare_model_for_kbit_training(model)
config = LoraConfig(
  # the rank of the adapter, the lower the fewer parameters you'll need to train
  r=8,
  lora_alpha=16, # multiplier, usually 2*r
  bias="none",
  lora_dropout=0.05,
  task_type="SEQ_CLS",
  target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)
model = get_peft_model(model, config)

#print trainable parameters
model.print_trainable_parameters()
#print memory
print(model.get_memory_footprint()/1e6)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.chat_template



# #SFT config object
# sft_config = SFTConfig(
#   ## GROUP 1: Memory usage
#   # These arguments will squeeze the most out of your GPU's RAM
#   # Checkpointing
#   gradient_checkpointing=True, # this saves a LOT of memory
#   # Set this to avoid exceptions in newer versions of PyTorch
#   gradient_checkpointing_kwargs={'use_reentrant': False},
#   # Gradient Accumulation / Batch size
#   # Actual batch (for updating) is same (1x) as micro-batch size
#   gradient_accumulation_steps=1,
#   # The initial (micro) batch size to start off with
#   per_device_train_batch_size=16,
#   # If batch size would cause OOM, halves its size until it works
#   auto_find_batch_size=True,
#   ## GROUP 2: Dataset-related
#   max_seq_length=64,
#   # Dataset
#   # packing a dataset means no padding is needed
#   packing=True,
#   ## GROUP 3: These are typical training parameters
#   num_train_epochs=10,
#   learning_rate=3e-4,
#   # Optimizer
#   # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
#   optim='paged_adamw_8bit',
#   ## GROUP 4: Logging parameters
#   logging_steps=10,
#   logging_dir='./logs',
#   output_dir='./llama-3.2-1b-adapter',
#   report_to='none'
# )


# train_dataset = load_dataset('json', data_files='output_dataset.json', split='train')

# trainer = SFTTrainer(
#   model=model,
#   processing_class=tokenizer,
#   args=sft_config,
#   train_dataset=train_dataset,
# )

trainer.save_model("./fine_tuned_model_v1")