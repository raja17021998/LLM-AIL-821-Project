import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from transformers import BloomForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import evaluate
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from torch import nn


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


df_train = pd.read_csv("eng_guj_train.csv")
df_val = pd.read_csv("eng_guj_val.csv")


unique_labels = df_train['sentiment'].unique()
print("Unique labels in the dataset:", unique_labels)

label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
print("Label mapping:", label_mapping)


df_train['sentiment'] = df_train['sentiment'].map(label_mapping)
df_val['sentiment'] = df_val['sentiment'].map(label_mapping)

df_train['sentiment'] = df_train['sentiment'].astype(int)
df_val['sentiment'] = df_val['sentiment'].astype(int)

df_train = df_train.dropna(subset=['sentiment'])
df_val = df_val.dropna(subset=['sentiment'])


train_texts = df_train['eng_text'].tolist()
train_labels = df_train['sentiment'].tolist()

val_texts = df_val['eng_text'].tolist()
val_labels = df_val['sentiment'].tolist()


guj_texts = df_val['guj_text'].tolist()
guj_labels = df_val['sentiment'].tolist()


print("Sample train labels after mapping:", df_train['sentiment'].unique())
print("Sample val labels after mapping:", df_val['sentiment'].unique())


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=67,
    )
    tokenized_inputs['labels'] = examples['label']  
    return tokenized_inputs


train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
guj_dataset = Dataset.from_dict({'text': guj_texts, 'label': guj_labels})

print("Columns in Train Dataset Before Tokenization:", train_dataset.column_names)


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
guj_dataset = guj_dataset.map(tokenize_function, batched=True)


print("Columns in Train Dataset After Tokenization:", train_dataset.column_names)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
guj_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


num_labels = len(label_mapping)
print("Number of labels:", num_labels)


model = BloomForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    device_map="auto",
    load_in_8bit=True
)


model = prepare_model_for_int8_training(model)


lora_config = LoraConfig(
    r=8,                
    lora_alpha=32,     
    target_modules=["self_attention.dense_h_to_4h", "self_attention.query_key_value"], 
    lora_dropout=0.1,   
    bias="none",        
    task_type="SEQ_CLS" 
)


model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir='./results/lora/guj',
    num_train_epochs=6,
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir='./logs/lora/guj',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits

       
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def _load_best_model(self):

        if self.args.load_best_model_at_end:
            try:
                
                super()._load_best_model()
            except Exception as e:
                logger.warning(f"Error while loading best model: {e}")

           
            if hasattr(self.model, "active_adapters"):
                try:
                    active_adapter = self.model.active_adapters 
                    if isinstance(active_adapter, list) and len(active_adapter) > 0:
                        active_adapter = active_adapter[0]  
                    logger.info(f"Loaded best model and activated adapter: {active_adapter}")
                except Exception as e:
                    logger.warning(f"Failed to activate adapter: {e}")



trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()


print("Evaluation on English validation set:")
eval_results = trainer.evaluate()
print(f"Accuracy on English validation set: {eval_results['eval_accuracy']}")


print("Evaluation on Gujarati dataset:")
guj_results = trainer.predict(guj_dataset)
guj_predictions = np.argmax(guj_results.predictions, axis=-1)
guj_accuracy = metric.compute(predictions=guj_predictions, references=guj_results.label_ids)
print(f"Accuracy on Gujarati test data: {guj_accuracy['accuracy']}")


train_logs = trainer.state.log_history


train_loss = [log['loss'] for log in train_logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in train_logs if 'eval_loss' in log]
steps = [log['step'] for log in train_logs if 'loss' in log]
eval_steps = [log['step'] for log in train_logs if 'eval_loss' in log]
eval_accuracy = [log['eval_accuracy'] for log in train_logs if 'eval_accuracy' in log]

os.makedirs("plots/task3/task3_lora/guj", exist_ok=True)


plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label='Train Loss', marker='o')
plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='x')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Steps')
plt.legend()
plt.grid()
plt.savefig("plots/task3/task3_lora/guj/training_validation_loss.png")
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(eval_steps, eval_accuracy, label='Validation Accuracy', marker='x', color='green')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Steps')
plt.legend()
plt.grid()
plt.savefig("plots/task3/task3_lora/guj/validation_accuracy.png")
plt.show()
