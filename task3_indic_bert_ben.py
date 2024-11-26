# Import necessary libraries

# !pip install transformers
# !pip install evaluate 

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import os 


df_train = pd.read_csv("eng_ben_train.csv")
df_val = pd.read_csv("eng_ben_val.csv")


train_texts = df_train['eng_text'].tolist()
train_labels = df_train['sentiment'].tolist()


val_texts = df_val['eng_text'].tolist()
val_labels = df_val['sentiment'].tolist()


ben_texts = df_val['ben_text'].tolist()
ben_labels = df_val['sentiment'].tolist()

model_name = "ai4bharat/indic-bert" 
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )


train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
ben_dataset = Dataset.from_dict({'text': ben_texts, 'labels': ben_labels})


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
ben_dataset = ben_dataset.map(tokenize_function, batched=True)


train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
ben_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


training_args = TrainingArguments(
    output_dir='results/indic_bert',
    num_train_epochs=40,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=500,
    logging_dir='logs/indic_bert',
    logging_steps=10,
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


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


trainer = Trainer(
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


print("Evaluation on Bengali dataset:")
ben_results = trainer.predict(ben_dataset)
ben_predictions = np.argmax(ben_results.predictions, axis=-1)
ben_accuracy = metric.compute(predictions=ben_predictions, references=ben_results.label_ids)
print(f"Accuracy on Bengali test data: {ben_accuracy['accuracy']}")

os.makedirs("plots/task3/task3_indic_bert", exist_ok=True)


train_logs = trainer.state.log_history


train_loss = [log['loss'] for log in train_logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in train_logs if 'eval_loss' in log]
steps = [log['step'] for log in train_logs if 'loss' in log]
eval_steps = [log['step'] for log in train_logs if 'eval_loss' in log]
eval_accuracy = [log['eval_accuracy'] for log in train_logs if 'eval_accuracy' in log]


plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label='Train Loss', marker='o')
plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='x')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Steps')
plt.legend()
plt.grid()
plt.savefig("plots/task3/task3_indic_bert/training_validation_loss.png") 
plt.show()
print("Train-Val Loss Saved!!")

plt.figure(figsize=(12, 6))
plt.plot(eval_steps, eval_accuracy, label='Validation Accuracy', marker='x', color='green')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Steps')
plt.legend()
plt.grid()
plt.savefig("plots/task3/task3_indic_bert/validation_accuracy.png") 
plt.show()
print("Val Acc Saved!!")