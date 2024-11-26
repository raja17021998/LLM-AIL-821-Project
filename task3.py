
import pandas as pd
from transformers import AutoTokenizer, BloomForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np
import torch
import bitsandbytes as bnb


df_train = pd.read_csv("eng_ben_train.csv")
df_val = pd.read_csv("eng_ben_val.csv")


train_texts = df_train['eng_text'].tolist()
train_labels = df_train['sentiment'].tolist()

val_texts = df_val['eng_text'].tolist()
val_labels = df_val['sentiment'].tolist()


ben_texts = df_val['ben_text'].tolist()
ben_labels = df_val['sentiment'].tolist()


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length",  
        truncation=True,       
        max_length=68,        
        return_tensors="pt"   
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
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=500,
    logging_dir='./logs',
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


model = BloomForSequenceClassification.from_pretrained(model_name, num_labels=2)


optimizer = bnb.optim.Adam8bit(model.parameters(), lr=5e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)  
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
