import torch
from transformers import AutoTokenizer, BloomForSequenceClassification
from peft import PeftModel, PeftConfig
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import os


df_val = pd.read_csv("eng_ben_val.csv")


val_texts = df_val['eng_text'].tolist()
val_labels = df_val['sentiment'].tolist()


ben_texts = df_val['ben_text'].tolist()
ben_labels = df_val['sentiment'].tolist()

checkpoint_path = "/home/pooja/shashwat/LLM-Project/results/lora/checkpoint-1890"


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)


base_model = BloomForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(set(val_labels)),  
    load_in_8bit=True                
)


peft_config = PeftConfig.from_pretrained(checkpoint_path)
model = PeftModel.from_pretrained(base_model, checkpoint_path)


model.eval()


metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=68, 
        return_tensors="pt"
    )


val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels}).map(tokenize_function, batched=True)
ben_dataset = Dataset.from_dict({'text': ben_texts, 'labels': ben_labels}).map(tokenize_function, batched=True)


val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
ben_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


from torch.utils.data import DataLoader

val_loader = DataLoader(val_dataset, batch_size=32)
ben_loader = DataLoader(ben_dataset, batch_size=32)

def evaluate_model(dataloader, dataset_name):
    all_predictions = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            attention_mask = batch["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
    
    return all_predictions, all_labels


val_predictions, val_labels = evaluate_model(val_loader, "English Validation Set")


ben_predictions, ben_labels = evaluate_model(ben_loader, "Bengali Validation Set")


val_accuracy = metric.compute(predictions=val_predictions, references=val_labels)["accuracy"]
print(f"Accuracy on English Validation Set: {val_accuracy}")
print("Classification Report (English):")
print(classification_report(val_labels, val_predictions))

ben_accuracy = metric.compute(predictions=ben_predictions, references=ben_labels)["accuracy"]
print(f"Accuracy on Bengali Validation Set: {ben_accuracy}")
print("Classification Report (Bengali):")
print(classification_report(ben_labels, ben_predictions))


os.makedirs("results/lora/ben_inference", exist_ok=True)

pd.DataFrame({
    "Text": val_texts,
    "True Label": val_labels,
    "Predicted Label": val_predictions
}).to_csv("results/lora/ben_inference/english_validation_results.csv", index=False)

pd.DataFrame({
    "Text": ben_texts,
    "True Label": ben_labels,
    "Predicted Label": ben_predictions
}).to_csv("results/lora/ben_inference/bengali_validation_results.csv", index=False)


os.makedirs("plots/task3/task3_lora/ben_inference", exist_ok=True)


ConfusionMatrixDisplay.from_predictions(val_labels, val_predictions)
plt.title("Confusion Matrix (English Validation Set)")
plt.savefig("plots/task3/task3_lora/ben_inference/confusion_matrix_english.png")
plt.show()


ConfusionMatrixDisplay.from_predictions(ben_labels, ben_predictions)
plt.title("Confusion Matrix (Bengali Validation Set)")
plt.savefig("plots/task3/task3_lora/ben_inference/confusion_matrix_bengali.png")
plt.show()





val_data_path = "/home/pooja/shashwat/LLM-Project/eng_guj_val.csv"
checkpoint_path = "/home/pooja/shashwat/LLM-Project/results/lora/guj/checkpoint-6590"

results_dir = "/home/pooja/shashwat/LLM-Project/results/lora/guj_inference"
plots_dir = "/home/pooja/shashwat/LLM-Project/plots/task3/task3_lora/guj_inference"


df_val = pd.read_csv(val_data_path)


unique_labels = df_val['sentiment'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
print("Label mapping:", label_mapping)


df_val['sentiment'] = df_val['sentiment'].map(label_mapping)


english_texts = df_val['eng_text'].tolist()
english_labels = df_val['sentiment'].tolist()
guj_texts = df_val['guj_text'].tolist()
guj_labels = df_val['sentiment'].tolist()


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)


base_model = BloomForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_mapping),
    device_map="auto",
    load_in_8bit=True
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval() 


metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=67, 
        return_tensors="pt"
    )

english_dataset = Dataset.from_dict({'text': english_texts, 'labels': english_labels}).map(tokenize_function, batched=True)
english_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

guj_dataset = Dataset.from_dict({'text': guj_texts, 'labels': guj_labels}).map(tokenize_function, batched=True)
guj_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


from torch.utils.data import DataLoader

english_loader = DataLoader(english_dataset, batch_size=32)
guj_loader = DataLoader(guj_dataset, batch_size=32)

def evaluate_model(dataloader, dataset_name):
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            attention_mask = batch["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            
            all_predictions.extend(predictions)
            all_labels.extend(torch.tensor(labels).numpy())
    
    return all_predictions, all_labels

def evaluate_and_save_results(loader, dataset_name, texts, labels, results_dir, plots_dir, label_mapping):
   
    predictions, true_labels = evaluate_model(loader, dataset_name)

    
    accuracy = metric.compute(predictions=predictions, references=true_labels)["accuracy"]
    print(f"Accuracy on {dataset_name}: {accuracy}")
    print(f"Classification Report ({dataset_name}):")
    print(classification_report(true_labels, predictions, target_names=list(label_mapping.keys())))

    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

   
    results_path = os.path.join(results_dir, f"{dataset_name}_validation_results.csv")
    pd.DataFrame({
        "Text": texts,
        "True Label": true_labels,
        "Predicted Label": predictions
    }).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    
    confusion_matrix_path = os.path.join(plots_dir, f"confusion_matrix_{dataset_name}.png")
    ConfusionMatrixDisplay.from_predictions(
        true_labels, predictions, display_labels=list(label_mapping.keys())
    )
    plt.title(f"Confusion Matrix ({dataset_name} Validation Set)")
    plt.savefig(confusion_matrix_path)
    plt.show()
    print(f"Confusion matrix saved to {confusion_matrix_path}")


evaluate_and_save_results(
    loader=english_loader,
    dataset_name="English",
    texts=english_texts,
    labels=english_labels,
    results_dir=results_dir,
    plots_dir=plots_dir,
    label_mapping=label_mapping
)

evaluate_and_save_results(
    loader=guj_loader,
    dataset_name="Gujarati",
    texts=guj_texts,
    labels=guj_labels,
    results_dir=results_dir,
    plots_dir=plots_dir,
    label_mapping=label_mapping
)
