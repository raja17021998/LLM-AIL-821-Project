import torch
from transformers import AutoTokenizer, BloomForSequenceClassification
from datasets import Dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import evaluate
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from transformers import BitsAndBytesConfig


df_val = pd.read_csv("eng_ben_val.csv")

val_texts = df_val['eng_text'].tolist()
val_labels = df_val['sentiment'].tolist()

ben_texts = df_val['ben_text'].tolist()
ben_labels = df_val['sentiment'].tolist()



# Define a configuration for 8-bit quantization with FP32 CPU offload
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,                        # Enable 8-bit quantization
#     llm_int8_enable_fp32_cpu_offload=True     # Offload FP32 modules to CPU if needed
# )

# # Load the tokenizer
# model_name = "bigscience/bloom-1b7"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the model with quantization configuration
# model = BloomForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=len(set(val_labels)),          # Number of unique labels
#     # Use the quantization config
#     device_map="auto"                         # Automatically map layers to available devices
# ).eval()


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BloomForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(set(val_labels)),
    load_in_8bit=True,
    device_map="auto"
).eval()


metric = evaluate.load("accuracy")

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=68, 
        return_tensors="pt"
    )
torch.cuda.empty_cache()

val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels}).map(tokenize_function, batched=True)
ben_dataset = Dataset.from_dict({'text': ben_texts, 'labels': ben_labels}).map(tokenize_function, batched=True)

val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
ben_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


def extract_hidden_states(dataset, layer_index=5):
    features, labels = [], []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting hidden states from layer {layer_index}"):
            inputs = {key: val.to(model.device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            labels.extend(batch['labels'].cpu().numpy())
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index]
            sentence_embeddings = hidden_states.mean(dim=1)
            features.append(sentence_embeddings.cpu().numpy())

    return np.concatenate(features, axis=0), np.array(labels)


def short_circuit_layer(model, layer_to_zero):
    def hook_function(module, input, output):
        return torch.zeros_like(output) if isinstance(output, torch.Tensor) else output
    handle = model.transformer.h[layer_to_zero].register_forward_hook(hook_function)
    return handle

def bypass_layer(model, layer_to_bypass):

    def hook_function(module, input, output):
        if isinstance(output, tuple):
            return tuple(input[0] if isinstance(input, tuple) else input for _ in output)
        return input[0] if isinstance(input, tuple) else input


    handle = model.transformer.h[layer_to_bypass].register_forward_hook(hook_function)
    return handle



n_layers = len(model.transformer.h)-1

bypass_accuracies = []
bypass_accuracies_short = []
short_circuit_accuracies = []
short_circuit_accuracies_short = []


for i, layer in enumerate(tqdm(range(n_layers), desc="Layers", total=n_layers, unit="layer")):
    specific_layer = i  
    
    val_features, val_labels = extract_hidden_states(val_dataset, layer_index=specific_layer)
    
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(val_features, val_labels)
    
    val_predictions = clf.predict(val_features)
    val_accuracy = metric.compute(predictions=val_predictions, references=val_labels)['accuracy']
    bypass_accuracies.append(val_accuracy)
    short_circuit_accuracies.append(val_accuracy)
    
    handle_bypass = bypass_layer(model, layer_to_bypass=specific_layer)
    val_features_bypass, val_labels_bypass = extract_hidden_states(val_dataset, layer_index=specific_layer)
    handle_bypass.remove()

    val_predictions_bypass = clf.predict(val_features_bypass)
    val_accuracy_bypass = metric.compute(predictions=val_predictions_bypass, references=val_labels_bypass)['accuracy']
    bypass_accuracies_short.append(val_accuracy_bypass)
    

    handle_short_circuit = short_circuit_layer(model, layer_to_zero=specific_layer)
    val_features_short, val_labels_short = extract_hidden_states(val_dataset, layer_index=specific_layer)
    handle_short_circuit.remove()
    

    val_predictions_short = clf.predict(val_features_short)
    val_accuracy_short = metric.compute(predictions=val_predictions_short, references=val_labels_short)['accuracy']
    short_circuit_accuracies_short.append(val_accuracy_short)

output_dir = "/home/pooja/shashwat/LLM-Project/plots/task2/probe_plots"
os.makedirs(output_dir, exist_ok=True)


layers = list(range(len(bypass_accuracies)))

# Plot 1: Bypass Without Modification
plt.figure(figsize=(12, 6))
plt.plot(layers, bypass_accuracies, label="Bypass Without Modification", marker="o", linestyle="-")
plt.xticks(layers)  # Set x-axis to display discrete layer numbers
plt.xlabel("Layer Index")
plt.ylabel("Validation Accuracy")
plt.title("Bypass Without Modification")
plt.legend()
plt.grid()

# Save and show the first plot
plot_path_bypass_without = os.path.join(output_dir, "bypass_without_modification.png")
plt.savefig(plot_path_bypass_without, format="png", dpi=300)
print(f"Plot saved to {plot_path_bypass_without}")
plt.show()

# Plot 2: Short-Circuit Without Modification
plt.figure(figsize=(12, 6))
plt.plot(layers, short_circuit_accuracies, label="Short-Circuit Without Modification", marker="x", linestyle="--")
plt.xticks(layers)  # Set x-axis to display discrete layer numbers
plt.xlabel("Layer Index")
plt.ylabel("Validation Accuracy")
plt.title("Short-Circuit Without Modification")
plt.legend()
plt.grid()


plot_path_short_circuit_without = os.path.join(output_dir, "short_circuit_without_modification.png")
plt.savefig(plot_path_short_circuit_without, format="png", dpi=300)
print(f"Plot saved to {plot_path_short_circuit_without}")
plt.show()

# Plot 3: Bypass With Modification
plt.figure(figsize=(12, 6))
plt.plot(layers, bypass_accuracies_short, label="Bypass With Modification", marker="o", linestyle="-")
plt.xticks(layers)  # Set x-axis to display discrete layer numbers
plt.xlabel("Layer Index")
plt.ylabel("Validation Accuracy")
plt.title("Bypass With Modification")
plt.legend()
plt.grid()


plot_path_bypass_with = os.path.join(output_dir, "bypass_with_modification.png")
plt.savefig(plot_path_bypass_with, format="png", dpi=300)
print(f"Plot saved to {plot_path_bypass_with}")
plt.show()

# Plot 4: Short-Circuit With Modification
plt.figure(figsize=(12, 6))
plt.plot(layers, short_circuit_accuracies_short, label="Short-Circuit With Modification", marker="x", linestyle="--")
plt.xticks(layers)  # Set x-axis to display discrete layer numbers
plt.xlabel("Layer Index")
plt.ylabel("Validation Accuracy")
plt.title("Short-Circuit With Modification")
plt.legend()
plt.grid()


plot_path_short_circuit_with = os.path.join(output_dir, "short_circuit_with_modification.png")
plt.savefig(plot_path_short_circuit_with, format="png", dpi=300)
print(f"Plot saved to {plot_path_short_circuit_with}")
plt.show()

