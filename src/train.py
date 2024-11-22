from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from classes.data_loader import DataLoader
from peft import LoraConfig, get_peft_model

def load_training_data(split='training'):

    dl = DataLoader()

    training = dl.load_dataset(split)
    train_unique_keys = set(training.keys())

    problems = []
    solutions = []

    for problem_id in list(train_unique_keys):
        inputs = ""
        outputs = ""
        for idx, example in enumerate(training[problem_id]['train_examples']):
            inputs += (f"Input {idx+1}: {example['input']}\n")
            inputs += (f"Output {idx+1}: {example['output']}\n")
        inputs += (f"Input {idx+2}: {training[problem_id]['test_input']}\n")
        inputs += (f"Output {idx+2}: ")

        outputs += (f"{training[problem_id]['solution'][0]}\n")

        problems.append(inputs)
        solutions.append(outputs)

    data = [{"input": inp, "target": sol} for inp, sol in zip(inputs, solutions)]
    dataset = Dataset.from_list(data)

    shuffled_dataset = dataset.shuffle(seed=42) 

    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)

    # Combine into final splits
    final_splits = {
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }

    # Access splits
    train_dataset = final_splits['train']
    val_dataset = final_splits['validation']
    test_dataset = final_splits['test']

    return train_dataset, val_dataset, test_dataset

def preprocess_dataset(examples, tokenizer):
    model_inputs = tokenizer(examples["input"])
    labels = tokenizer(examples["target"])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(model, tokenizer, train_dataset, val_dataset):

    train_dataset_tok = train_dataset.map(lambda x: preprocess_dataset(x, tokenizer), batched=True)
    val_dataset_tok = val_dataset.map(lambda x: preprocess_dataset(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,                      # Low-rank size
        lora_alpha=32,            # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
        lora_dropout=0,         # Dropout for LoRA layers
        bias="none",              # Type of bias adjustment
        task_type="CAUSAL_LM",    # Task type
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,  # Limits the number of checkpoints
        logging_dir='./logs',  # Directory for logs
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tok,
        eval_dataset=val_dataset_tok,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

def main():

    model_name = "meta-llama/Llama-3.2-1B-Instruct" 

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded")
    
    print("Loading training data...")
    train_dataset, val_dataset, test_dataset = load_training_data(split='training')
    print("Data loaded")

    print("Training model...")
    train_model(model, tokenizer, train_dataset, val_dataset)