from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from classes.data_loader import DataLoader
from peft import LoraConfig, get_peft_model
import json

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_training_data(config, split='training'):

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

    data = [{"input": inp, "target": sol} for inp, sol in zip(problems, solutions)] # Corrected line: use problems and solutions lists
    dataset = Dataset.from_list(data)

    shuffled_dataset = dataset.shuffle(seed=config['shuffle_seed'])

    train_test = shuffled_dataset.train_test_split(test_size=config['train_test_split_ratio'], seed=config['shuffle_seed']) # Use shuffled dataset
    val_test = train_test['test'].train_test_split(test_size=config['val_test_split_ratio'], seed=config['shuffle_seed'])

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
    # You can set a max_length value appropriate for your data/model.
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 512  
    model_inputs = tokenizer(
        examples["input"],
        padding="max_length",  # pad to max_length
        truncation=True,       # truncate sequences longer than max_length
        max_length=max_length
    )
    labels = tokenizer(
        examples["target"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(model, tokenizer, train_dataset, val_dataset):

    train_dataset_tok = train_dataset.map(lambda x: preprocess_dataset(x, tokenizer), batched=True)
    val_dataset_tok = val_dataset.map(lambda x: preprocess_dataset(x, tokenizer), batched=True)

    # Check if tokenizer has a pad token and set one if not
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=15,
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

    config = load_config()
    model_name = config['model_name']

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded")

    print("Loading training data...")
    train_dataset, val_dataset, test_dataset = load_training_data(config, split='training')
    print("Data loaded")

    print("Training model...")
    train_model(model, tokenizer, train_dataset, val_dataset)

if __name__ == "__main__":
    main()