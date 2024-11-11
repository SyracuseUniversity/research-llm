# train_model.py

from transformers import Trainer, TrainingArguments, LLaMAModel

def load_model(model_path):
    """Loads the LLaMA model."""
    model = LLaMAModel.from_pretrained(model_path)
    return model

def train_model(tokenized_dataset, model_path):
    """Trains the LLaMA model on the tokenized dataset."""
    model = load_model(model_path)
    training_args = TrainingArguments(
        output_dir="./llama_summarizer",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
