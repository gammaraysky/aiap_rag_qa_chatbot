from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.utils.inference_utils import get_tokenizer, get_retriever, get_model
from src.utils.finetune_utils import Seq2SeqDataset


def finetune_rag_generator(training_data_path):

    tokenizer = get_tokenizer()
    retriever = get_retriever
    model = get_model(retriever=retriever)
    train_dataset = Seq2SeqDataset(tokenizer=tokenizer, data_dir=training_data_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(output_dir="./models/fine-tuned")

    return model
