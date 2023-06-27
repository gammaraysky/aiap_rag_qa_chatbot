from src.utils.inference_utils import get_tokenizer, get_retriever, get_model
from src.utils.convert_custom_csv import csv_to_dataset
import torch


def ask_question(
    question, tuned_model=None, custom_csv=False, csv_path=None, output_dir=None
):
    tokenizer = get_tokenizer()
    if custom_csv:
        dataset = csv_to_dataset(csv_path, output_dir)
        retriever = get_retriever(dataset=dataset)
    else:
        retriever = get_retriever()
    if tuned_model:
        model = tuned_model
    else:
        model = get_model(retriever)
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    return generated_string


if __name__ == "__main__":
    question = input("Question about AIAP: ")
    answer = ask_question(question)
    print(f"Answer: {answer}")
