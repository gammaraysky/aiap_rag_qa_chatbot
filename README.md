# AIAP RAG Chatbot
This repo is for AI Singapore's AIAP Batch 12 Group Presentation Topic: "Transfer Learning in NLP". 

In this repo, we attempt to create a custom chatbot using pretrained models. The model we use is the RAG model.

Paper: https://arxiv.org/abs/2005.11401


# Requirements
- [Transformers](https://github.com/huggingface/transformers) (HuggingFace Transformers Library)
- [Datasets](https://github.com/huggingface/datasets) (HuggingFace Datasets Library)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [faiss-cpu](https://github.com/facebookresearch/faiss) (Library for efficient similarity search and clustering of dense vectors)


# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this repo, assuming you want to implement an Open Domain Question Answering (ODQA) chatbot using a pretrained RAG model and your own custom dataset.
- Provide your custom dataset in a CSV format with each line containing the 'title' and 'text' of the dataset, tab delimited. Convert this csv file into the required Dataset format by calling the `convert_to_dataset` function from `convert_custom_csv.py` in the `utils` folder found in `src` folder.

```python
from src.utils.convert_custom_csv import convert_to_dataset
# dataset conversion
csv_path = './custom_csv/my_own_dataset.csv'
output_dir_path = './custom_dataset'
dataset = convert_to_dataset(csv_path, output_dir_path)
``` 


- The beauty of pretrained models is that you can now conduct inference without any fine tuning at all. From `src`  folder import the `ask_question` function in `inference.py`

```python
from src.inference import ask_question
# inference
question = 'How much is AIAP allowance?'
answer = ask_question(question)
print(answer)
```

- If you would like to fine-tune the generator, you will need to provide two files in `training_data` folder. The first file should be named train.source and should contain questions in each newline. The second file should be named train.target and contains the answer (ground truth) for each corresponding question in train.source. From the `finetune.py` file in the `utils` folder, import `finetune_rag_generator` function
```python
from src.finetune import finetune_rag_generator
# finetune model
training_data_path = './training_data'
tuned_model = finetune_rag_generator(training_data_path)

# inference with tuned model
question = 'What if I decide to drop out of AIAP?'
answer = ask_question(question, tuned_model=tuned_model)
print(answer)
```

**A demo notebook has been provided to illustrate the above steps.**


# In Details
```
├──  custom_csv
│    └── aiap_ds.csv  - custom AI Singapore Apprenticeship Program csv.
│
│
├──  custom_dataset
│    ├── my_knowledge_dataset  - folder containing your converted dataset.
│    └── my_knowledge_dataset_hnsw_index.faiss  - faiss index of your converted dataset.
│
│
├──  training_data  
│    ├── training.source  - training set containing questions to train model
│    └── training.target  - training set containing answers to training.source
│
│
├──  notebooks
│   ├── aisg_faq_scrape.ipynb  - notebook on scraping AISG faq
│   ├── data_cleaning.ipynb  - notebook on cleaning scraped data
│   └── data_augmentation.ipynb  - notebook on augmenting scraped data
│
│
├──  scraped_data
│   ├── augmented_data.csv  - csv file of augmented scraped data
│   └── scraped_data.csv  - csv file of scraped data
│
│  
├──  src
│   ├── finetune.py  - this file contains the tuner for the 
│   ├── inference.py  - this file contains the inference process.
│   └── utils
│     ├── convert_custom_csv.py  - this file contains the converter for custom csv file
│     ├── finetune_utils.py  - this file contain helper function for the finetune function
│     └── inference_utils.py  - this file contain helper functions for the inference function
```


# Acknowledgments
Please refer to AUTHORS.


