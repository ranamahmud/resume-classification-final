from pypdf import PdfReader
import os
import torch
from tqdm import tqdm
import argparse
import shutil
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def read_pdf(file):
    reader = PdfReader(file)
    text = [page.extract_text() for page in reader.pages][100:]
    # remove first line to remove data leackage
    return ' '.join(text)

# read all files


def get_pdf_files_from_directory(directory):
    """Recursively get all PDF files from the given directory."""
    pdf_files = []

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):  # Check for PDF files
                pdf_files.append(os.path.join(root, file))

    return pdf_files




def classify_resume(resume):
    
    # Load the dictionary from the JSON file
    with open('id2label.json', 'r', encoding="utf-8") as json_file:
        id2label = json.load(json_file)
    for key,value in id2label.items():
        folder_path = os.path.join("output", key)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    # Tokenize the input text
    inputs = tokenizer(resume, truncation=True,
                       padding='max_length', max_length=512, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label (output logits -> predicted class)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return id2label[str(predicted_class_id)]

# make dataframe and save


def inference(folder_path):
    
    predictions = []
    files = get_pdf_files_from_directory(folder_path)
    for file in tqdm(files):
        pred = classify_resume(file)
        predictions.append(pred)
        # copy to folder
        destination_folder = os.path.join("output",pred)
        shutil.copy(file, destination_folder)
        
    df = pd.DataFrame({"FileName":files, "Category":predictions})
    df.to_csv("categorized_resumes.csv", index = False)

if __name__ == "__main__":

    path = "./resume_classification_model"
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("folder", type=str, default="./",
                        help="The folder containing the PDF resumes")
    args = parser.parse_args()
    if args.folder:
        inference(args.folder)
    else:
        print("Print pass argument parser")
