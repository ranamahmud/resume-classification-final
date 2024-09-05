# Rational Behind the model
From the unigram and bigram model we fitted the following models in our data for classifying resumes. But didn't got good performance. LogisticRegression, RidgeClassifier, KNeighborsClassifier,RandomForestClassifier,LinearSVC,SGDClassifier,NearestCentroid,ComplementNB
We didn't good performance for classifying Resume as all the models had accuracy, f1 score less than 0.70.
As classical machine learning models don't perform well on resume classification tasks, I've experimented with finetuning distilbert-base-uncased. The resume dataset had a class imbalance problem. For this reason, I've chosen the f1 score to measure model performance based on a valid set f1 score of 0.80 and test f1 score of 0.81, validation accuracy of 0.81, test data set accuracy of 0.82. This model generalized well on unseen data. I've decided to use this model for the final terminal application.
# Running Instructions
You need to have a cuda enabled nvidia gpu installed to run the script.
requirements.txt will install necessary cuda libraries for PyTorch

Tested on Python version 3.12.3
# Setup Python Environment
python -m venv venv
# activate it
on linux run
```
. venv/source/activate
```
On windows run

```
.\venv\Scripts\activate
```
# Install the requirements
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
```

Keep the model folder and inference.py, id2label.json files in the same directory
Then run inference.py from terminal this way to get inference and output categorized_resumes.csv
Outputs will be generated in output folder in the same directory where the script is run.
Example command to generate inference
python inference.py data_folder
```
python inference.py "./data/"
 ```