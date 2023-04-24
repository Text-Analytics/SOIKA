from transformers import pipeline
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

REP_ID = "cc-model"

def category_predictor(text, classifier):
    pred = pd.DataFrame(classifier(text, top_k=3))
    classifier.call_count = 0
    cats = ', '.join(pred['label'].tolist())
    probs = ', '.join(pred['score'].round(3).astype(str).tolist())
    return [cats, probs]

def classify_text(path_to_excel):
    classifier = pipeline("text-classification", model=REP_ID, tokenizer=REP_ID, max_length=2048, truncation=True, device=0)
    df_predict = pd.read_excel(path_to_excel)
    df_predict[['cats','probs']] = pd.DataFrame(df_predict['Текст'].progress_map(lambda x: category_predictor(x, classifier)).to_list())
    df_predict['category'] = df_predict.cats.map(lambda x: x.split(', ')[0])
    df_predict['probability'] = df_predict.probs.map(lambda x: float(x.split(', ')[0]))

    return df_predict