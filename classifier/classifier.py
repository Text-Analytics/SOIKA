from transformers import pipeline
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

class TextClassifier:
    def __init__(self, repository_id:str = "Sandrro/cc-model", number_of_categories:int = 1, device_type = None):
          self.REP_ID = repository_id
          self.CATS_NUM = number_of_categories
          self.classifier = pipeline("text-classification", model=self.REP_ID, tokenizer='cointegrated/rubert-tiny2', max_length=2048, truncation=True, device=device_type)

    def run(self, t)-> list[pd.Series]:
        pred = pd.DataFrame(self.classifier(t, top_k=self.CATS_NUM))
        self.classifier.call_count = 0 #else warnings
        if self.CATS_NUM > 1:
            cats = ', '.join(pred['label'].tolist())
            probs = ', '.join(pred['score'].round(3).astype(str).tolist())
        else:
            cats = pred['label'][0]
            probs = pred['score'].round(3).astype(str)[0]
        return [cats, probs]