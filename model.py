import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DebertaV2ForTokenClassification, TrainingArguments, Trainer, DebertaTokenizer
import seqeval

class Model():

    def __init__(self, params: dict=None):
        
        # TODO Import roBERTa model for Token Classification.

        self.deberta = DebertaV2ForTokenClassification.from_pretrained('microsoft/deberta-v3-base')
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v3-base')
        
    def compute_metrics(eval_pred):
        metric = seqeval.metrics.F1Score()
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, train_data: pd.DataFrame, train_target: np.array, test_data: pd.DataFrame, test_target: np.array) -> None:
        '''
        Trains the model.
        '''
        training_args = TrainingArguments(
                                        output_dir="deberta",
                                        learning_rate=2e-5,
                                        per_device_train_batch_size=16,
                                        per_device_eval_batch_size=16,
                                        num_train_epochs=5,
                                        weight_decay=0.01,
                                        evaluation_strategy="epoch",
                                        save_strategy="epoch",
                                        load_best_model_at_end=True,
                                    )
        trainer = Trainer(model=self.deberta, 
                          training_args=training_args,
                          tokenizer=self.tokenizer,
                          train_dataset=train_data,
                          test_dataset=test_data,
                          compute_metrics=self.compute_metrics)
        
        trainer.train()
        return 'success'


    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Predicts the data.
        '''
        return self.model.predict(data)
