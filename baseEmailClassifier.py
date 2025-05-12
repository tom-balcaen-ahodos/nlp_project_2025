import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding # Good practice to explicitly use data collator
)
import numpy as np
import evaluate # Import the evaluate library
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from typing import Dict, Optional, List

class BaseEmailClassifier:
    def __init__(
        self,
        token: Optional[str] = None,  # Hugging Face token for login
    ):
        if token:
            login(token=token)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    