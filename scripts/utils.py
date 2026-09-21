"""
Extra functions to train a DINOv2 and ResNet18 models to classify experimental Chagas Disease from stereomicroscopic images
"""

############################
#Loading libraries
import json
import multiprocessing
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix,f1_score,precision_score,
                             recall_score,roc_auc_score,precision_recall_fscore_support,)
from transformers import (Trainer,TrainingArguments,)
from torch.utils.data import Dataset

def set_seed(seed: int) -> None:
    """
    Function to set random seeds to improve reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cross_validation_metadata(train_csv_path: str,val_csv_path: str,test_csv_path: str,) -> pd.DataFrame:
    """
    Cross-validation
    """

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df["original_split"] = "train"
    val_df["original_split"] = "validation"
    test_df["original_split"] = "test"

    dataframe = pd.concat([train_df, val_df, test_df],ignore_index=True,)
    dataframe = dataframe.dropna(subset=["image_path","label","individual_id",]).copy()
    dataframe["image_path"] = (dataframe["image_path"].astype(str).str.strip())
    dataframe["label"] = (dataframe["label"].astype(str).str.strip())
    dataframe["individual_id"] = (dataframe["individual_id"].astype(str).str.strip())
    dataframe = dataframe.drop_duplicates(subset=["image_path","individual_id",]).reset_index(drop=True) #drop duplicates
    return dataframe

def verify_individual_labels(dataframe: pd.DataFrame,) -> None:
    """
    Verify that all of a individual's images have the same label.
    """
    labels_per_patient = (dataframe.groupby("individual_id")["label"].nunique())
    inconsistent_patients = labels_per_patient[labels_per_patient > 1]
    if not inconsistent_patients.empty:
        individual_ids = inconsistent_patients.index.tolist()
        raise ValueError("Los siguientes pacientes tienen más de una etiqueta: "f"{individual_ids[:10]}")
    print("\nTodos los pacientes tienen una única etiqueta.")

def create_global_label_mapping(dataframe: pd.DataFrame,):
    """
    Function to create a single class mapping for all folds.
    """
    class_names = sorted(dataframe["label"].unique().tolist())
    label2id = {class_name: class_id for class_id, class_name in enumerate(class_names)}
    id2label = {class_id: class_name for class_name, class_id in label2id.items()}
    dataframe = dataframe.copy()
    dataframe["label_id"] = (dataframe["label"].map(label2id).astype(int))
    return dataframe, class_names, label2id, id2label

def print_individual_distribution(dataframe: pd.DataFrame,) -> None:
    """
    Shows the distribution by images and individuals.
    """
    print("\n Distribution by images")
    print(dataframe["label"].value_counts().sort_index())
    individual_level_df = (dataframe[["individual_id", "label"]].drop_duplicates(subset=["individual_id"]))

    print("\nDistribution by individuals")
    print(individual_level_df["label"].value_counts().sort_index())

class HistopathologyDataset(Dataset):
    """
    PyTorch dataset for loading histopathological images.
    """

    def __init__(self,dataframe: pd.DataFrame,transform=None,):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self,index: int,) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = row["image_path"]
        label_id = int(row["label_id"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"The image could not be loaded.:\n{image_path}") from error

        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label_id,dtype=torch.long,),}

def collate_fn(batch: List[Dict[str, torch.Tensor]],) -> Dict[str, torch.Tensor]:
    """
    Function to group images and labels to create a batch.
    """
    pixel_values = torch.stack([sample["pixel_values"] for sample in batch])
    labels = torch.stack([sample["labels"] for sample in batch])
    return {"pixel_values": pixel_values,"labels": labels,}

def print_trainable_parameters(model) -> None:
    """
    Parameters info
    """
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum( parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_parameters = (total_parameters - trainable_parameters)
    trainable_percentage = (100.0 * trainable_parameters / total_parameters )

    print("\nModel parameters")
    print("=====================")
    print(f"Total parameters: " f"{total_parameters:,}")
    print(f"Frozen parameters: "f"{frozen_parameters:,}")
    print(f"Trainable parameters: "f"{trainable_parameters:,}")
    print(f"trainable percentage: "f"{trainable_percentage:.6f}%")
    print("\nTrainable layers:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}: "f"{parameter.numel():,}")


def stable_softmax(logits: np.ndarray,) -> np.ndarray:
    """
    Function to compute softmax
    """
    shifted_logits = logits - np.max(logits,axis=1,keepdims=True,)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials,axis=1,keepdims=True,)
    return probabilities

def extract_logits(predictions,) -> np.ndarray:
    """
    Function to extract logits
    """
    if isinstance(predictions, tuple):
        return predictions[0]
    return predictions


def build_compute_metrics(num_classes: int,):
    """
    Function to compute metrics
    """

    def compute_metrics(eval_prediction,) -> Dict[str, float]:
        logits = extract_logits(eval_prediction.predictions)
        labels = eval_prediction.label_ids
        probabilities = stable_softmax(logits)
        predictions = np.argmax(probabilities,axis=1,)
        metrics = {"accuracy": accuracy_score(labels,predictions,),
                   "balanced_accuracy": balanced_accuracy_score(labels,predictions,),
                   "precision_macro": precision_score(labels,predictions,average="macro",zero_division=0,),
                   "recall_macro": recall_score(labels,predictions,average="macro",zero_division=0,),
                   "f1_macro": f1_score(labels,predictions,average="macro",zero_division=0,),
                   "f1_weighted": f1_score(labels,predictions,average="weighted",zero_division=0,),}

        if num_classes == 2: #binary classification
            try:
                metrics["auc"] = roc_auc_score(labels,probabilities[:, 1],)
            except ValueError:
                metrics["auc"] = float("nan")
        return metrics
    return compute_metrics

def calculate_fold_class_weights(train_df: pd.DataFrame,num_classes: int,) -> torch.Tensor:
    """
    Calculate class weights using only the internal training set of the current fold.
    """
    class_counts = (train_df["label_id"].value_counts().reindex(range(num_classes),fill_value=0,).sort_index())
    if (class_counts == 0).any():
        raise ValueError("Al menos una clase no aparece en el entrenamiento " f"interno. Conteos: {class_counts.to_dict()}")

    counts = class_counts.to_numpy(dtype=np.float32)
    total_samples = counts.sum()
    weights = (total_samples/ (num_classes * counts))
    class_weights = torch.tensor(weights,dtype=torch.float32,)
    print("Training counts:",class_counts.to_dict(),)
    print("Weights:",class_weights.tolist(),)
    return class_weights

class WeightedCrossEntropyTrainer(Trainer):
    """
    Trainer with weighted cross-entropy.
    """

    def __init__(self,*args, class_weights: torch.Tensor,**kwargs,):
        super().__init__(*args,**kwargs,)
        self.class_weights = class_weights

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None,):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(device=logits.device,dtype=logits.dtype,))
        loss = loss_function(logits,labels,)

        if return_outputs:
            return loss, outputs
        return loss


def calculate_fold_metrics(true_labels: np.ndarray,predicted_labels: np.ndarray,probabilities: np.ndarray,class_names: list,) -> dict:
    """
    Function to calculate global and per-class metrics for a fold.
    """

    precision, recall, f1, support = (precision_recall_fscore_support(true_labels,predicted_labels,labels=np.arange(len(class_names)),zero_division=0,))

    metrics = {
        "accuracy": accuracy_score(true_labels,predicted_labels,),
        "balanced_accuracy": balanced_accuracy_score(true_labels,predicted_labels,),
        "f1_macro": f1_score(true_labels,predicted_labels,average="macro",zero_division=0,),}

    if len(class_names) == 2:
        try:
            metrics["auc"] = roc_auc_score(true_labels,probabilities[:, 1],)
        except ValueError:
            metrics["auc"] = np.nan

    for class_id, class_name in enumerate(class_names):
        metrics[f"precision_{class_name}"] = precision[class_id]
        metrics[f"recall_{class_name}"] = recall[class_id]
        metrics[f"f1_{class_name}"] = f1[class_id]
        metrics[f"support_{class_name}"] = int(support[class_id])
    return metrics


def create_inner_train_validation_split(outer_train_df: pd.DataFrame,n_splits: int,random_state: int,):
    """
    Split the training set into:
    - internal training; 
    - internal validation. 
    Respect `individual_id` and try to preserve the class distribution.
    """
    inner_splitter = StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=random_state,)
    split_iterator = inner_splitter.split(X=outer_train_df,y=outer_train_df["label_id"],groups=outer_train_df["individual_id"],)
    inner_train_indices, inner_val_indices = next(split_iterator)
    inner_train_df = outer_train_df.iloc[inner_train_indices].reset_index(drop=True)
    inner_val_df = outer_train_df.iloc[inner_val_indices].reset_index(drop=True)
    return inner_train_df, inner_val_df

def verify_fold_separation(train_df: pd.DataFrame,val_df: pd.DataFrame,test_df: pd.DataFrame,fold_number: int,) -> None:
    """
    Verify that no individual_id is shared between the train, validation, and test sets.
    """
    train_patients = set(train_df["individual_id"])
    val_patients = set(val_df["individual_id"])
    test_patients = set(test_df["individual_id"])
    assert train_patients.isdisjoint(val_patients), f"Fuga train-validation en fold {fold_number}"
    assert train_patients.isdisjoint(test_patients), f"Fuga train-test en fold {fold_number}"
    assert val_patients.isdisjoint(test_patients), f"Fuga validation-test en fold {fold_number}"
    print(f"Fold {fold_number}: correct patient separation.")

def create_fold_training_arguments(fold_output_dir,seed,LEARNING_RATE,
                                   WEIGHT_DECAY, BATCH_SIZE,NUM_EPOCHS,
                                   NUM_WORKERS) -> TrainingArguments:
    """
    Create independent TrainingArguments for a fold.
    """

    use_bf16 = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = (torch.cuda.is_available() and not use_bf16)

    return TrainingArguments(output_dir=fold_output_dir,
                             eval_strategy="epoch",
                             save_strategy="epoch",
                             logging_strategy="epoch",
                             learning_rate=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY,
                             per_device_train_batch_size=BATCH_SIZE,
                             per_device_eval_batch_size=BATCH_SIZE,
                             num_train_epochs=NUM_EPOCHS,
                             warmup_ratio=0.10,
                             load_best_model_at_end=True,
                             metric_for_best_model="f1_macro",
                             greater_is_better=True,
                             save_total_limit=1,
                             fp16=use_fp16,
                             bf16=use_bf16,
                             dataloader_num_workers=NUM_WORKERS,
                             remove_unused_columns=False,
                             report_to="none",
                             seed=seed,
                             data_seed=seed,)

def summarize_cross_validation_metrics(fold_metrics_df: pd.DataFrame,output_dir: str,) -> pd.DataFrame:
    """
    Function to summarize metrics
    """
    excluded_columns = {"fold","seed","n_train","n_validation","n_test",}
    metric_columns = [column for column in fold_metrics_df.columns if column not in excluded_columns and not column.startswith("support_")]
    summary_rows = []
    print("\n" + "=" * 70)
    print("----------- Summary cross-validation -----------")
    print("=" * 70)

    for metric_name in metric_columns:
        values = pd.to_numeric(fold_metrics_df[metric_name],errors="coerce",)
        mean_value = values.mean()
        std_value = values.std(ddof=1)
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": mean_value,
                "std": std_value,
                "mean_std": (f"{mean_value:.4f} ± " f"{std_value:.4f}"),
            }
        )

        print(f"{metric_name}: "f"{mean_value:.4f} ± {std_value:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir,"cross_validation_summary.csv",),index=False,)
    return summary_df

def calculate_global_oof_metrics(oof_predictions_df: pd.DataFrame,class_names: list,output_dir: str,) -> dict:
    """
    Calculate global metrics using all predictions (out-of-fold)
    """

    true_labels = oof_predictions_df["true_label_id"].to_numpy()
    predicted_labels = oof_predictions_df["predicted_label_id"].to_numpy()
    probabilities = np.column_stack(
        [
            oof_predictions_df[f"probability_{class_name}"].to_numpy()
            for class_name in class_names
        ]
    )

    global_metrics = calculate_fold_metrics(true_labels=true_labels,predicted_labels=predicted_labels,probabilities=probabilities,class_names=class_names,)
    print("\n" + "=" * 70)
    print("Global metrics OUT-OF-FOLD")
    print("=" * 70)

    for metric_name, metric_value in (global_metrics.items()):
        print(f"{metric_name}: {metric_value}")

    print("\nGlobal report")

    print(classification_report(true_labels,predicted_labels,labels=np.arange(len(class_names)),
                              target_names=class_names,digits=4,zero_division=0,))
    global_confusion_matrix = confusion_matrix(true_labels,predicted_labels,labels=np.arange(len(class_names)),)

    print("\nMatriz de confusión global")
    print(global_confusion_matrix)

    pd.DataFrame(global_confusion_matrix,
                 index=[f"actual_{class_name}" for class_name in class_names],
                 columns=[f"predicted_{class_name}" for class_name in class_names],).to_csv(os.path.join(output_dir,"global_oof_confusion_matrix.csv",))

    with open(os.path.join(output_dir,"global_oof_metrics.json",),"w",encoding="utf-8",) as file:
        json.dump({key: float(value) for key, value in global_metrics.items()},file,indent=4,)
    return global_metrics

def save_experiment_configuration(TRAIN_CSV_PATH,VAL_CSV_PATH,TEST_CSV_PATH,
                                  MODEL_NAME, SEED, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
                                  LEARNING_RATE, WEIGHT_DECAY,NUM_EPOCHS,EARLY_STOPPING_PATIENCE,
                                  WARMUP_RATIO,NUM_WORKERS,
                                  output_dir, 
                                  class_names, 
                                  label2id, 
                                  id2label,) -> None:
    """
    Function to save settings as json file
    """
    configuration = {
        "train_csv_path": TRAIN_CSV_PATH,
        "validation_csv_path": VAL_CSV_PATH,
        "test_csv_path": TEST_CSV_PATH,
        "model_name": MODEL_NAME,
        "class_names": class_names,
        "label2id": label2id,
        "id2label": {str(class_id): class_name for class_id, class_name in id2label.items()},
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": (GRADIENT_ACCUMULATION_STEPS),
        "effective_batch_size": (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": (EARLY_STOPPING_PATIENCE),
        "warmup_ratio": WARMUP_RATIO,
        "num_workers": NUM_WORKERS,}
    configuration_path = os.path.join(output_dir,"experiment_configuration.json",)
    with open(configuration_path,"w",encoding="utf-8",) as file:
        json.dump(configuration,file,indent=4,ensure_ascii=False)