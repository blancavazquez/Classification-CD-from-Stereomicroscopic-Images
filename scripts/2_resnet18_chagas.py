"""
Code for training a ResNet-18 model to classify experimental Chagas Disease from stereomicroscopic images
Stratified Group K-Fold + validación interna + OOF metrics
"""
############################
#Loading libraries
import multiprocessing
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from sklearn.metrics import (classification_report,confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import (CenterCrop,Compose,Normalize,
                                    RandomHorizontalFlip,RandomResizedCrop,RandomRotation,
                                    RandomVerticalFlip,Resize,ToTensor,)
from transformers import EarlyStoppingCallback, Trainer
from transformers.modeling_outputs import ImageClassifierOutput

from utils import (set_seed, load_cross_validation_metadata, verify_individual_labels,
                   create_global_label_mapping, print_individual_distribution,
                   HistopathologyDataset, collate_fn, print_trainable_parameters,
                   stable_softmax, extract_logits, build_compute_metrics,
                   calculate_fold_class_weights, calculate_fold_metrics,
                   create_inner_train_validation_split, WeightedCrossEntropyTrainer,
                   verify_fold_separation, create_fold_training_arguments,
                   summarize_cross_validation_metrics,calculate_global_oof_metrics,
                   save_experiment_configuration)
#########################
#Parameters for cross-validation
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
SEED = 7
MODEL_NAME = "resnet18_imagenet1k"
OUTPUT_DIR = "outputs/resnet18_stratified_group_kfold"

#Parameters for the model
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
WARMUP_RATIO = 0.10
NUM_WORKERS = 2
FREEZE_BACKBONE = True
USE_WEIGHTED_LOSS = True

IMAGE_SIZE = 224
RANDOM_CROP_MIN_SCALE = 0.75
ROTATION_DEGREES = 20

IMAGENET_MEAN = [0.485, 0.456, 0.406] #normalization by ImageNet-1K de ResNet18.
IMAGENET_STD = [0.229, 0.224, 0.225]

#########################
#Path of metadata
TRAIN_CSV_PATH = "../data_LLM/metadata_train.csv"
VAL_CSV_PATH = "../data_LLM/metadata_val.csv"
TEST_CSV_PATH = "../data_LLM/metadata_test.csv"
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_transforms():
    """
    Data augmentation
    """

    normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transforms = Compose(
        [
            RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE),scale=(RANDOM_CROP_MIN_SCALE, 1.0),ratio=(0.90, 1.10),),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=ROTATION_DEGREES),
            ToTensor(),
            normalize,
        ]
    )

    evaluation_transforms = Compose(
        [
            Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            CenterCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
            normalize,
        ]
    )

    return train_transforms, evaluation_transforms


class ResNet18ForImageClassification(nn.Module):
    """
    Function to use torchvision's ResNet18 with the Hugging Face Trainer. 
    The output contains the `.logits` attribute, just like Transformers classification models.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values)
        return ImageClassifierOutput(logits=logits)


def create_model(num_classes: int) -> ResNet18ForImageClassification:
    return ResNet18ForImageClassification(num_classes=num_classes,pretrained=True,)


def freeze_resnet18_backbone(model: ResNet18ForImageClassification) -> None:
    """Freeze the entire Resnet18 backbone and leave only the classification head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.resnet.fc.parameters():
        parameter.requires_grad = True

def run_stratified_group_cross_validation(dataframe: pd.DataFrame,class_names: List[str],id2label: Dict[int, str],
                                          train_transforms,evaluation_transforms,output_dir: str,):    
    """
    Execute Stratified Group K-Fold. 
    For each fold:
    1. Separate the external test set. 
    2. Separate an internal validation set from the external training set. 
    3. Calculate weights using only the internal training set. 
    4. Load a new model. 
    5. Freeze the backbone. 
    6. Train the head. 
    7. Evaluate on the external test set. 
    8. Save out-of-fold predictions.
        """

    os.makedirs(output_dir,exist_ok=True,)
    num_classes = len(class_names)
    outer_splitter = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS,shuffle=True,random_state=SEED,)
    all_fold_metrics = []
    all_oof_predictions = []
    split_iterator = outer_splitter.split(X=dataframe,y=dataframe["label_id"],groups=dataframe["individual_id"],)


    for fold_index, (outer_train_indices,outer_test_indices,) in enumerate(split_iterator,start=1,):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_index}/{N_OUTER_FOLDS}")
        print("=" * 70)
        fold_seed = SEED #+ fold_index
        set_seed(fold_seed)
        outer_train_df = dataframe.iloc[outer_train_indices].reset_index(drop=True)
        fold_test_df = dataframe.iloc[outer_test_indices].reset_index(drop=True)
        fold_train_df, fold_val_df = (create_inner_train_validation_split(outer_train_df=outer_train_df,n_splits=N_INNER_FOLDS,random_state=fold_seed,))
        verify_fold_separation(train_df=fold_train_df,val_df=fold_val_df,test_df=fold_test_df,fold_number=fold_index,)

        print("\nDistribución del fold")
        print("Train:",fold_train_df["label"].value_counts().to_dict(),)
        print("Validation:",fold_val_df["label"].value_counts().to_dict(),)
        print("Test:",fold_test_df["label"].value_counts().to_dict(),)
        fold_output_dir = os.path.join(output_dir,f"fold_{fold_index}",)
        os.makedirs(fold_output_dir,exist_ok=True,)

        fold_train_df.to_csv(os.path.join(fold_output_dir,"metadata_train.csv",),index=False,) #save data
        fold_val_df.to_csv(os.path.join(fold_output_dir,"metadata_validation.csv",),index=False,)
        fold_test_df.to_csv(os.path.join(fold_output_dir,"metadata_test.csv",),index=False,)

        # Dataset.
        train_dataset = HistopathologyDataset(dataframe=fold_train_df,transform=train_transforms,)
        val_dataset = HistopathologyDataset(dataframe=fold_val_df,transform=evaluation_transforms,)
        test_dataset = HistopathologyDataset(dataframe=fold_test_df,transform=evaluation_transforms,)

        # Pesos del fold.
        class_weights = calculate_fold_class_weights(train_df=fold_train_df,num_classes=num_classes,)
        
        # New model for each fold
        model = create_model(num_classes=num_classes)

        if FREEZE_BACKBONE: freeze_resnet18_backbone(model)

        print_trainable_parameters(model)
        training_args = create_fold_training_arguments(fold_output_dir,fold_seed,LEARNING_RATE,
                                                       WEIGHT_DECAY, BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS)
        compute_metrics = build_compute_metrics(num_classes=num_classes)

        trainer_class = WeightedCrossEntropyTrainer if USE_WEIGHTED_LOSS else Trainer
        trainer_kwargs = dict(model=model,args=training_args,
                              train_dataset=train_dataset,
                              eval_dataset=val_dataset,
                              data_collator=collate_fn,
                              compute_metrics=compute_metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],)

        if USE_WEIGHTED_LOSS:
            trainer_kwargs["class_weights"] = class_weights

        trainer = trainer_class(**trainer_kwargs)
        trainer.train()

        # Evaluación en el test externo del fold.
        test_output = trainer.predict(test_dataset)
        logits = extract_logits(test_output.predictions)
        probabilities = stable_softmax(logits)
        predicted_labels = np.argmax(probabilities, axis=1)
        true_labels = test_output.label_ids

        fold_metrics = calculate_fold_metrics(true_labels=true_labels,predicted_labels=predicted_labels,probabilities=probabilities,class_names=class_names,)

        fold_metrics["fold"] = fold_index
        fold_metrics["seed"] = fold_seed
        fold_metrics["n_train"] = len(fold_train_df)
        fold_metrics["n_validation"] = len(fold_val_df)
        fold_metrics["n_test"] = len(fold_test_df)
        all_fold_metrics.append(fold_metrics)

        print("\nMetrics for external test")
        for metric_name, metric_value in (fold_metrics.items()):
            print(f"{metric_name}: {metric_value}")

        print("\nClassification report")
        print(classification_report(true_labels,predicted_labels,
                                    labels=np.arange(num_classes),
                                    target_names=class_names,
                                    digits=4, zero_division=0,))

        print("\nConfusion matrix")
        print(confusion_matrix(true_labels,predicted_labels,labels=np.arange(num_classes),))

        # Predicciones out-of-fold.
        fold_predictions_df = fold_test_df.copy()
        fold_predictions_df["fold"] = fold_index
        fold_predictions_df["true_label_id"] = true_labels
        fold_predictions_df["predicted_label_id"] = predicted_labels
        fold_predictions_df["predicted_label"] = [id2label[int(label_id)] for label_id in predicted_labels]
        fold_predictions_df["confidence"] = (probabilities.max(axis=1))

        for class_id, class_name in enumerate(class_names):
                    fold_predictions_df[f"probability_{class_name}"] = probabilities[:, class_id]
        all_oof_predictions.append(fold_predictions_df)

        # Free memory before the next fold.
        del trainer, model, train_dataset, val_dataset, test_dataset
        if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    fold_metrics_df = pd.DataFrame(all_fold_metrics).sort_values("fold")
    oof_predictions_df = pd.concat(all_oof_predictions,ignore_index=True,)
    fold_metrics_df.to_csv(os.path.join(output_dir,"cross_validation_fold_metrics.csv",),index=False,)
    oof_predictions_df.to_csv(os.path.join(output_dir,"cross_validation_oof_predictions.csv",),index=False,)
    return fold_metrics_df, oof_predictions_df

def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n Device")
    print("===========")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA available: No")

    print("\n Loading metadata...")
    dataframe = load_cross_validation_metadata(train_csv_path=TRAIN_CSV_PATH,val_csv_path=VAL_CSV_PATH,test_csv_path=TEST_CSV_PATH,)

    verify_individual_labels(dataframe)

    dataframe, class_names, label2id, id2label = create_global_label_mapping(dataframe)

    print_individual_distribution(dataframe)
    print(f"\nTotal of images: {len(dataframe)}")
    print("Total of individuals:", dataframe["individual_id"].nunique())
    print("Classes:", class_names)

    save_experiment_configuration(TRAIN_CSV_PATH,VAL_CSV_PATH,TEST_CSV_PATH,
                                  MODEL_NAME, SEED, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
                                  LEARNING_RATE, WEIGHT_DECAY,NUM_EPOCHS,EARLY_STOPPING_PATIENCE,
                                  WARMUP_RATIO,NUM_WORKERS,
                                  OUTPUT_DIR, 
                                  class_names, 
                                  label2id, 
                                  id2label,)

    train_transforms, evaluation_transforms = create_transforms()
    print(f"Size (input): {IMAGE_SIZE} × {IMAGE_SIZE}")

    fold_metrics_df, oof_predictions_df = (
        run_stratified_group_cross_validation(dataframe=dataframe,class_names=class_names,id2label=id2label,train_transforms=train_transforms,
            evaluation_transforms=evaluation_transforms,output_dir=OUTPUT_DIR,))
    summarize_cross_validation_metrics(fold_metrics_df=fold_metrics_df,output_dir=OUTPUT_DIR)
    calculate_global_oof_metrics(oof_predictions_df=oof_predictions_df,class_names=class_names,output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
