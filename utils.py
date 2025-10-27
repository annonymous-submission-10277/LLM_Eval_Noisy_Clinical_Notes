import json
import os
import argparse
import datetime
import ast
import pickle
import pandas as pd

from tqdm import tqdm
from json_repair import repair_json
from typing import List, Tuple, Any, Callable


def ordinal(n: int) -> str:
    mapping = {
        1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth",
    }
    return mapping.get(n, f"{n}th")


def make_demographic_prompt(age_group, sex, race):
    sex_str = "female" if sex == "F" else "male"
    return f"The patient is a {race.title()} {sex_str} aged {age_group} years."


def load_dataset(args: argparse.Namespace) -> Tuple[List, List, List]:
    """
    Load dataset based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of dataset components
    """
    
    if args.degradation is not None:
        grouped_df = pd.read_csv("./processed/mimic_iv/diagnosis_final_new/hierarchical_data_{}_{}.csv".format(args.degradation, args.ratio))
    else:
        grouped_df = pd.read_csv("./processed/mimic_iv/diagnosis_final_new/hierarchical_data.csv")

    grouped_df['hadm_id'] = grouped_df['hadm_id'].apply(ast.literal_eval)
    grouped_df['discharge_summary'] = grouped_df['discharge_summary'].apply(ast.literal_eval)
    grouped_df['diagnosis_mid_categories'] = grouped_df['diagnosis_mid_categories'].apply(ast.literal_eval)
    grouped_df['diagnosis_top_categories'] = grouped_df['diagnosis_top_categories'].apply(ast.literal_eval)
    grouped_df['next_diagnosis_mid_categories'] = grouped_df['next_diagnosis_mid_categories'].apply(ast.literal_eval)
    grouped_df['next_diagnosis_top_categories'] = grouped_df['next_diagnosis_top_categories'].apply(ast.literal_eval)
    
    ids   = grouped_df["subject_id"].tolist()
    notes = grouped_df["discharge_summary"].tolist()
    ys    = grouped_df["next_diagnosis_mid_categories"].tolist()
    ys_parent    = grouped_df["next_diagnosis_top_categories"].tolist()

    ages          = grouped_df["age_group"].tolist()
    sexs          = grouped_df["gender"].tolist()
    races         = grouped_df["race"].tolist()
    insurances    = grouped_df["insurance"].tolist()

    return ids, notes, ys, ys_parent, ages, sexs, races, insurances

def process_result(result: str, y: Any, y_parental: Any) -> Tuple[List[str], Any, str, List[str]]:
    """
    Process the LLM result into prediction and get the corresponding label.

    Args:
        result: LLM result string (JSON-like with 'think' and 'answer')
        y: Ground truth labels

    Returns:
        Tuple of (processed prediction list, ground truth label, think string, parental categories)
    """
    label = y
    label_parental = y_parental

    try:
        result_dict = repair_json(result, return_objects=True)
        if isinstance(result_dict, list):
            result_dict = result_dict[-1]

        think = result_dict['think']
        answer = result_dict['answer']

        if isinstance(answer, dict):
            pred = [item for sublist in answer.values() for item in sublist]
            pred_parental = list(answer.keys())
        else:
            raise ValueError(f"Expected dictionary format for 'answer', got {type(answer)}")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON content: {result}")
    except Exception as e:
        raise ValueError(f"Error processing result: {e}")
    
    pred = ', '.join(pred)
    pred_parental = ', '.join(pred_parental)
    label = ', '.join(label)
    label_parental = ', '.join(label_parental)

    return pred, pred_parental, label, label_parental, think


def setup_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Set up output paths for logits and prompts.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (logits path, prompts path, performance path)
    """
    
    ratio_str = f"{args.ratio:.3f}".rstrip("0").rstrip(".")

    if args.degradation is not None:        
        logits_path = os.path.join('results_{}_demo'.format(args.task), '{}_{}_{}'.format(args.model, args.degradation, ratio_str), 'qualitative')
        perf_path = os.path.join('results_{}_demo'.format(args.task), '{}_{}_{}'.format(args.model, args.degradation, ratio_str), 'quantitative')

    else:
        logits_path = os.path.join('results_{}_demo'.format(args.task), '{}_{}_{}'.format(args.model, 'original', ratio_str), 'qualitative')
        perf_path = os.path.join('results_{}_demo'.format(args.task), '{}_{}_{}'.format(args.model, 'original', ratio_str), 'quantitative')     

    os.makedirs(logits_path, exist_ok=True)
    os.makedirs(perf_path, exist_ok=True)

    return logits_path