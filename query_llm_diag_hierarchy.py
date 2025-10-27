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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)

from openai import OpenAI

from utils import ordinal, make_demographic_prompt, load_dataset, process_result, setup_output_paths
from prompt import SYSTEMPROMPT, INSTRUCTION_PROMPT


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm_openai(
    model_name: str,
    llm: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, int, int]:
    """
    Query the LLM with retry logic.

    Args:
        model_name: Name of the model to use
        llm: OpenAI client instance
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model

    Returns:
        Tuple of (model response content, prompt tokens used, completion tokens used)
    """
    try:
        result = llm.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )
    except Exception as e:
        raise e
    
    return result.choices[0].message.content


def is_transient_error(exc):
    # Gemini API errors often have a .code or status property
    msg = str(exc).lower()
    return any(code in msg for code in ["429", "500", "503", "unavailable", "timeout"])

@retry(retry=retry_if_exception(is_transient_error), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm_gemini(
    model_name: str,
    llm: Callable,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, int, int]:
    """
    Query the LLM with retry logic.

    Args:
        model_name: Name of the model to use
        llm: OpenAI client instance
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model

    Returns:
        Tuple of (model response content, prompt tokens used, completion tokens used)
    """
    try:
        result = llm.generate_content(
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            generation_config={"temperature": 0}
        )
    
    except Exception as e:
        raise e
        
    return result.text


def prepare_prompt(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Prepare the prompt for the LLM based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (system prompt, instruction prompt)
    """
    # Get system prompt
    system_prompt = SYSTEMPROMPT.strip()

    # Get instruction prompt
    instruction_prompt = INSTRUCTION_PROMPT[args.task].strip()

    return system_prompt, instruction_prompt


def run(args: argparse.Namespace):
    """
    Main function to run the LLM evaluation.

    Args:
        args: Command line arguments
    """

    # Load the dataset
    ids, notes, ys, ys_parental, ages, sexs, races, insurances = load_dataset(args)

    # Prepare the system prompt, instruction_prompt
    system_prompt, instruction_prompt = prepare_prompt(args)

    # Initialize LLM
    if args.model == 'gpt-4o-mini':
        llm = OpenAI(
            api_key='YOUR_OPENAI_API_KEY'
        )

    elif args.model == 'gemini-2.0-flash':
        import google.generativeai as genai  
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        llm = genai.GenerativeModel(model_name=args.model, system_instruction=system_prompt)

    # Setup output paths
    logits_path = setup_output_paths(args)

    for pid, note, y, y_parental, age, sex, race, insurance in tqdm(zip(ids, notes, ys, ys_parental, ages, sexs, races, insurances), total=len(notes)):

        # Process patient ID
        if isinstance(pid, float):
            pid = str(round(pid))

        # Check if the patient has already been processed
        if os.path.exists(os.path.join(logits_path, f'{pid}.pkl')):
            print(f'Patient {pid} already processed, skipping.')
            continue

        segments = [
            f"{ordinal(i)} Visit Note:\n{note_i}\n"
            for i, (note_i) in enumerate(zip(note), start=1)
        ]

        demographic_prompt = make_demographic_prompt(age, sex, race)

        user_prompt = (
            f"{instruction_prompt}\n\n"
            f"{demographic_prompt}\n\n"
            + "\n\n".join(segments)
        )

        # Query LLM and save results if required
        if args.model == 'gpt-4o-mini':
            try:
                result = query_llm_openai(
                    model_name=args.model,
                    llm=llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            except Exception as e:
                print(f'Error when querying LLM for patient {pid}: {e}')
                continue
        
        elif args.model == 'gemini-2.0-flash':
            try:
                result = query_llm_gemini(
                    model_name=args.model,
                    llm=llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            except Exception as e:
                print(f'Error when querying LLM for patient {pid}: {e}')
                continue
            
        # Process the result
        try:
            pred, pred_parental, label, label_parental, think = process_result(result, y, y_parental)

            # Save the result
            pd.to_pickle({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'response': result,
                'think': think,
                'pred': pred,
                'label': label,    
                'pred_parental': pred_parental,
                'label_parental': label_parental,                  
                'age': age,
                'sex': sex,
                'race': race,
                'insurance': insurance,
            }, os.path.join(logits_path, f'{pid}.pkl'))

        except Exception as e:
            print(f'Error when processing result for patient {pid}: {e}')

            # Save original result for debugging
            pd.to_pickle({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'response': result,
            }, os.path.join(logits_path, f'{pid}.pkl'))

            continue


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run LLM evaluation on unstructured EHR data')

    # Dataset and task configuration
    parser.add_argument('--dataset', type=str, default='mimic-iv', choices=['mimic-iv'], help='Dataset to use')
    parser.add_argument('--task', type=str, default='diagnosis_hierarchical', help='Task to perform')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='LLM model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--ratio', type=float, default=0.0)
    parser.add_argument('--degradation', type=str, default=None, help='Text degradation')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    print(f"Running with configuration: Model: {args.model}, Dataset: {args.dataset}, Task: {args.task}. Output {'logits and performance'}.")
    run(args) 


    # python query_llm_diag_hierarchy.py --task diagnosis_hierarchical --model gemini-2.0-flash
    # python query_llm_diag_hierarchy.py --task diagnosis_hierarchical --degradation random_lab_values_erasure --ratio 1.0 --model gemini-2.0-flash
    # python query_llm_diag_hierarchy.py --task diagnosis_hierarchical --degradation copy_paste_from_previous --ratio 0.8 --model gemini-2.0-flash
    # python query_llm_diag_hierarchy.py --task diagnosis_hierarchical --degradation ocr_corrupt_document --ratio 0.3 --model gemini-2.0-flash
    # python query_llm_diag_hierarchy.py --task diagnosis_hierarchical --degradation replace_with_homophones_fast --ratio 0.3 --model gemini-2.0-flash