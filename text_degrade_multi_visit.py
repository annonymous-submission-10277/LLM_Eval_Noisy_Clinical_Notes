from pathlib import Path
import re
import random
import pronouncing
import numpy as np
import pandas as pd
import nlpaug.augmenter.char as nac
from wordfreq import word_frequency
from tqdm import tqdm


def random_lab_values_erasure(texts, max_prob=0.3, seed=1234):
    
    if seed is not None:
        random.seed(seed)

    # Match patterns like "WBC-5.0*", "ALT(SGPT)-100*", etc.    
    pattern = re.compile(r'\b[a-zA-Z_]{2,10}\d+\.?\d*\b', re.IGNORECASE)

    def erase_lab_values(text, prob_i):
        def maybe_remove(match):
            return '' if random.random() < prob_i else match.group(0)
        cleaned = pattern.sub(maybe_remove, text)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return re.sub(r'\n{2,}', '\n\n', cleaned)

    return [erase_lab_values(text, np.random.beta(3, 6) * max_prob) for text in texts]


def ocr_corrupt_document(texts, max_prob=0.2, seed=1234):
    
    if seed is not None:
        random.seed(seed)

    aug = nac.OcrAug()
    result = []

    for text in texts:
        prob_i = np.random.beta(3, 6) * max_prob
        words = text.split()
        augmented = []
        for word in words:
            if random.random() < prob_i:
                try:
                    augmented_word = aug.augment(word)
                    augmented.append(augmented_word[0] if isinstance(augmented_word, list) else augmented_word)
                except:
                    augmented.append(word)
            else:
                augmented.append(word)
        result.append(' '.join(augmented))
    return result


def get_homophones(word, cache):

    word_l = word.lower()
    if word_l in cache:
        return cache[word_l]
    
    phones = pronouncing.phones_for_word(word_l)
    if not phones:
        cache[word_l] = []
        return []
    
    homophones = set()
    for phone in phones:
        for homo in pronouncing.search(phone):
            homo = homo.lower()
            if homo != word_l:
                homophones.add(homo)
    
    homophones = list(homophones)

    homophones = sorted(homophones, key=lambda w: -word_frequency(w, 'en'))
    cache[word_l] = homophones
    return homophones


def replace_with_homophones_fast(texts, top_k=1, max_prob=0.2, seed=1234):

    if seed is not None:
        random.seed(seed)

    def replace_in_text(text, prob_i):
        words = text.split()
        word_indices = list(range(len(words)))
        n_replace = int(len(words) * prob_i)

        selected_indices = set(random.sample(word_indices, n_replace)) if n_replace > 0 else set()
        cache = {}
        new_words = []

        for i, word in enumerate(words):
            if i in selected_indices:
                homos = get_homophones(word, cache)
                if homos:
                    chosen = random.choice(homos[:top_k])
                    new_words.append(chosen)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        return ' '.join(new_words)

    return [replace_in_text(text, np.random.beta(3, 6) * max_prob) for text in texts]


def copy_paste_from_previous(texts, max_prob=0.3, seed=1234):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(texts) <= 1:
        return texts

    corrupted = [texts[0]]  # First note remains unchanged

    for i in range(1, len(texts)):
        prob_i = np.random.beta(3, 6) * max_prob
        if random.random() < prob_i:

            # Append previous note to current note
            combined = texts[i - 1] + ", " + texts[i]
            corrupted.append(combined)
        else:
            corrupted.append(texts[i])

    return corrupted


def degrade_text_and_save(data_path, max_prob, func):

    data_path = Path(data_path)

    # Determine file type
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        import ast
        df['discharge_summary'] = df['discharge_summary'].apply(ast.literal_eval)
    elif data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {data_path.suffix}")
        
    df['DegradedText'] = df['discharge_summary'].apply(lambda text: func(text, max_prob=max_prob))
    df = df.drop(columns=['discharge_summary'])
    df = df.rename(columns={'DegradedText': 'discharge_summary'})

    # Format probability nicely
    prob_ = f"{max_prob}".rstrip("0").rstrip(".")

    # Build output path using pathlib for safe path handling
    if data_path.suffix == '.csv':
        data_path = Path(data_path)
        output_filename = f"{data_path.stem}_{func.__name__}_{prob_}.csv"
        output_path = data_path.parent / output_filename

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    else:
        data_path = Path(data_path)
        output_filename = f"{data_path.stem}_{func.__name__}_{prob_}.parquet"
        output_path = data_path.parent / output_filename

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    print(f"Saving to: {output_path}")


def main(data_path):

    degrade_text_and_save(data_path, 1.0, random_lab_values_erasure)
    degrade_text_and_save(data_path, 0.8, copy_paste_from_previous)
    degrade_text_and_save(data_path, 0.3, ocr_corrupt_document)
    degrade_text_and_save(data_path, 0.3, replace_with_homophones_fast)


if __name__ == '__main__':
    main(data_path = "./processed/mimic_iv/diagnosis_final/hierarchical_data.csv")