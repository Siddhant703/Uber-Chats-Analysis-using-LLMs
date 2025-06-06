import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Initialize OpenAI Client
client = OpenAI(api_key='')  # TODO: Insert your API key

# 1. DATA PREPARATION FUNCTIONS 

def load_single_file(file_path, num_rows=None):
    """Load data from a single Excel or CSV file."""
    df = pd.read_excel(file_path)
    return df if num_rows is None else df[:num_rows]

def load_multiple_files(file_paths, num_rows):
    """Load and concatenate multiple CSV files."""
    data_frames = [pd.read_csv(path).iloc[:num_rows] for path in file_paths]
    return pd.concat(data_frames, ignore_index=True)

# 2. CLASSIFICATION FUNCTIONS

CLASSIFICATION_PROMPT = '''
INPUT_PROMPT
'''

def generate(text):
    """Call OpenAI API to classify messages into leadership/power/helping."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": CLASSIFICATION_PROMPT + text}]
    )
    return completion.choices[0].message.content

def parallel_generate(text_list, max_workers=100):
    """Classify texts in parallel using threading."""
    results = [None] * len(text_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate, text): idx for idx, text in enumerate(text_list)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error at {idx}: {e}")
    return results

def extract_labels(results):
    """Clean output by extracting classification labels from bracketed format."""
    return [str(r).replace("[","").replace("]","").split(",")[0] for r in results]

def extract_explanation(results):
    """Clean output by extracting classification labels from bracketed format."""
    return [''.join(str(r).replace("]","").replace("[","").split(",")[1:]) for r in results]

# 3. EVALUATION FUNCTIONS

def assign_binary_labels(df, label_column):
    df['value'] = df[label_column].apply(lambda x: 0 if 'no' in str(x).lower() else 1)
    return df

def compute_metrics(df, gold_col):
    acc = (df[gold_col].astype(int) == df['value'].astype(int)).sum() / len(df) * 100
    recall_df = df[df[gold_col] == 1]
    recall = (recall_df[gold_col].astype(int) == recall_df['value'].astype(int)).sum() / len(recall_df) * 100

    tp = ((df[gold_col] == 1) & (df['value'] == 1)).sum()
    fp = ((df[gold_col] == 0) & (df['value'] == 1)).sum()
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0

    return acc, recall, precision

# 4. CHUNKING ANALYSIS FUNCTIONS FOR TONE/CLUSTERING/TOPIC ANALYSIS

def parallel_generate_chunks(prompt, texts, step=50, max_workers=100):
    results = [None] * ((len(texts) + step - 1) // step)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate, prompt + ''.join(texts[i:i + step])): idx
            for idx, i in enumerate(range(0, len(texts), step))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Chunk error at index {idx}: {e}")
    return results

def iterative_table_refinement(initial_tables, prompt, step=20):
    current = initial_tables
    while True:
        refined = parallel_generate_chunks(prompt, current, step)
        if len(refined) <= step:
            return refined
        current = refined

if __name__ == "__main__":
    # ====== STEP 1: Load the data ======
    # Option A: Load a single Excel file
    # df = load_single_file("/content/India Leadership Class.xlsx", num_rows=500)

    # Option B: Load and concatenate multiple CSV files
    file_paths = [
        '/content/final_collected_data_Confessions (1).csv',
        '/content/final_collected_data_Advice.csv',
        '/content/final_collected_data_Complaints.csv'
    ]
    df = load_multiple_files(file_paths, num_rows=50000)

    # ====== STEP 2: Run classification ======
    text_column = 'content'  # or 'text' depending on dataset
    raw_results = parallel_generate(df[text_column].tolist())
    df['Tone'] = extract_labels(raw_results)
    df['Explanation'] = extract_explanation(raw_results)

    # ====== STEP 3: Save initial output ======
    df.to_excel("Tone_Analysis_Results.xlsx", index=False)

    # ====== STEP 4: Assign binary labels ======
    df = assign_binary_labels(df, label_column='Tone')

    # ====== STEP 5: Evaluate (Optional) ======
    # Replace 'ManualLabelColumn' with the actual column name of your golden labels
    # accuracy, recall, precision = compute_metrics(df, gold_col='ManualLabelColumn')
    # print(f"Accuracy: {accuracy:.2f}%, Recall: {recall:.2f}%, Precision: {precision:.2f}%")

    # ====== STEP 6: Filter for positives for clustering ======
    df_filtered = df[df['value'] == 1]
    analysis_input = [
        f"\nUber Message - {row[text_column]}\nExplanation - {row['Explanation']}"
        for _, row in df_filtered.iterrows()
    ]

    # ====== STEP 7: Run chunk-based analysis ======
    CHUNK_PROMPT = '''INPUT PROMPT FOR CLUSTERING'''
    initial_tables = parallel_generate_chunks(CHUNK_PROMPT, analysis_input)

    # ====== STEP 8: Iteratively refine tables ======
    TABLE_REFINEMENT_PROMPT = '''TABLE REFINEMENT PROMPT'''
    final_tables = iterative_table_refinement(initial_tables, TABLE_REFINEMENT_PROMPT)

    # ====== STEP 9: Save final clustering results ======
    pd.DataFrame(final_tables, columns=["Clustered Results"]).to_excel("Final_Cluster_Output.xlsx", index=False)
