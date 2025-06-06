# Uber-Chats-Analysis-using-LLMs
This project leverages large language models (LLMs) via a Google Colab notebook to classify, analyze, and evaluate textual content such as confessions, advice, and messages. It supports categorization, clustering, topic modeling, and tone detection using custom prompts and parallel processing techniques.

📎 Colab Notebook: Run the Notebook

🗂️ Project Structure
The process is broken into four main stages:

1. 🔧 Data Preparation
1.1 Method 1 – Single File Processing
A single CSV file is loaded for analysis.

Control the number of rows using the num parameter.

1.2 Method 2 – Multiple File Merging
Multiple datasets (e.g., confessions, advice) are read and concatenated into one dataset.

Use the num parameter to limit rows per file.

Note: Update the 'content' column with data from Sid Results -> Data Files.

2. 🧠 Classification Process
2.1 How Classification Works
✅ Prompting the Model
Predefined prompts classify text into categories such as:

Power

Leadership

Helping

⚙️ Parallel Processing
Messages are processed in parallel using ThreadPoolExecutor to improve performance.

🧾 Output Columns
Label: Predicted category (e.g., "Leadership")

Explanation: Justification for the assigned label

Make sure the correct column (e.g., 'Column Name') is passed to the parallel_generate function.
Choose the appropriate prompt from Section 2 – All Prompts for Analysis based on your topic (e.g., leadership/helping/power).

3. 📊 Evaluating Model Performance
This section is for evaluating classification accuracy using manually labeled data.

3.1 Metrics
Accuracy: % of correct predictions

Recall: Ability to find all relevant cases

Precision: Proportion of true positive predictions

3.2 Steps to Evaluate
Provide the column name with manual labels.

Run the metric calculation cells.

Review printed scores to evaluate model performance.

4. 🔍 Analysis Process: Clustering, Topic, and Tone
This phase uses chunking and LLMs for further textual insights.

4.1 When to Use What
4.1.1 Clustering
Run this for messages labeled as 1 (positive).

Both messages and explanations are used.

4.1.2 Tone and Topic Analysis
Analyze both 0s and 1s.

Only messages are considered here.

4.2 Chunking Parameters
Step Size: Number of messages (+ explanations) per LLM call (Recommended: 50–100)

Max Workers: Number of parallel calls for faster performance

4.3 Step-by-Step Analysis
🧾 Step 1: Initial Table Generation (Prompt 1)
The model generates initial tables based on messages (and optionally explanations).

Use the appropriate prompt from Section 2 – All Prompts for Analysis.

🔁 Step 2: Iterative Table Refinement (Prompt 2)
Initial tables are refined iteratively using another prompt for structured output.

📤 Step 5: Final Output Formatting
Once the final table is ready:

Copy-paste it into ChatGPT.

Use the following prompt:

pgsql
Copy
Edit
Please format this table, exactly giving the full table without changing any content or text
<FINAL TABLE PASTE HERE>
📌 Notes
Prompts are modular and topic-specific; always pick the relevant one based on what you are analyzing (e.g., classification, clustering, tone).

The process uses OpenAI models under the hood, accessed via custom prompt engineering and chunked parallel calls for efficient handling of large datasets.

📁 Repository Contents
notebook.ipynb: Main Colab notebook to run the full pipeline

README.md: This file

prompts/: (Optional) Folder to store prompt templates

data/: Folder for uploading CSV files

outputs/: Folder for saving results

🚀 Getting Started
Open the Colab Notebook

Follow the steps from data loading to analysis.

Choose prompts and parameters as needed.

Run evaluation if manual labels are available.

Export and format final outputs using ChatGPT.
