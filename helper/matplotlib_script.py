import os
import re
import json
from pathlib import Path
from openai import OpenAI

# Set the folder paths
BASE_DIR = "../../"
SOURCE_FOLDER = os.path.join(BASE_DIR, "hw")
LIBRARY_FOLDER = os.path.join(BASE_DIR, "ml_algo_lib/")

# Create the library folder if it doesn't exist
os.makedirs(LIBRARY_FOLDER, exist_ok=True)

def get_openai_api_key():
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    return api_key

# Set up OpenAI client
client = OpenAI(api_key=get_openai_api_key())

# Regex to identify function definitions
def function_regex():
    return re.compile(r"^def\s+(\w+)\(.*\):")

def extract_code_from_notebook(notebook_path):
    """Extracts all code cells from a Jupyter notebook."""
    with open(notebook_path, 'r') as file:
        notebook = json.load(file)
    
    code_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return "\n".join(["".join(cell) for cell in code_cells])

def extract_functions_from_code(code):
    """Extracts functions from a code string."""
    lines = code.splitlines()
    in_function = False
    function_name = ""
    function_code = []
    extracted_functions = {}
    
    for line in lines:
        if line.strip().startswith("def "):
            if in_function:
                extracted_functions[function_name] = "\n".join(function_code)
            in_function = True
            function_code = [line]
            function_name_match = function_regex().match(line)
            function_name = function_name_match.group(1) if function_name_match else ""
        elif in_function:
            function_code.append(line)
    if in_function:
        extracted_functions[function_name] = "\n".join(function_code)
    return extracted_functions

def generate_refactored_code_with_llm(code, notebook_name):
    """Uses an LLM to generate semantically meaningful and well-structured library code."""
    try:
        prompt = (
            f"The following code was extracted from a Jupyter notebook named '{notebook_name}'. "
            "Refactor the code into well-structured and reusable libraries, ensuring clean interfaces and meaningful separation of functionality.\n\n"
            f"{code}\n\n"
            "Provide the refactored code with explanations for each module created, making it easy to understand and maintain."
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that refactors code into well-structured libraries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        print("Please check your OpenAI API key and quota limits at: https://platform.openai.com/account/usage")
        return f"# Error occurred during refactoring:\n# {str(e)}\n\n{code}"

def extract_and_refactor():
    # Traverse through all homework notebooks in the directory
    for hw_folder in [f for f in os.listdir(BASE_DIR) if f.startswith("hw")]:
        hw_path = os.path.join(BASE_DIR, hw_folder)
        for file in os.listdir(hw_path):
            if file.endswith(".ipynb"):
                full_path = os.path.join(hw_path, file)
                # Extract all code from the notebook
                code = extract_code_from_notebook(full_path)
                # Use LLM to refactor code into meaningful libraries
                refactored_code = generate_refactored_code_with_llm(code, file)
                # Save the refactored code to a new library file
                refactored_file_path = os.path.join(LIBRARY_FOLDER, f"refactored_{file.replace('.ipynb', '.py')}")
                with open(refactored_file_path, 'w') as library_file:
                    library_file.write(refactored_code)

if __name__ == "__main__":
    extract_and_refactor()
    print(f"Library files are generated in: {LIBRARY_FOLDER}")
