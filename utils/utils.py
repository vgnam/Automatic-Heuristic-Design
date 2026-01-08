from litellm import completion
import os
import logging
import concurrent.futures
import time
import re
import inspect
from population_encode import get_embedding, compute_cosine_similarity
import numpy as np


def compute_novelty_scores(population, k=3):
    """
    Compute novelty scores for each individual based on k-nearest neighbor distances.
    """
    if len(population) <= 1:
        return [1.0] * len(population)  # Maximum novelty if only one or no individuals

    # Get embeddings for all individuals
    embeddings = [get_embedding(ind['code']) for ind in population]

    # Compute pairwise cosine similarities
    similarity_matrix = compute_cosine_similarity(embeddings)

    novelty_scores = []
    for i in range(len(population)):
        # Get similarities to all other individuals (excluding self)
        similarities = similarity_matrix[i]
        # Convert similarities to distances (1 - similarity)
        distances = [1 - sim for j, sim in enumerate(similarities) if i != j]
        distances.sort()

        # Average distance to k nearest neighbors
        if len(distances) >= k:
            knn_dist = np.mean(distances[:k])
        else:
            knn_dist = np.mean(distances) if distances else 0.0

        novelty_scores.append(knn_dist)

    return novelty_scores

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):

    assert isinstance(messages_list, list), "messages_list should be a list."
    try:
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
    except:
        print(messages_list)
        raise IndexError("Something is wrong.")

    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

    num_workers = os.cpu_count()
    if "gpt" not in model:
        # Transform messages if n > 1
        messages_list *= n
        n = 1
        num_workers = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(n, messages, model, temperature) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c.message.content)
    return contents


# def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
#     """
#     Generate n responses using OpenAI Chat Completions API
#     """
#
#     for attempt in range(100):
#         try:
#             response_cur = completion(model=model,
#                                       messages=messages,
#                                       temperature=temperature,
#                                       n=n)
#                                       # api_base="http://localhost:1234/v1",
#                                       # api_key="sk-no-key-required")
#             break
#         except Exception as e:
#             logging.info(f"Attempt {attempt + 1} failed with error: {e}")
#             time.sleep(3)
#     if response_cur is None:
#         logging.info("Code terminated due to too many failed attempts!")
#         exit()
#
#     return response_cur.choices

def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    def clean_text(text: str) -> str:
        return text.encode("ascii", errors="replace").decode()

    response_cur = None

    for attempt in range(100):
        try:
            response_cur = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                n=n
            )
            break
        except Exception as e:
            err = clean_text(str(e))
            logging.info(f"Attempt {attempt + 1} failed with error: {err}")
            time.sleep(3)

    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()

    return response_cur.choices



def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    if "import" not in code_string:
        code_string = "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals['parameter_ranges']
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return parameter_ranges, function_block


def format_messages(cfg, pre_messages):
    messages = [{"role": "system", "content": pre_messages["system"]},
                {"role": "user", "content": pre_messages["user"]}]
    return messages