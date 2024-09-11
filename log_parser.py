import os
import json
import re

def parse_section(section, verbose=False):
    """ Parse the complete section to extract necessary details with an optional verbose output """
    if verbose:
        print("Section preview:", section[:200])  # Print beginning of the section to inspect

    repeat_count_match = re.search(r'---- repeat count: \s*(\d+)', section)
    repeat_count = int(repeat_count_match.group(1)) if repeat_count_match else 0

    prompt_inputs = re.findall(r'---- prompt_input_\d+ (.*?)(?=----|$)', section, re.DOTALL)

    prompt = re.findall(r'---- prompt: (.*?)(?=----|$)', section, re.DOTALL)[0]

    func_validate = re.findall(r'---- func_validate: (.*?)(?=----|$)', section, re.DOTALL)[0]

    gpt_parameter = re.findall(r'---- gpt_parameter: (.*?)(?=----|$)', section, re.DOTALL)[0]

    prompt_template = re.findall(r'---- prompt_template: (.*?)(?=----|$)', section, re.DOTALL)[0].strip()

    curr_response = re.findall(r'---- curr_gpt_response: (.*?)(?=----|$)', section, re.DOTALL)[0]

    func_cleanup = re.findall(r'----  func_clean_up: (.*?)(?=----|$)', section, re.DOTALL)[0]

    func_validate_match = re.search(r'---- func_validate: \s*(True|False)', section)
    func_validate = func_validate_match.group(1).lower() == 'true' if func_validate_match else False


    return {
        "prompt_inputs": prompt_inputs,
        "repeat_count": repeat_count,
        "prompt": prompt,
        "prompt_template": prompt_template,
        "gpt_parameter": gpt_parameter,
        "func_validate": func_validate,
        "curr_response": curr_response,
        "func_cleanup": func_cleanup
    }

def read_large_file(input_file_path, output_directory, verbose=False):
    os.makedirs(output_directory, exist_ok=True)
    section = ""
    file_count = 0
    in_section = False
    count_dict = {}  # Dictionary to store file counts per directory

    with open(input_file_path, 'r') as file:
        for line in file:
            if "------- BEGIN SAFE GENERATE --------" in line:
                in_section = True
                section = line
            elif "------- END TRIAL" in line and in_section:
                section += line
                result = parse_section(section, verbose)
                if result:
                    # Extracting the basename from the 'prompt_template' key
                    template_path = result.get('prompt_template', 'ERROR')
                    basename = os.path.basename(template_path)
                    # Removing file extension
                    basename = os.path.splitext(basename)[0]

                    # Directory for the basename
                    output_subdir = os.path.join(output_directory, basename)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Check if we already have a count for this directory, otherwise count files
                    if basename not in count_dict:
                        existing_files = [f for f in os.listdir(output_subdir) if f.endswith('.json')]
                        count_dict[basename] = len(existing_files) + 1
                    else:
                        count_dict[basename] += 1

                    # Writing the result to the JSON file
                    json_filename = os.path.join(output_subdir, f"{count_dict[basename]}.json")
                    with open(json_filename, 'w') as json_file:
                        json.dump(result, json_file, indent=4)
                    file_count += 1
                section = ""
                in_section = False
            elif in_section:
                section += line

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_log_file_path> <output_directory_path> [<verbose>]")
    else:
        input_log_file_path = sys.argv[1]
        output_directory_path = sys.argv[2]
        verbose = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
        read_large_file(input_log_file_path, output_directory_path, verbose)