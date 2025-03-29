import os
import trafilatura
from tqdm import tqdm
from bs4 import BeautifulSoup
import multiprocessing
from functools import partial

def html_to_markdown(html_file_path, output_dir):
    """
    Convert HTML file to Markdown using trafilatura, preserving directory structure.

    Args:
        html_file_path (str): Path to the HTML file.
        output_dir (str): Base directory to save the Markdown output, mirroring the HTML file's directory structure.
        
    Returns:
        tuple: (html_file_path, success, error_message)
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Extract the main content using trafilatura
        downloaded = trafilatura.extract(html_content)

        if downloaded:
            # Determine the relative path from the base directory
            relative_path = os.path.relpath(html_file_path, start=HTML_DIR)

            # Create the output directory path
            output_path = os.path.join(output_dir, os.path.dirname(relative_path))

            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Create the output file path
            output_file_path = os.path.join(output_path, os.path.splitext(os.path.basename(html_file_path))[0] + '.md')

            # Save the Markdown content to the output file
            with open(output_file_path, 'w', encoding='utf-8') as md_file:
                md_file.write(downloaded)

            return (html_file_path, True, None)  # Indicate success
        else:
            return (html_file_path, False, "Failed to extract content")  # Indicate failure to extract

    except Exception as e:
        return (html_file_path, False, str(e))  # Indicate failure with error message

def process_directory(html_dir, output_dir, num_processes=None):
    """
    Process all HTML files in a directory and its subdirectories.
    Files in each directory are processed in parallel.

    Args:
        html_dir (str): Root directory containing HTML files.
        output_dir (str): Base directory to save the Markdown output, mirroring the HTML file's directory structure.
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Count total files for progress tracking
    total_files = 0
    for root, _, files in os.walk(html_dir):
        total_files += len([f for f in files if f.endswith('.html') or f.endswith('.htm')])

    # Create a progress bar for tracking the overall conversion
    with tqdm(total=total_files, desc="Converting HTML to Markdown") as pbar:
        # Process each directory, but parallelize the file processing within each directory
        for root, _, files in os.walk(html_dir):
            # Filter for HTML files in the current directory
            html_files = [os.path.join(root, file) for file in files if file.endswith('.html') or file.endswith('.htm')]
            
            if not html_files:
                continue

            # Create a process pool
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Create a partial function with fixed arguments
                process_func = partial(html_to_markdown, output_dir=output_dir)
                
                # Process files in parallel and handle results as they complete
                for file_path, success, error_msg in pool.imap_unordered(process_func, html_files):
                    pbar.update(1)
                    if not success:
                        pbar.write(f"Failed to convert {file_path}: {error_msg}")

# Example Usage (replace with your actual directories)
HTML_DIR = 'html_output'  # Replace with the path to your HTML directory
MARKDOWN_DIR = 'markdown_output'  # Replace with the desired output directory for Markdown files

if __name__ == "__main__":
    # Optional: specify number of processes, defaults to CPU count if not specified
    num_processes = multiprocessing.cpu_count()
    process_directory(HTML_DIR, MARKDOWN_DIR, num_processes)
