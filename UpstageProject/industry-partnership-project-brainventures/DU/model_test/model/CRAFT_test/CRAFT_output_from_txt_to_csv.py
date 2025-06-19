import os
import csv
import sys

# --- Configuration ---
# Input directory containing the .txt files
input_dir = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/CRAFT-pytorch-master/result"
# Output directory for the CSV file
output_dir = "/data/ephemeral/home/industry-partnership-project-brainventures/output"
# Name of the output CSV file
output_csv_filename = "CRAFT_base.csv"
# Prefix to remove from the original filenames
prefix_to_remove = "res_"
# Default extension to ensure filenames have
default_extension = ".jpg"
# --- End Configuration ---

def create_csv_from_txt(input_folder, output_folder, output_filename, prefix_remove, ext_to_add):
    """
    Reads all .txt files from an input folder, processes filenames and content,
    and writes the transformed data into a single CSV file matching the Easyocr_test.csv format.

    Transformations:
    - Filename: Removes prefix, ensures it ends with ext_to_add.
    - Content: Replaces '\n' with '|', replaces ',' with ' '. (No surrounding quotes expected from CRAFT txt)

    Args:
        input_folder (str): Path to the folder containing .txt files.
        output_folder (str): Path to the folder where the CSV will be saved.
        output_filename (str): Name for the output CSV file.
        prefix_remove (str): The prefix string to remove from filenames.
        ext_to_add (str): The file extension to ensure filenames have (e.g., ".jpg").
    """
    # Construct the full path for the output CSV file
    output_csv_path = os.path.join(output_folder, output_filename)

    # Ensure the output directory exists, create it if it doesn't
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output directory '{output_folder}' ensured.")
    except OSError as e:
        print(f"Error creating output directory '{output_folder}': {e}", file=sys.stderr)
        return # Exit if we can't create the output directory

    # List all files in the input directory
    try:
        all_files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Error: Input directory not found: '{input_folder}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error listing files in '{input_folder}': {e}", file=sys.stderr)
        return

    # Filter for .txt files only
    txt_files = [f for f in all_files if f.lower().endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in '{input_folder}'. CSV file will be empty or contain only headers.")
        # Create an empty CSV with headers anyway
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Use QUOTE_NONE to mimic the target format
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerow(['filename', 'polygons']) # Write header
            print(f"Created empty CSV with headers: {output_csv_path}")
        except IOError as e:
            print(f"Error writing empty CSV file '{output_csv_path}': {e}", file=sys.stderr)
        return

    print(f"Found {len(txt_files)} .txt files to process.")

    # --- Open CSV file for writing ---
    # Use 'w' mode to overwrite if exists, specify newline='' and encoding
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # --- Create CSV Writer ---
            # Use quoting=csv.QUOTE_NONE to prevent adding extra quotes.
            # Use escapechar='\\' just in case the delimiter (,) appears unexpectedly
            # in the already transformed data (though it shouldn't).
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')

            # --- Write Header ---
            writer.writerow(['filename', 'polygons']) # Match Easyocr_test.csv header

            # --- Process each .txt file and write row ---
            rows_written = 0
            for txt_filename in txt_files:
                # --- 1. Process Filename ---
                processed_filename = txt_filename
                # Remove prefix if it exists
                if processed_filename.startswith(prefix_remove):
                    processed_filename = processed_filename[len(prefix_remove):]

                # Ensure the filename ends with the desired extension (Requirement 4)
                base_name, current_ext = os.path.splitext(processed_filename)
                if not current_ext: # No extension found
                     output_filename_csv = base_name + ext_to_add
                elif current_ext.lower() != ext_to_add.lower(): # Different extension found
                     print(f"Warning: Filename '{processed_filename}' had extension '{current_ext}', replacing with '{ext_to_add}'.")
                     output_filename_csv = base_name + ext_to_add
                else: # Already has the correct extension
                     output_filename_csv = processed_filename

                # Construct the full path to the current .txt file
                txt_filepath = os.path.join(input_folder, txt_filename)

                # --- 2. Read and Transform Content ---
                try:
                    with open(txt_filepath, 'r', encoding='utf-8') as f:
                        # Read the entire file content, strip leading/trailing whitespace
                        content = f.read().strip()

                        # Requirement 3: Replace newline characters (\n) with "|"
                        content = content.replace('\n', '|')

                        # Requirement 2: Replace commas (,) with a single space
                        content = content.replace(',', ' ')

                except Exception as e:
                    print(f"Warning: Could not read or process file '{txt_filepath}': {e}", file=sys.stderr)
                    content = "" # Use empty string if file reading/processing fails

                # --- 3. Write Row to CSV ---
                writer.writerow([output_filename_csv, content])
                rows_written += 1

            print(f"Successfully created and transformed CSV file: {output_csv_path}. Wrote {rows_written} data rows.")

    except IOError as e:
        print(f"Error writing to CSV file '{output_csv_path}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}", file=sys.stderr)

# --- Main execution ---
if __name__ == "__main__":
    print("Starting script...")
    create_csv_from_txt(input_dir, output_dir, output_csv_filename, prefix_to_remove, default_extension)
    print("Script finished.")

