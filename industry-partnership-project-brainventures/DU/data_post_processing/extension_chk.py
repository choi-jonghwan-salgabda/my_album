import os
import collections
import csv # Using csv module for potentially more robust handling, though split might work too

# --- Configuration ---
file_path = '/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/easyocr.csv'
target_extension = '.jpg' # The extension we are primarily checking against (lowercase)
# --- ---

extension_counts = collections.Counter()
non_target_entries = []
line_count = 0
processed_count = 0
error_count = 0

print(f"Analyzing file: {file_path}")

try:
    with open(file_path, 'r', encoding='utf-8') as infile:
        # Use csv.reader with tab delimiter
        reader = csv.reader(infile, delimiter='\t')
        for line_num, row in enumerate(reader, 1):
            line_count += 1
            if not row: # Skip empty rows
                print(f"Warning: Line {line_num} is empty.")
                error_count += 1
                continue

            filename_part = row[0].strip() # Get the first column and remove leading/trailing whitespace

            if not filename_part:
                print(f"Warning: Line {line_num} has an empty first column.")
                extension_counts['(empty)'] += 1
                non_target_entries.append(f"Line {line_num}: (empty first column)")
                processed_count += 1
                continue

            # Use os.path.splitext for robust extension extraction
            try:
                base, ext = os.path.splitext(filename_part)

                if ext: # If an extension exists
                    normalized_ext = ext.lower() # Normalize to lowercase for counting
                    extension_counts[normalized_ext] += 1
                    if normalized_ext != target_extension:
                        non_target_entries.append(f"Line {line_num}: {filename_part} (Extension: {ext})")
                else: # No extension found
                    print(f"Warning: Line {line_num}: '{filename_part}' has no extension.")
                    extension_counts['(no extension)'] += 1
                    non_target_entries.append(f"Line {line_num}: {filename_part} (No extension)")

                processed_count += 1

            except Exception as split_error:
                 print(f"Error processing filename on line {line_num}: '{filename_part}' - {split_error}")
                 extension_counts['(error processing)'] += 1
                 non_target_entries.append(f"Line {line_num}: Error processing '{filename_part}'")
                 error_count += 1


except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred at line ~{line_count}: {e}")
    exit()

# --- Report Results ---
print("\n--- Analysis Complete ---")
print(f"Total lines read: {line_count}")
print(f"Entries processed in first column: {processed_count}")
if error_count > 0:
    print(f"Lines skipped due to errors/emptiness: {error_count}")

print("\n--- File Extension Statistics (Case-Insensitive) ---")
if extension_counts:
    total_valid_entries = sum(count for ext, count in extension_counts.items() if not ext.startswith('('))
    print(f"Total entries with identifiable extensions or specific issues: {sum(extension_counts.values())}")
    # Sort by count descending for better readability
    for ext, count in extension_counts.most_common():
        percentage = (count / processed_count) * 100 if processed_count > 0 else 0
        print(f"{ext}: {count} ({percentage:.2f}%)")
else:
    print("No extensions found or file was empty/unreadable.")

print(f"\n--- Check for non-{target_extension} entries ---")
if non_target_entries:
    print(f"Found {len(non_target_entries)} entries that are NOT '{target_extension}' (case-insensitive) or had issues.")
    # Optionally print the first few examples if the list isn't too long
    print("Examples:")
    for entry in non_target_entries[:20]: # Print up to 20 examples
       print(f"  - {entry}")
    if len(non_target_entries) > 20:
       print(f"  ... ({len(non_target_entries) - 20} more)")
else:
    print(f"All valid entries seem to have the {target_extension} extension (case-insensitive).")

