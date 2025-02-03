import os
import gzip
import shutil

# Define source and destination folders
source_folder = '.'  # Replace with your source folder path
destination_folder = '.'  # Replace with your destination folder path

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all CSV files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        # Define full paths for input and output files
        file_path = os.path.join(source_folder, filename)
        gzipped_file_path = os.path.join(destination_folder, filename + '.gz')

        # Open the original CSV file and the output gzip file
        with open(file_path, 'rb') as f_in:
            with gzip.open(gzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)  # Copy content to the gzipped file

        print(f"Compressed {filename} to {gzipped_file_path}")

        # Optional: Remove original CSV file after compression
        os.remove(file_path)
        print(f"Removed original file: {filename}")

print("All CSV files have been compressed and moved.")





