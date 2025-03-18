

from collections import OrderedDict

def extract_unique_filenames_in_order(input_file, output_file):
    # Read the content of the original file
    with open(input_file, 'r') as file:
        content = file.readlines()

    # Process each line and extract the filenames while maintaining the order
    extracted_filenames = []
    seen_filenames = set()
    for line in content:
        # Split each line by spaces and get the first part of each pair
        filenames = line.strip().split(' ')
        for filename in filenames:
            # Remove the file extension (.tif)
            extracted_filename = filename.split('/')[-1].split('.')[0]
            # Add the filename to the list if it's not a duplicate
            if extracted_filename not in seen_filenames:
                extracted_filenames.append(extracted_filename)
                seen_filenames.add(extracted_filename)

    # Write the extracted filenames to a new file while preserving order
    with open(output_file, 'w') as file:
        file.write('\n'.join(extracted_filenames))

if __name__ == "__main__":
    input_file = r'D:\study\banjiandu\UniMatch-main\UniMatch-main\splits\pascal\val.txt'
    output_file = './val.txt'

    extract_unique_filenames_in_order(input_file, output_file)
