import mne
import re
import os

# Directory containing .fif files
data_dir = './lsl_data/combined/'
output_ext = f"_eeglab"
file_ext = '.fif'

def simplify_annotations(raw):
    """
    Simplify annotation descriptions by removing trial numbers and '_start' suffixes.
    For example:
        "trial_1_baseline_start" -> "baseline"
        "trial_12_motor_execution_start" -> "motor_execution"
    """
    new_descriptions = []
    pattern = re.compile(r'trial_\d+_(.+?)(?:_start)?$')

    for desc in raw.annotations.description:
        match = pattern.match(desc)
        if match:
            new_desc = match.group(1)
        else:
            new_desc = desc  # leave unchanged if no match
        new_descriptions.append(new_desc)

    # Replace with simplified annotations
    raw.annotations.description[:] = new_descriptions

    return raw

# Directory containing .fif files
data_dir = './lsl_data/combined/'
output_ext = f"_eeglab"
file_ext = '.fif'

for fname in os.listdir(data_dir):
    if fname.endswith(file_ext):
        file_base = fname[:-len(file_ext)]  # Remove the '.fif' extension
        file_path = os.path.join(data_dir, fname)
        output_path = os.path.join(file_base)  # Path for the .set and .fdt files

        print(f"Loading {fname}")
        raw = mne.io.read_raw_fif(file_path, preload=True)
        print(raw._data)
        print(raw.info)
        raw = simplify_annotations(raw)
        # Optional: Save the cleaned raw file
        raw.save(f'{file_base}_new.fif', overwrite=True)
