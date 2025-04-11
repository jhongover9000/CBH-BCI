import mne
import os

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
        try:
            # Read the .fif file
            raw = mne.io.read_raw_fif(file_path, preload=True)
            print(raw._data)
            print(raw.info)
            raw._data = raw._data / 1e6
            # Export to EEGLAB format using the modified output path
            raw.export(f"{output_path}.set", fmt="eeglab")

            print(f"Successfully converted '{fname}' to '{output_path}.set' and '{output_path}.fdt'")

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")