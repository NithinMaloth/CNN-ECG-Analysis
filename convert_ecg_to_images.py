import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the WFDB folder within the ECG Data directory on your desktop
wfdb_folder = os.path.join(os.path.expanduser('~'), 'Downloads', 'archive(1)', 'WFDB')

# Generate the list of ECG record names
record_names = [f'E{str(i).zfill(5)}' for i in range(1, 10345)]

# Length of subsequence in milliseconds
W_milliseconds = 1000

# Sampling frequency (replace with the actual value)
sampling_frequency = 500  # Adjust this value based on your data

# Number of pixels for the resulting image
image_size = 64

# Preprocessing and converting to images
def create_ecg_image(ecg_subsequence):
    # Normalize the subsequence
    normalized_subsequence = (ecg_subsequence - ecg_subsequence.min()) / (ecg_subsequence.max() - ecg_subsequence.min())
    
    # Resize the normalized subsequence to image_size
    resized_subsequence = np.resize(normalized_subsequence, (image_size, image_size))
    
    # Convert to grayscale image with 256 levels
    grayscale_image = (resized_subsequence * 255).astype(np.uint8)
    
    return grayscale_image

# Iterate through ECG records and convert to images
for record_name in record_names:
    # Load ECG signal data from .mat file
    mat_file_path = os.path.join(wfdb_folder, record_name + '.mat')
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract the Lead II signal data
    lead_data = mat_data['val'][1]  # Assuming Lead II is at index 1
    
    # Adjusted gain and baseline values based on header information
    gain = 4880 / 24  # Replace with the gain value for Lead II (16+24 bits)
    baseline = -48   # Replace with the baseline value for Lead II (16+24 bits)
    
    # Apply scaling and preprocessing to lead_data
    lead_data = (lead_data - baseline) / gain
    
    # Create a directory to save the images
    image_save_folder = os.path.join(os.path.expanduser('~'), 'Desktop', 'ECG Images', record_name)
    os.makedirs(image_save_folder, exist_ok=True)
    
    # Iterate through ECG segments and convert to images
    for i in range(len(lead_data)):
        # Take the subsequence of length W milliseconds
        subsequence_length = int((W_milliseconds / 1000) * sampling_frequency)
        ecg_subsequence = lead_data[i:i+subsequence_length]
        
        ecg_image = create_ecg_image(ecg_subsequence)
        
        # Save the image as a PNG file
        image_path = os.path.join(image_save_folder, f'subsequence_{i + 1}.png')
        plt.imsave(image_path, ecg_image, cmap='gray')

print("Images saved successfully!")
