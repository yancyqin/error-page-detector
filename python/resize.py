from PIL import Image
import os

# Define the folder path containing the images
folder_path = 'devJam2024'

# Define the desired image size
desired_size = (224, 224)  # Change this to your desired image size

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Open the image file
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            # Resize the image
            resized_img = img.resize(desired_size)
            
            # Save the resized image, overwriting the original file
            resized_img.save(image_path)
            print(f"Resized and saved: {image_path}")
