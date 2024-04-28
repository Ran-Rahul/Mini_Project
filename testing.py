import os
import cv2
import pandas as pd

def load_dataset(data_dir):
    dataset = []

    # Iterate over train and val directories
    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, f'D_{split}', 'images')
        labels_dir = os.path.join(data_dir, f'D_{split}', 'labels')

        # Load metadata from metadata.csv
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path)

        # Iterate over rows in metadata CSV
        for _, row in metadata_df.iterrows():
            image_filename = row['Image_Filename']
            label_filename = row['Label_Filename']

            # Read image
            image_path = os.path.join(images_dir, image_filename)
            image = cv2.imread(image_path)

            # Read label (you might have a different format, adjust accordingly)
            label_path = os.path.join(labels_dir, label_filename)
            # Read label data here

            # Append image and label to dataset
            dataset.append({'image': image, 'label': None})  # Update label_data accordingly

    return dataset

def main():
    # Load dataset
    data_dir = 'archive/weapon_detection/train'  # Update with your dataset directory
    dataset = load_dataset(data_dir)
    print(f'Loaded {len(dataset)} samples from the dataset.')

    # Open default camera (usually camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Camera', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
