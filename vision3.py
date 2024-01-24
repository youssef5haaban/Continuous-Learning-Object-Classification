import cv2
import streamlit as st
from deploy import classify_image, get_feature_extraction_model, capture_images, train, setup_custom_dataloader
import os
import torch
from torchvision import transforms
from slda_model import StreamingLDA
import retrieve_any_layer
from tqdm import tqdm
import json

def main():
    st.title("Continuous Learning Project (Wakeb:8)")
    st.header("Webcam with Real-time Classification")

    # Load your model (ensure this is done according to your model's requirements)
    model = StreamingLDA(input_shape=512, num_classes=1000)
    model.load_model(save_path='D:/DEBI - AI and Data Science/Graduation Ptoject/Implementation/project/', save_name='final_slda_model')

    with open('category_mapping.txt', 'r') as file:
        category_mapping = json.load(file)

    feature_extraction_model = get_feature_extraction_model(
        arch='resnet18', imagenet_pretrained=True, feature_size=512, num_classes=1000).cpu()

    feature_extraction_wrapper = retrieve_any_layer.ModelWrapper(
        feature_extraction_model.eval(), ['layer4.1'], return_single=True).eval()

    # Callback function to toggle the capture state
    def toggle_capture():
        st.session_state.capture = not st.session_state.capture

    # Initialize session state
    if 'capture' not in st.session_state:
        st.session_state.capture = False

    # Set button text based on the current state
    btn_text = "Stop" if st.session_state.capture else "Start"

    # Button with callback
    st.button(btn_text, on_click=toggle_capture)

    # Frame Placeholder
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.capture:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break
        # Image preprocessing and classification
        predicted_class, probability = classify_image(model, feature_extraction_wrapper, frame)
        # Use the category mapping to get the category name
        category_name = category_mapping.get(str(predicted_class), "Unknown")  # Convert to string for JSON keys

        # Display the result on the frame
        text = f'Category: {category_name}, Prob: {probability:.2f}'
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        cv2.imshow('Camera Feed', frame)

    cap.release()

    # Retraining Section
    if 'update' not in st.session_state:
        st.session_state['update'] = False

    # Button to toggle retraining mode
    if st.button("Add New Class"):
        st.session_state['update'] = True

    # When retrain mode is active, show input and button for capturing and training
    if st.session_state['update']:
        new_object = st.text_input("Enter the Name for the new class:")

        if st.button("Capture and Train"):
            # Automatically generate the new label based on the last index
            if category_mapping:
                last_index = max(map(int, category_mapping.keys()))
                new_label = str(last_index + 1)
            else:
                new_label = "1"  # If no classes exist, start with index 1

            # Only add the new label if a valid name is provided
            if new_object:
                category_mapping[new_label] = new_object
                with open('category_mapping.txt', 'w') as file:
                    json.dump(category_mapping, file)

                # Ensure label is not empty
                if new_label:
                    if 'cap' in locals() or 'cap' in globals():
                        cap.release()
                    st.write("Starting Capturing...")
                    cap = cv2.VideoCapture(0)

                    # Capture images for the new class with timer
                    new_class_images = capture_images(cap, duration=5)
                    cap.release()

                    # Create a folder and save images
                    folder_path = f'captured_images/{new_label}'
                    os.makedirs(folder_path, exist_ok=True)

                    for i, image in enumerate(new_class_images):
                        image_path = os.path.join(folder_path, f'{new_label}_{i}.jpg')
                        cv2.imwrite(image_path, image)

                    # Prepare images for training
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    new_class_labels = [int(new_label)] * len(new_class_images)
                    new_class_loader = setup_custom_dataloader(new_class_images, new_class_labels, batch_size=128,
                                                                shuffle=True, transform=transform)

                    st.write("Model Updating in progress...")
                    train(model, feature_extraction_wrapper, new_class_loader)
                    model.save_model('D:/DEBI - AI and Data Science/Graduation Ptoject/Implementation/project/', 'final_slda_model')

                    st.session_state['update'] = False  # Reset the retrain toggle after the process is complete
                    st.success(f"Model Updating completed for '{new_object}'.")
                else:
                    st.error("Please enter a valid label before capturing and training.")
            else:
                st.error("Please enter a valid name for the new class.")


if __name__ == "__main__":
    main()
