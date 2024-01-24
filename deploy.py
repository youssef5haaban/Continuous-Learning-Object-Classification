import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from torchvision import transforms
from slda_model import StreamingLDA
import retrieve_any_layer
from torchvision import models
import torch.nn as nn
import time
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def setup_custom_dataloader(images, labels, batch_size=256, transform=None, shuffle=False, num_workers=8):
    dataset = CustomDataset(images, labels, transform=transform)

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, sampler=sampler
    )
    return loader


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)
    return feat


def get_feature_extraction_model(arch, imagenet_pretrained, feature_size, num_classes):
    feature_extraction_model = models.__dict__[arch](pretrained=imagenet_pretrained)
    feature_extraction_model.fc = nn.Linear(feature_size, num_classes)
    return feature_extraction_model


def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_frame = transform(frame)

    if preprocessed_frame is None:
        print("Error: preprocessed_frame is None.")
    return preprocessed_frame


def classify_image(model, feature_extraction_wrapper, image):
    preprocessed_image = preprocess_image(image)
    input_tensor = torch.unsqueeze(preprocessed_image, 0)
    features = feature_extraction_wrapper(input_tensor)
    pooled_features = pool_feat(features)
    predictions = model.predict(pooled_features, return_probas=True)
    probabilities, predicted_class = torch.max(predictions, 1)
    return predicted_class.item(), probabilities.item()


def capture_images(cap, duration=5):
    start_time = time.time()
    images = []
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
    return images


def train(model, feature_extraction_wrapper, train_loader):
    print('\nTraining on %d images.' % len(train_loader.dataset))

    for train_x, train_y in tqdm(train_loader, total=len(train_loader)):
        batch_x_feat = feature_extraction_wrapper(train_x)
        batch_x_feat = pool_feat(batch_x_feat)

        # train one sample at a time
        for x_pt, y_pt in zip(batch_x_feat, train_y):
            model.fit(x_pt.cpu(), y_pt.view(1, ))


def main():
    try:
        # Load your trained SLDA model
        model = StreamingLDA(input_shape=512, num_classes=1000)

        # Create the feature extraction model
        feature_extraction_model = get_feature_extraction_model(
            arch='resnet18', imagenet_pretrained=True, feature_size=512, num_classes=1000).cpu()

        feature_extraction_wrapper = retrieve_any_layer.ModelWrapper(
            feature_extraction_model.eval(), ['layer4.1'], return_single=True).eval()

        cap = None  # Initialize cap outside the loop

        while True:
            cap = None  # Initialize cap outside the loop
            add_new = input("Do you want to add a new class? (y/n): ").lower()

            if add_new == 'n':
                # Classify the captured frame normally
                # Open a connection to the camera (camera index 0 by default)
                if cap is None:
                    cap = cv2.VideoCapture(0)

                while True:
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        break

                    # Classify the captured frame
                    predicted_class, probability = classify_image(model, feature_extraction_wrapper, frame)

                    # Display the result on the frame
                    text = f'Predicted Class: {predicted_class}, Probability: {probability:.2f}'
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Display the frame
                    cv2.imshow('Camera Feed', frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Release the camera
                cap.release()
                cv2.destroyAllWindows()
                cap = None  # Set cap to None to reopen it next time if needed

            elif add_new == 'y':

                new_label = input("Enter the label for the new class: ")

                # Open a connection to the camera (camera index 0 by default)
                if cap is None:
                    cap = cv2.VideoCapture(0)

                # Capture images for the new class with timer
                new_class_images = capture_images(cap, duration=5)

                # Release the camera
                cap.release()

                # Create a folder for the new class images
                folder_path = f'captured_images/{new_label}'
                os.makedirs(folder_path, exist_ok=True)

                # Save captured images for the new class
                for i, image in enumerate(new_class_images):
                    image_path = os.path.join(folder_path, f'{new_label}_{i}.jpg')
                    cv2.imwrite(image_path, image)

                # Train the model with the new data
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

                # Use the train function to train the new class

                train(model, feature_extraction_wrapper, new_class_loader)

                print(f"Model trained with the new class '{new_label}'.")

            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if cap is not None:
            # Release the camera
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
