import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Constants
IMAGE_SIZE = (100, 100)  # 1080p resolution, you can increase this for higher quality
NUM_IMAGES = 2000  # Increased number of image pairs to generate
MIN_DIAMETER = 5  # Minimum diameter of the ball
MAX_DIAMETER = 10  # Maximum diameter of the ball
MIN_OFFSET = 1  # Minimum offset for the stereo pair
MAX_OFFSET = 10  # Maximum offset for the stereo pair
IMAGE_FOV = 90  # Field of view of the camera in degrees
STEREO_DISTANCE = 0.01  # Distance between the stereo cameras in meters
DATASET_DIR = "dataset"  # Directory to save the images

# Training constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Updated learning rate
EPOCHS = 5


class CustomDistanceLayer(nn.Module):
    def __init__(self):
        super(CustomDistanceLayer, self).__init__()

    def forward(self, disparity_map):
        batch_size = disparity_map.size(0)
        height, width = disparity_map.size(2), disparity_map.size(3)
        
        # Create tensor of x-coordinates (0 to width-1) using broadcasting
        x_coords = torch.arange(width, device=disparity_map.device).float()
        x_coords = x_coords.view(1, 1, 1, -1)  # Shape: (1, 1, 1, width)
        
        # Calculate x_left and x_right directly on the disparity_map tensor
        x_left = x_coords - disparity_map / 2
        x_right = x_coords + disparity_map / 2
        
        # Convert x-coordinates to angles in radians
        theta_left = self.theta_rad_from_img(x_left, width)
        theta_right = self.theta_rad_from_img(x_right, width)
        
        # Compute the denominator to avoid division by zero
        denominator = torch.sin(theta_left) - torch.sin(theta_right)
        small_denom_mask = torch.abs(denominator) < 1e-6
        denominator[small_denom_mask] = 1e-6 * torch.sign(denominator[small_denom_mask])
        
        # Compute distances based on disparity
        distances = torch.abs(STEREO_DISTANCE / denominator)
        distances[small_denom_mask] = 1e6
        
        return distances

    def theta_rad_from_img(self, x, width):
        # Convert x-coordinates to angles in radians
        ret = ((width / 2 - x) * IMAGE_FOV) / width
        return torch.deg2rad(ret)


class DistancePredictor(nn.Module):
    def __init__(self):
        super(DistancePredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        
        self.custom_distance_layer = CustomDistanceLayer()
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

        # Additional layers after final_conv
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8), 1)  # Adjust size based on pooling
    
    def forward(self, x):
        # # Concatenate images along the channel dimension
        # x = torch.cat((left_image, right_image), dim=1)
        
        # Generate disparity map
        disparity_map = self.conv1(x)
        disparity_map = self.relu(disparity_map)
        disparity_map = self.conv2(disparity_map)
        
        # Compute distances from the disparity map
        distances = self.custom_distance_layer(disparity_map)
        
        # Condense to the distance of a specific object
        object_distance = self.final_conv(distances)

        # Apply additional convolutional layers
        x = disparity_map  # Shape: (N, 1, H, W)
        x = self.conv3(self.relu(x))  # Apply ReLU and convolution
        x = self.pool(x)  # Apply pooling
        
        x = self.conv4(self.relu(x))  # Apply ReLU and convolution
        x = self.pool(x)  # Apply pooling
        
        x = self.relu(self.conv5(self.relu(x))) # Apply ReLU and convolution
        x = self.pool(x)  # Apply pooling
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Apply the fully connected layer to get the final output
        x = self.fc(x)  # Output a single value per sample
        
        return x

def load_data(dataset_dir):
    image_files = [f for f in os.listdir(dataset_dir) if f.startswith("left_image")]
    data = []
    labels = []

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    for file in image_files:
        left_image_path = os.path.join(dataset_dir, file)
        right_image_path = left_image_path.replace("left_image", "right_image")

        if not os.path.exists(right_image_path):
            print(f"Warning: Matching right image for '{file}' not found. Skipping.")
            continue

        left_image = transform(Image.open(left_image_path))
        right_image = transform(Image.open(right_image_path))

        # combined_tensor = torch.cat((left_image, right_image), dim=0).flatten()
        # data.append(combined_tensor)
        # Combine left and right images into a single tensor with 2 channels
        combined_tensor = torch.cat((left_image, right_image), dim=0)
        data.append(combined_tensor)
        match = re.search(r'_dist([\d.]+)_', file)
        if match:
            distance = float(match.group(1))
            labels.append(distance)
        else:
            print(f"Warning: Filename '{file}' does not match expected pattern. Skipping this file.")
            continue

    return torch.stack(data), torch.tensor(labels, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)  # Flatten output for loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}')

def test_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    mae = mean_absolute_percentage_error(actuals, predictions)
    print(f'Test MAE: {mae}')

# Load data
data, labels = load_data(DATASET_DIR)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create DataLoader instances
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistancePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train and test the model
train_model(model, train_loader, criterion, optimizer)
test_model(model, test_loader)