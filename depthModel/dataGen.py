from PIL import Image, ImageDraw
import random
import os
import math

# Constants
IMAGE_SIZE = (100, 100)  # 1080p resolution, you can increase this for higher quality
NUM_IMAGES = 2000  # Increased number of image pairs to generate
MIN_DIAMETER = 10  # Minimum diameter of the ball
MAX_DIAMETER = 20  # Maximum diameter of the ball
MIN_OFFSET = 1  # Minimum offset for the stereo pair
MAX_OFFSET = 5  # Maximum offset for the stereo pair
IMAGE_FOV = 90  # Field of view of the camera in degrees
STEREO_DISTANCE = 0.01  # Distance between the stereo cameras in meters
DATASET_DIR = "dataset"  # Directory to save the images

# Ensure the dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

def create_ball_image(diameter, image_size, ball_position):
    """
    Create an image with a ball on a black background.
    
    Parameters:
    - diameter: Diameter of the ball.
    - image_size: Size of the image (width, height).
    - ball_position: Position of the ball's top-left corner (x, y).
    
    Returns:
    - Image object with the ball.
    """
    image = Image.new("RGB", image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    left_up_point = ball_position
    right_down_point = (ball_position[0] + diameter, ball_position[1] + diameter)
    draw.ellipse([left_up_point, right_down_point], fill=(255, 255, 255))
    
    return image

def generate_stereo_pair(image_size, diameter, ball_position, offset):
    """
    Generate a stereo pair by shifting the ball position.
    
    Parameters:
    - image_size: Size of the image (width, height).
    - diameter: Diameter of the ball.
    - ball_position: Initial position of the ball's top-left corner (x, y).
    - offset: Horizontal offset to shift the ball for the stereo pair.
    
    Returns:
    - A tuple of (left_image, right_image) as stereo image pair.
    """
    left_image = create_ball_image(diameter, image_size, ball_position)
    
    right_ball_position = (ball_position[0] + offset, ball_position[1])
    right_image = create_ball_image(diameter, image_size, right_ball_position)
    
    return left_image, right_image

def theta_rad_from_img(x, width):
    ret = ((width / 2 - x) * IMAGE_FOV) / width
    return math.radians(ret)
def calculate_distance(x_left, x_right):
    """
    Calculate the distance between the ball positions in the stereo pair.
    
    Parameters:
    - x_left: The x-coordinate of the ball in the left image.
    - x_right: The x-coordinate of the ball in the right image.
    
    Returns:
    - An integer representing the calculated distance.
    """
    # Formula to calculate distance will be filled in by the user
    image_width = IMAGE_SIZE[1]

    try:
        dist = abs(STEREO_DISTANCE / (math.sin(theta_rad_from_img(x_left, image_width)) - math.sin(theta_rad_from_img(x_right, image_width))))
    except ZeroDivisionError:
        dist = float('inf')  # Handle division by zero
    return dist

def main():
    for i in range(NUM_IMAGES):
        # Randomize diameter and offset
        diameter = random.randint(MIN_DIAMETER, MAX_DIAMETER)
        offset = random.randint(MIN_OFFSET, MAX_OFFSET)
        ball_position = (random.randint(0, IMAGE_SIZE[0] - diameter - MAX_OFFSET), 
                         random.randint(0, IMAGE_SIZE[1] - diameter))
        
        # Generate stereo images
        left_image, right_image = generate_stereo_pair(IMAGE_SIZE, diameter, ball_position, offset)
        
        # Calculate distance (for naming purposes)
        distance = calculate_distance(ball_position[0], ball_position[0] + offset)
        
        # Save images with descriptive names
        left_image.save(f"{DATASET_DIR}/left_image_{i}_d{diameter}_o{offset}_dist{distance}_fov{IMAGE_FOV}_sd{STEREO_DISTANCE}.png")
        right_image.save(f"{DATASET_DIR}/right_image_{i}_d{diameter}_o{offset}_dist{distance}_fov{IMAGE_FOV}_sd{STEREO_DISTANCE}.png")
        
        # Optionally, print progress
        if (i + 1) % 100 == 0:
            print(f"Generated pair {i+1}/{NUM_IMAGES}")

if __name__ == "__main__":
    main()
