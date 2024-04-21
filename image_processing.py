from PIL import Image
import matplotlib.pyplot as plt

def convert_to_black_and_white(image_path, threshold):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Apply thresholding to convert to black and white
    img_bw = img_gray.point(lambda x: 0 if x < threshold else 255, '1')
    
    # Get the black pixels coordinates
    black_pixels = [(x, y) for x in range(img_bw.width) for y in range(img_bw.height) if img_bw.getpixel((x, y)) == 0]
    
    return black_pixels

def get_black_pixels(image_path, threshold):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Apply thresholding to convert to black and white
    img_bw = img_gray.point(lambda x: 0 if x < threshold else 255, '1')
    
    # Get the black pixels coordinates
    black_pixels = [(x, y) for x in range(img_bw.width) for y in range(img_bw.height) if img_bw.getpixel((x, y)) == 0]
    
    return black_pixels

def plot_black_pixels(black_pixels):
    # Plot black pixels
    if black_pixels:
        plt.scatter(*zip(*black_pixels), color='black', marker='s', s=2)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.show()
    else:
        print("No black pixels found.")

# Example usage
def sample_data(image_path='images/logo.png', threshold=70):
    threshold = 70  # Adjust threshold as needed
    black_pixels = get_black_pixels(image_path, threshold)
    # plot_black_pixels(black_pixels)
    rescaled = rescale_points(black_pixels)

    return rescaled

def rescale_points(coordinates, minimum=0, maximum=10):
    # Find the minimum and maximum values of x and y coordinates
    x_min = min(coord[0] for coord in coordinates)
    x_max = max(coord[0] for coord in coordinates)
    y_min = min(coord[1] for coord in coordinates)
    y_max = max(coord[1] for coord in coordinates)
    
    # Calculate the current range of x and y coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate the scaling factor based on the larger of the two ranges
    x_scale_factor = (maximum - minimum) / x_range
    y_scale_factor = (maximum - minimum) / y_range
    
    # Rescale each coordinate tuple
    rescaled_coordinates = [(x_scale_factor * (coord[0] - x_min) + minimum, 
                             y_scale_factor * (coord[1] - y_min) + minimum) 
                            for coord in coordinates]
    
    return rescaled_coordinates


sample_data()