import cv2
import csv

def detect(image_path, output_image_path, stars_datafile_path, threshold):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to isolate bright regions (stars)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Find contours (potential stars)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars = []

    # Draw detections and collect data
    for i, cnt in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, r = int(x), int(y), int(radius)

        if 0.1 < r < 6:
            brightness = int(gray[y, x])  # approximate brightness
            stars.append([x, y, r, brightness, len(stars) + 1])
            cv2.circle(img, (x, y), r + 2, (0, 255, 0), 2)
            cv2.putText(img, str(len(stars)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_image_path, img)

    with open(stars_datafile_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'r', 'b', '#'])
        for star in stars:
            writer.writerow(star)

    print("Detected stars [x, y, r, b, #]:")
    for star in stars:
        print(star)

if __name__ == '__main__':
    detect("fr2.jpg", "stars.jpg", "stars.csv", 200)