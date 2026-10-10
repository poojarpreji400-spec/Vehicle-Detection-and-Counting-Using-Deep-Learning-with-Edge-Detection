from ultralytics import YOLO
import cv2

# load the pretrained model
model = YOLO('yolov8n.pt')

image = cv2.imread('traffic.jpg')

if image is None:
    raise FileNotFoundError('Please put traffic.jpg')

# perform edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converts color into grayscale
edges = cv2.Canny(gray, 100, 200) # find edges in the image

# display the edge- detected image.
cv2.imwrite('Edge Detection.jpg', edges)

# detect objects in the images
results = model(image) 

# vehicle class ID's in the COCO dataset
# YOLOv8's standard pretrained COCO model uses numerical IDs for object categories.
vehicle_class = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
} 

# counter for each vehicle
counts = {'Car':0,
          'Motorcycle': 0,
          'Bus': 0,
          'Truck': 0}

# count the detected vehicles

for result in results:
    for box in result.boxes: # examine each detected object.
        class_id = int(box.cls[0]) # retrives the predicted class
        
        if class_id in vehicle_class: # checks whether it is one of our four vehicle types.
            vehicle_name = vehicle_class[class_id]
            counts[vehicle_name] +=1 # increases that vehicle's count by one.

# draw detection boxes around them.
# and add confidence scores around them.          
output_image = results[0].plot()

# save the ouput image.
# The processed image is saved in the project folder so you can view it later or include it in your GitHub repository.
cv2.imwrite('detected_vehicles.jpg', output_image)

print("\nvehicle Detection Report")
print("-------------")

for vehicle, count in counts.items():
    print(f'{vehicle}:{count}')
    
total_vehicles = sum(counts.values())
print('------------')
print(f'Total Vehicles:{total_vehicles}')

# display the detected image

cv2.imshow('Vehicle DEtection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('\nOutput saved as detected_vehicles.jpg')           

