import cv2 
import sqlite3 # connects to a database and save vehicle detection history
from ultralytics import YOLO
from datetime import datetime # works with date and time, records when images were observed and processed
from pathlib import Path # works with path files, checks where yolov8n.pt exists.
import numpy as np 
import pandas as pd 

base_dir = Path(__file__).resolve().parent
# file__: refers to the current main.py.
# .resolve(): This gets the full, absolute path of the script (e.g., C:/Users/Name/Projects/MyTrafficApp/main.py), resolving any shortcuts or symbolic links.
# .parent finds the folder containing that file.
model_file = base_dir/"yolov8n.pt"
# / Operator: In pathlib, the forward slash / is used to combine paths safely, automatically handling whether your system uses
db_file = base_dir/"traffic_history.db"

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
        'Motorcycle': 0, 'Bus': 0, 'Truck': 0}


def load_model():
    """Load the pretrained YOLO model."""
    if not model_file.exists():
        raise FileNotFoundError('Please put yolov8n.pt in the project folder.')
    return YOLO(str(model_file))


# connecting the database
def connect_db():
    """Connect to the SQLite database"""
    return sqlite3.connect(str(db_file), timeout = 10)
#timeout =10:Python to wait for up to 10 seconds if the database is currently locked by another program or thread before throwing an error

#setup_database()
def setup_database():
    """ Create the detections table if it does not exists"""
    
    with connect_db() as conn:
        conn.execute("""
CREATE TABLE IF NOT EXISTS detections
(id INTEGER PRIMARY KEY AUTOINCREMENT,
filename TEXT NOT NULL,
created_at TEXT NOT NULL,
observed_at TEXT NOT NULL,
car_count INTEGER NOT NULL,
motorcycle_count INTEGER NOT NULL,
bus_count INTEGER NOT NULL,
truck_count INTEGER NOT NULL,
total_count INTEGER NOT NULL)""")
# created_at & observed_at: Timestamps showing when the record was saved and when the vehicles were actually seen.

#save_record()
def save_record(filename, observed_at, counts):
    """Save vehicle counts and timestamps in SQLite."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# created_at automatically grabs the exact current date and time when the code runs.
# YYYY-MM-DD HH:MM:SS

    if isinstance(observed_at, datetime):
        observed_text = observed_at.strftime("%Y-%m-%d %H:%M:%S")
    else:
        observed_text = str(observed_at)
        #  If it's already a string, it leaves it as-is.

    with connect_db() as conn:
        conn.execute("""
                        INSERT INTO detections( filename, created_at, observed_at, car_count, motorcycle_count, bus_count, truck_count, total_count)
                        VALUES(?,?,?,?,?,?,?,?)""",
                        (filename, created_at, observed_text, int(counts["Car"]),
                         int(counts["Motorcycle"]),
                         int(counts["Bus"]),
                        int(counts["Truck"]),
                        int(sum(counts.values())))
        ) # • sum(counts.values()-  dynamically sums up all numbers inside your dictionary to calculate the total_count automatically before saving.


# filename	car_count	bus_count	total_count
# traffic.jpg	20          2            30

# read saved detection history
def read_records():
    """Return Saved records as a Pandas DataFrame"""
    setup_database()
    with connect_db() as conn:
        records = pd.read_sql_query("""
                                    SELECT *
                                    FROM detections 
                                    ORDER BY observed_at DESC, id DESC
                    """,  conn,)
# read_records(), connects to a database, 
# uns a SQL query, and 
# returns the contents of the detections table as a pandas DataFrame,
#sorted from newest to oldest.
    return records

def image_from_upload(upload):
    """Convert an uploaded image into an OpenCV image"""
    raw_bytes = np.frombuffer(
    upload.getvalue(), dtype = np.uint8)
#  It takes the file data uploaded by the user (upload.getvalue()), 
# which is just a raw stream of 1s and 0s in memory, 
# and turns it into a flat, 1-dimensional array of unsigned 8-bit integers

    image = cv2.imdecode(raw_bytes , cv2.IMREAD_COLOR)
#  It takes that raw array of bytes and 
# decodes it into a standard OpenCV image matrix

    if image is None:
        raise ValueError(f'Could not read{upload.name}. Please upload a valid image')

    return image



# count the detected vehicles

def detect_vehicles(image, model, confidence = 0.25):
    """Detect vehicles and return counts and an annotated RGB image"""
    result = model.predict(image,
                        conf= float(confidence), verbose = False)[0]
# • What it does: It passes your image into the object-detection model.
# Confidence Filter: The conf parameter tells the model to ignore any
# detections it is less than 25% (0.25) sure about.
# Output: It grabs the first result ([0]), which contains data about all the detected
# objects (locations, labels, etc.).


    counts = {'Car':0,
        'Motorcycle': 0, 'Bus': 0, 'Truck': 0}

    for box in result.boxes:
        class_id = int(box.cls[0])
    
        if class_id in vehicle_class:
                vehicle_name = vehicle_class[class_id]
                counts[vehicle_name] +=1
# Initialization: It starts with a fresh dictionary (e.g., {'car': 0, 'truck': 0}).
# The Loop: It looks at every bounding box the model detected.
# Filtering: It checks the ID of the detected object (class_id).  it looks up its name (e.g., "car") and adds 1 to that vehicle's total count.
                
        
    marked_bgr = result.plot()
    marked_rgb = cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)
# • Plotting: result.plot() automatically draws rectangles and labels around every detected object on top of the original image.
# Color Correction: The AI package generates this image in BGR format (Blue, Green, Red), which is standard for OpenCV. Because most other Python tools (like Matplotlib or web interfaces) expect RGB (Red, Green, Blue), cv2.cvtColor flips the color channels so the image colors don't look inverted/weird when you display it.


    return counts, marked_rgb
# the dictionary containing the numeric counts of each vehicle, and the newly annotated image.


# perform edge detection
def detect_edges(image, low_threshold = 100,high_threshold = 200):
    """Convert the image to greyscale and detect edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converts color into grayscale

    edges = cv2.Canny(gray, int(low_threshold), int(high_threshold),)
    
    return edges


# chatbot 
def answer_question(question, records):
    """Answer common qusetions using saved detection records"""
    question_text = question.lower().strip()
    # It converts the user's question to lowercase and removes accidental spaces.
    
    if records.empty:
        return "There are no saved detections yet. Analyse an image first"
    latest = records.iloc[0]
    
    # check questions about the busiesthour.
    if any(word in question_text for word in('hour', 'by hour','which time')):
        hourly_data = records.copy()
        hourly_data["hour"] = pd.to_datetime(hourly_data['observed_at'], errors= 'coerce').dt.strftime('%Y-%m-%d %H:00')
# If a time keyword is found, it safely copies the dataset and standardizes the timestamps in the observed_at column.
# It rounds every timestamp down to its respective hour block (e.g., 2026-10-18 13:34:00 becomes 2026-10-18 13:00). This prepares the data to be grouped so you can count how many detections happened in each specific hour.

        hourly_data = hourly_data.dropna(subset = ['hour'])
        # What it does: It cleans the data. It removes any rows from the dataset (hourly_data) where the 'hour' column is missing or blank (NaN). 
        # This ensures the code only looks at valid timestamps.
        if hourly_data.empty:
            return 'There are no valid observation timestamps to compare'
        #  This is a safety check. If removing the blank rows left the dataset completely empty, the function stops immediately and
        # returns a helpful message instead of crashing.
        hourly_totals = hourly_data.groupby('hour')['total_count'].sum()
        #  It groups the data by each hour of the day (e.g., 1 AM, 2 AM, etc.) and adds up (.sum()) all the detections
        # (intended to be total_count) for each of those hours
        busiest_hour = hourly_totals.idxmax()
        # The .idmax() function finds the highest total in hourly_totals and grabs its corresponding label
        # —which is the specific hour itself (the index).
        busiest_count = int(hourly_totals.max())
        return(
            f"The busiest recorded hour is {row['filename']}"
            f"with {int(row['total_count'])} vehicles,"
            f"observed at {row['observed_at']}."
        )
 # Check which saved image has the highest count.
    if any(word in question_text for word in ['highest','busiest','most traffic', 'maximum']):
        row = records.loc[records['total_count'].idxmax()]
        return(
            f"The highest-count saved image is {row['filename']}"
    f"with{int(row['total_count'])} vehicles," 
    f"oberved at {row['observed_at']}.")
    
 # Check the overall total.
    if('total' in question_text
or 'how many vehicles' in question_text
or 'vehicle count' in question_text
):
        return(
            f"Across all saved image records, there are "
    f"{int(records['total_count'].sum())} detections "
    f"across{len(records)} image records." 
    "Overlapping images may include duplicate vehicles.")
    
# Check questions about specific vehicle types.
    vehicle_words={ "cars":"car_count", "car": "car_count", "motorcycles":"motorcycle_count", "motorcycle": "motorcycle_count", "bikes":"motorcycle_count","bike": "motorcycle_count", "buses":"bus_count", "bus": "bus_count","trucks": "truck_counts", "truck":"truck_count",}

    for word, column in vehicle_words.items():
        if word in question_text:
            total = int(records[column].sum())
            latest_count = int(latest[column])
            return(
    f"There are{total}{word} detections across all saved records."
    f"In the latest saved iamge({latest['filename']})," 
    f"the count is {latest_count}."
)

    # Check questions about the latest image.
    if any(word in question_text for word in ["latest", "recent","last image"]):
        return(f"Latest saved iamge:{latest['filename']};" 
            f"observed at {latest['observed_at']}." 
            f"total vehicles:{int(latest['total_count'])}." 
            )
        return(
        "try asking:'How many cars?', 'How many buses?',"
    "'What is the total vehicle count?'," 
    "'Which image ahs the highest traffic?'," 
    "or 'Which hour is busiest?'")
    
setup_database()

        

