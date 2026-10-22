import streamlit as st 
import pandas as pd 

from main import(
    load_model,
    image_from_upload, # Converts an uploaded image into a format OpenCV can process.
    detect_vehicles, # Detects vehicles and counts each type.
    detect_edges,
    save_record, # Saves the results in the SQLite database.
    read_records, # Reads saved detection history.
    answer_question,
)

st.set_page_config(
    page_title= 'Vehicle Detection & Traffic Analysis.',
    page_icon ='🚗',
    layout = 'wide'
)

st.title('🚗 Vehicle detection and Analysis')
st.write('detect vehicles in images and analyse saved traffic records.')

@st.cache_resource # Load the YOLO model
def get_model():
    return load_model()

try:
    model = get_model()
except Exception as e:
    st.error(f'Could not load the YOLO model:{e}')
    st.stop()

st.sidebar.header('Detection Settings')

confidence = st.sidebar.slider(
    'confidence threshold',
    min_value = 0.10,
    max_value = 0.90,
    value = 0.25,
    step = 0.05,
)

tab1, tab2, tab3 = st.tabs(
    ['🔍 Detect Vehicles', '🚥🚦 traffic history', 'Ask Questions 🤔❓']
)

# tab 1
with tab1:
    st.subheader('Upload a traffic image')
    
    uploaded_file = st.file_uploader('Choose an image', type = ['jpg','jpeg','png'],)
    
    observed_at = st.datetime_input('When was this image observed?', value = pd.Timestamp.now().to_pydatetime(),)
    # This is intended to let users choose when the image was observed. The default value is the current date and time.
    if uploaded_file is not None:
        try:
            image = image_from_upload(uploaded_file)
            st.image(
                image,
                channels= 'BGR',
                caption ='Uploaded image',
                use_container_width = True, # lets the image use the available width.
            )
            
            if st.button('Detect Vehicles', type = 'primary'): # primary - Colored background with white text. Highly visible.
                with st.spinner('Detecting Vehicles...'):
                    counts, annotated_image = detect_vehicles(
                        image, model, confidence
                    )
                    # counts — a dictionary containing the number of cars, motorcycles, buses and trucks.
                    # annotated_image — the image with detection boxes and labels drawn on it.
                    save_record(uploaded_file.name, observed_at, counts,)
                    
                    st.success('Detection completed and saved!')
                    
                    st.subheader('detection Result')
                    
                    st.image(
                        annotated_image, 
                        caption = 'Detected Vehicles',
                        use_container_width= True
                    ) # Displays the image with the detected vehicles marked.
                    
                    st.subheader('Vehicle Counts')
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    col1.metric('Cars', counts['Car'])
                    col2.metric('Motorcycles', counts['Motorcycle'])
                    col3.metric('Buses', counts['Bus'])
                    col4.metric('Trucks', counts['Truck'])
                    col5.metric('Total', sum(counts.values()))
                    
                    st.subheader('Edge Detection')
                    
                    edges = detect_edges(image)
                    
                    st.image(edges,
                            caption = 'Canny edge detection', 
                            clamp = True,
                            use_container_width = True,
                            )
            
        except Exception as e:
            st.error(f'Error processing the image: {e}')
            
# traffic history

with tab2:
    st.subheader('Saved Traffic records')
    
    try:
        records = read_records() # This reads the saved database records and returns them as a Pandas DataFrame.
        
        if records.empty:
            st.info('No saved records yet. Detect a vehicle image first.')
        else:
            col1, col2 = st.columns(2)
            
            col1.metric('Saved Images', len(records)) # len(records) counts the saved rows.
            col2.metric('Total vehicle Detection',
                        int(records['total_count'].sum()), # adds up the total detections across all saved records.
                        )
            
            st.dataframe(
                records,
                use_container_width = True,
                hide_index = True,
            ) # Displays the database records in a table.
            
            st.subheader('Vehicle Counts By Inage')
            
            chart_data = records.set_index('filename')[
                [
                    'car_count',
                    'motorcycle_count',
                    'bus_count',
                    'truck_count',
                ]
            ]
            
            st.bar_chart(chart_data) # displays their values as a bar chart.
            
            csv_data = records.to_csv(index= False).encode('utf-8')
            # Converts the table to CSV text and then to bytes, which can be downloaded.
            st.download_button(
                'Download history as csv',
                data = csv_data,
                file_name = 'traffic_history.csv',
                mime = 'text/csv'
            )
            
            if st.button('Refresh History'):
                st.rerun()
                
    except Exception as e:
        st.error(f'Could not load traffic history: {e}')
        
# ask questions

with tab3:
    st.subheader('Ask about Traffic History')
    
    st.write('Ask about saved vehicle detctions,'
            'the highest- count image, or the busiest hour.')
    
    question = st.text_input('Enter your question',
                            placeholder = 'Example: How many cars?',)
    
    if st.button('Get Answer'):
        if not question.strip(): # question.strip() removes spaces from the beginning and end.
            st.warning('Please enter a question')
            
        else:
            try:
                records = read_records()
                answer = answer_question(question, records)
                st.info(answer)
            
            except Exception as e:
                st.error(f'Could you answer the question: {e}')












