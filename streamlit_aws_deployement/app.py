import streamlit as st
import os
import base64
import tempfile
from PIL import Image
from textile_core import predict_image
import cv2
from streamlit_option_menu import option_menu
import shutil
from streamlit_webrtc import webrtc_streamer
import av
import sys
import platform

# Helper function to encode image to base64
def _get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Set Streamlit page config
#st.set_page_config(page_title="Insight Wave", layout="centered")
st.set_page_config(page_title="Insight Wave", page_icon="insight_wave.jpeg", layout="wide", initial_sidebar_state="expanded")



st.markdown(
    """
    <style>
    /* Global Styling */
    body {
        background-color: #1e1e1e;
        color: white;
        padding-top: 100px;
    }
    
    header {visibility: hidden;}

    /* Header Banner */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #24005e;
        padding: 36px 20px;  
        border-radius: 10px;
        color: white;
        font-size: 24px;
        font-weight: bold;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }

    /* Header Title */
    .header-title {
        text-align: center;   
        color: white;
        font-size: 40px;
        padding-bottom: 100px; /* Adds space below */
    }

    .header-logo {
        position: absolute;
        left: 20px;  
        width: 150px;
        border-radius: 10px;
    }

    /* Main Content */
    .main-content {
        margin-top: 100px;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #24005e;
        color: white;
        padding: 10px;
        text-align: center;
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 50px;
        z-index: 1000;
    }

    .footer-icons {
        display: flex;
        margin-top: 10px;
        position: absolute;
        left: 15px;
        padding-left: 100px;
        padding-top: 15px;
    }

    .footer-icons img {
        width: 26px;
        margin-right: 26px;
    }

    /* Prediction Box */
    .prediction-box {
        width: 60%;
        margin: auto;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        color: black;
    }

    .spacer {
        margin-top: 80px;
    }

    .info-box {
        margin-bottom: 0px !important;
    }

    /* Purple Box */
    .purple-box {
        background-color: #24005e;  
        color: white;  
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        font-family: Arial, sans-serif;
        line-height: 1.6;
        margin-top: 20px;
    }

    .purple-box p, .purple-box ul {
        margin-bottom: 10px;
    }

    .purple-box ul {
        padding-left: 20px;
    }

    .purple-box strong {
        font-size: 18px;
        color: #ffcc00;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# Get the current directory and logo path
logo_path = os.path.join(os.path.dirname(__file__), "insight_wave.jpeg")


# Display header with logo and centered title
st.markdown(
    f"""
    <div class="header">
        <img src="data:image/png;base64,{_get_image_base64(logo_path)}" class="header-logo">
        <h1 class="header-title">Textile Manufacturing App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# Webcam classification function
def classify_frame(frame):
    """Process frame and classify using the model."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)
    class_name, confidence = predict_image(temp_file.name)
    return class_name, confidence



# Encode the image before inserting it
website_logo = _get_image_base64("insight_wave.jpeg")

# Sidebar using streamlit_option_menu
with st.sidebar:
    # Add styled image
    st.markdown(
        f"""
        <style>
            .custom-image {{
                position: relative; 
                top: -30px;  /* Adjust padding */
                display: block;
                margin: auto; /* Centering */
                width: 130px;
            }}
        </style>
        
        <img class="custom-image" src="data:image/png;base64,{website_logo}">  
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")  # Creates a visual divider

    # Sidebar menu
    selected_option = option_menu(
        menu_title="NAV",  # Sidebar title
        options=["Upload Image","Guide", "Upload Video", "Real_Time Classification"],  # Menu options
        icons=["upload", "upload", "camera"],  # Icons for each option
        menu_icon="cast",  # Sidebar menu icon
        default_index=0,  # Default selected option
        styles={
            "nav-link-selected": {
                "background-color": "#24005e",
                "color": "white",
                "font-weight": "bold",
            },
            "nav-link": {
                "background-color": "#24005e",
                "color": "white",
                "font-size": "18px",
                "border-radius": "7px",
                "padding": "6px",
                "transition": "0.01s",
                "font-family": "Arial, sans-serif",
            },
            "nav-link:hover": {
                "background-color": "#8f00ff",
                "color": "#fff",
                "transform": "scale(1.05)",
            },
        },
    )


    




# Main Content Wrapper
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Add extra space between title and instruction
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)


# Option 1: Upload Image
if selected_option == "Upload Image":
    
    st.info("📂 Please upload an image to start classification.", icon="📂")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Convert uploaded file to OpenCV format
        temp_dir = "temp_dir"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image
        img = Image.open(file_path)
        st.image(img,   use_container_width=True)

        # Classify the image
        st.write(" **Classifying... Please wait.**")
        with st.spinner("⏳ Processing..."):
            try:
                # Prediction
                class_name, confidence = predict_image(file_path)

                # Color map for different classes
                color_map = {
                    "Good": "#2ECC71",  # Green
                    "Hole": "#F1C40F",  # Yellow
                    "Objects": "#E67E22",  # Orange
                    "Oil Spot": "#E74C3C",  # Red
                    "Thread Error": "#9B59B6"  # Purple
                }

                # Get the color based on the class
                prediction_color = color_map.get(class_name, "#3498DB")  # Default blue

                # Display prediction result in a styled box
                st.markdown(
                    f"""
                        <div class='prediction-box' style='background-color: {prediction_color};'>
                            🏷️ Prediction: {class_name}
                        </div>
                        """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

def show_guide():
    st.title("System Requirements")

    # Purple Box Container for Entire Content
    st.markdown(
        """
        <div style="background-color: #24005e; color: white; padding: 6px; border-radius: 10px;">
        """,
        unsafe_allow_html=True
    )

    # **Latency**
    st.markdown("### Latency", unsafe_allow_html=True)
    st.write("- **Image resolution/file type:** .png/.jpg")
    st.write("- **Communication protocol:** Bandwidth (TCP-IP, COM)")
    st.write("- **Preprocessing time:** Processing power (CPU/GPU)")
    st.write("- **Inference time:** Model architecture (YOLO/LLM)")

    st.markdown("**Note:** Local is always faster than cloud because the sensors are local.", unsafe_allow_html=True)

    # **Storage Example**
    st.markdown("### Storage Example")
    st.write("- **Image size:** 5 MB")
    st.write("- **FPS:** 5")
    st.write("- **Time:** 16H per day for one month")

    # **Storage Calculation**
    st.markdown("### Storage Calculation")
    st.code("5 * 5 * 3600 * 16 * 30 = 42,187.5 GB ≈ 42 TB", language="python")

    # **Hardware Requirements**
    st.markdown("### Key Hardware Requirements")
    st.write("Recommended specs for **Inference** in Image Classification, Object Detection, and Time Series Regression:")

    # **CPU**
    st.markdown("#### CPU")
    st.write("**Light workloads (few inferences per second):**")
    st.write("- Intel i7 (10th Gen+) / AMD Ryzen 7 (5000 series+)")
    st.write("- Minimum: 6 cores / 12 threads")
    
    st.write("**Continuous real-time inference:**")
    st.write("- Intel i9 / AMD Ryzen 9 / Threadripper (16+ cores)")
    st.write("- Xeon/EPYC for server-grade stability")

    # **GPU**
    st.markdown("#### GPU")
    st.write("- **NVIDIA RTX 3060 / 4060+** (light inference, batch size 1-2)")
    st.write("- **NVIDIA RTX 3090 / 4090 / A6000** (continuous inference, 30+ FPS)")
    st.write("- **Edge Devices:** Jetson Xavier NX / Jetson Orin (low power inference)")

    # **RAM**
    st.markdown("#### RAM")
    st.write("- Minimum: **16GB** (for light inference)")
    st.write("- Recommended: **32GB+** (for multiple streams & higher resolutions)")

    # **Storage**
    st.markdown("#### Storage")
    st.write("- **SSD (1TB+ NVMe):** Fast read/write speeds")
    st.write("- **HDD:** If long-term video storage is needed")

    st.markdown("**Note:** Some applications **don’t need a GPU for inference**, especially on edge devices that **only collect data** instead of processing it.", unsafe_allow_html=True)

    # Closing the Purple Box
    st.markdown("</div>", unsafe_allow_html=True)

# Add this function call where you manage your sidebar options
if selected_option == "Guide":
    show_guide()

def process_uploaded_video(video_file):
    """Handles video processing and classification, then returns a downloadable processed video."""
    st.subheader("Processing Video...")
    
    # Create a temporary directory to store video files
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input_video.mp4")
    output_path = os.path.join(temp_dir, "output_video.mp4")
    
    # Save the uploaded video
    with open(input_path, "wb") as f:
        f.write(video_file.read())
    
    # Initialize video processing
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Classify the frame and overlay text
        label, confidence = classify_frame(frame)
        cv2.putText(
            frame, f"{label} ({confidence:.2f}%)", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # Display the processed frame
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    st.success("✅ Video processing completed!")
    
    # Provide download option
    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f.read(), "processed_video.mp4", "video/mp4")
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)



# Video Upload Section
if selected_option == "Upload Video":
    st.info("📂 Upload a video to start classification.", icon="📂")
    uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("▶ Start Video Processing"):
            process_uploaded_video(uploaded_video)

# Option 3: Real_Time Classification
elif selected_option == "Real_Time Classification":
    st.title("Real Time Classification 🎥")

    # Initialize session state variables
    if 'live_classifying' not in st.session_state:
        st.session_state.live_classifying = False
    if 'video_file_path' not in st.session_state:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        st.session_state.video_file_path = temp_video.name
        temp_video.close()

    # Buttons for controlling classification
    start_button = st.button("▶ Start Classification")
    stop_button = st.button("⏹ Stop Classification")

    if start_button:
        st.session_state.live_classifying = True

    if stop_button:
        st.session_state.live_classifying = False

    # Start video capture
    if st.session_state.live_classifying:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access the webcam.")
        else:
            stframe = st.empty()
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
            fps = 20.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(st.session_state.video_file_path, fourcc, fps, (frame_width, frame_height))

            while cap.isOpened() and st.session_state.live_classifying:
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠ Video feed lost.")
                    break

                # Resize frame for consistent processing
                frame_resized = cv2.resize(frame, (64, 64))

                # Predict class
                class_name, confidence = classify_frame(frame_resized)

                # Overlay text on frame
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert OpenCV image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                # Save frame to video file
                out.write(frame)

            cap.release()
            
            

    else:
        st.info('ℹ️ Press " ▶ Start Classification" to begin.')

 

# Close main content wrapper
st.markdown("</div>", unsafe_allow_html=True)

# Footer with Contact Us section
st.markdown(
    f"""
    <div class="footer">
        <p style="margin-bottom:12px;"><strong><u>Contact Us</u></strong></p>
        <div class="footer-icons">
            <a href="https://www.facebook.com/InsightMindMatrix" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook">
            </a>
            <a href="https://www.linkedin.com/company/insight-mind-matrix/?lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BQ1SdG%2FXITMCIh1yKZo3YRw%3D%3D" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn">
            </a>
            <a href="mailto:info@insightmindmatrix.com" target="_blank"style="margin-right:20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
            </a>
            <a href="https://insightmindmatrix.com/" target="_blank" style="color:white;margin-left:70px;"><strong id="about">about us</strong>
                <img src="data:image/png;base64,{_get_image_base64("logo.png")}" style="margin-left: 10px; width:70px; margin-bottom:15px;"> 
            </a>
        </div>
        <p><i>© 2025 Textile Classification App. | All Rights Reserved to Insight Mind Matrix </i></p>
    </div>
    """,
    unsafe_allow_html=True,
)