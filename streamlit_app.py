import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
import io
import os
from deepdream_core import deep_dream_static_image
from config import RUN_CONFIG

# Page configuration
st.set_page_config(
    page_title="DeepDream Generator",
    page_icon="🌈",
    layout="wide"
)

st.title("🌈 DeepDream Generator")
st.markdown("Upload an image or take a picture to create psychedelic DeepDream art!")

# Display current configuration
st.info(f"🎛️ **Current Settings:** {RUN_CONFIG['img_width']}px width, {RUN_CONFIG['pyramid_size']} pyramid levels, {RUN_CONFIG['num_gradient_ascent_iterations']} iterations, Layer: {list(RUN_CONFIG['layers_to_use'].keys())[0]}")

# Image input section
st.header("📸 Input Image")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'], 
        help="Upload a PNG, JPG, or JPEG image"
    )

with col2:
    st.subheader("Camera Capture")
    camera_input = st.camera_input("Take a picture")

# Function to process uploaded image
def process_image(image_source, source_type="upload"):
    """Convert uploaded image or camera input to numpy array"""
    if source_type == "upload" and image_source is not None:
        # Convert uploaded file to PIL Image
        pil_image = Image.open(image_source)
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert to numpy array
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return img_array
    
    elif source_type == "camera" and image_source is not None:
        # Camera input is already in the right format
        pil_image = Image.open(image_source)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return img_array
    
    return None

# Main processing section
st.header("🎨 Generate DeepDream")

# Determine which image source to use
img_array = None
source_name = ""

if uploaded_file is not None:
    img_array = process_image(uploaded_file, "upload")
    source_name = uploaded_file.name
    st.success(f"✅ Loaded uploaded image: {source_name}")
elif camera_input is not None:
    img_array = process_image(camera_input, "camera")
    source_name = "camera_capture"
    st.success("✅ Loaded camera image")

# Display input image if available
if img_array is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Image")
        st.image(img_array, caption="Input Image", use_column_width=True)
    
    # Generate button
    if st.button("🌈 Generate DeepDream", type="primary", use_container_width=True):
        with st.spinner("Generating DeepDream... This may take a few minutes"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing...")
                progress_bar.progress(10)
                
                # Generate DeepDream using config.yaml parameters
                status_text.text("Processing image...")
                progress_bar.progress(30)
                
                # Call the main function with our image
                dream_img = deep_dream_static_image(img=img_array)
                
                progress_bar.progress(90)
                status_text.text("Finalizing...")
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                # Display result
                with col2:
                    st.subheader("DeepDream Result")
                    st.image(dream_img, caption="DeepDream Output", use_column_width=True)
                
                # Provide download option
                st.success("✨ DeepDream generated successfully!")
                
                # Convert to PIL Image for download
                dream_pil = Image.fromarray((dream_img * 255).astype(np.uint8))
                
                # Create download buffer
                buf = io.BytesIO()
                dream_pil.save(buf, format='JPEG')
                byte_im = buf.getvalue()
                
                # Create filename with current config
                layer_name = list(RUN_CONFIG['layers_to_use'].keys())[0]
                st.download_button(
                    label="💾 Download DeepDream Image",
                    data=byte_im,
                    file_name=f"deepdream_{source_name}_{layer_name}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating DeepDream: {str(e)}")
                st.error("Please check your input image and try again.")

else:
    st.info("👆 Please upload an image or take a picture to get started!")

# Configuration note
st.markdown("---")
st.markdown("⚙️ **Want to change settings?** Edit the `config.yaml` file and restart the app to use different parameters.")