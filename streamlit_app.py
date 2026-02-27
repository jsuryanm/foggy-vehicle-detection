import streamlit as st
import requests
import tempfile
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Fog Vehicle Detection", layout="wide")

st.title("Fog-Resilient Vehicle Detection System")

# Select input type FIRST
input_type = st.radio("Select Input Type", ["Image", "Video"])

conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou = st.slider("IoU Threshold", 0.0, 1.0, 0.6, 0.05)

uploaded_file = st.file_uploader(
    "Upload File",
    type=["jpg", "jpeg", "png", "mp4"]
)

# IMAGE FLOW
if uploaded_file and input_type == "Image":

    if uploaded_file.type.startswith("image"):

        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        if st.button("Run Detection"):

            with st.spinner("Running detection..."):
                response = requests.post(
                    f"{API_URL}/predict/image",
                    files={"file": uploaded_file.getvalue()},
                    data={"conf": conf, "iou": iou}
                )

            if response.status_code == 200:
                data = response.json()

                st.image(
                    data["image_path"],
                    caption="Detected Output",
                    use_column_width=True
                )

                st.subheader("Detection Summary")
                st.json(data["counts"])
                st.write(f"Inference Time: {data['inference_time']} seconds")
            else:
                st.error("Prediction failed")

    else:
        st.warning("Please upload an image file for Image mode.")


# VIDEO FLOW
if uploaded_file and input_type == "Video":

    if uploaded_file.type.startswith("video"):

        st.video(uploaded_file)

        if st.button("Run Detection"):

            with st.spinner("Processing video..."):
                response = requests.post(
                    f"{API_URL}/predict/video",
                    files={"file": uploaded_file.getvalue()},
                    data={"conf": conf, "iou": iou}
                )

            if response.status_code == 200:

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(response.content)

                st.video(tmp.name)

                counts = json.loads(response.headers.get("X-Counts", "{}"))
                inference_time = response.headers.get("X-Inference-Time", "0")

                st.subheader("Detection Summary")
                st.json(counts)
                st.write(f"Total Processing Time: {inference_time} seconds")

            else:
                st.error("Video prediction failed ")

    else:
        st.warning("Please upload a video file for Video mode.")