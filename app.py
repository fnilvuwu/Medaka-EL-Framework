import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ensemble_utils import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted

st.set_page_config(layout="wide", page_title="Ensemble Models Inference")

st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("Ensemble Models Inference Dashboard")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_DIR = os.path.join(BASE_DIR, "Ensemble Models")

# Sidebar
st.sidebar.header("Settings")

categories = [d for d in os.listdir(ENSEMBLE_DIR) if os.path.isdir(os.path.join(ENSEMBLE_DIR, d))]

all_available_models = {}
for cat in categories:
    cat_dir = os.path.join(ENSEMBLE_DIR, cat)
    if os.path.isdir(cat_dir):
        for f in os.listdir(cat_dir):
            if f.endswith(".pt"):
                all_available_models[f"{cat} / {f}"] = os.path.join(cat_dir, f)

st.sidebar.subheader("Model Selection")

group_options = ["All 13 Models"] + categories + ["Custom Selection"]
selected_group = st.sidebar.selectbox("Quick Select Group", group_options)

if selected_group == "All 13 Models":
    default_selection = list(all_available_models.keys())
elif selected_group == "Custom Selection":
    default_selection = []
else:
    default_selection = [k for k in all_available_models.keys() if k.startswith(selected_group)]

selected_models_keys = st.sidebar.multiselect(
    "Select Models for Ensemble",
    options=list(all_available_models.keys()),
    default=default_selection
)

st.sidebar.subheader("Ensemble Settings")
ensemble_methods = ["WBF", "NMS", "Soft-NMS", "NMW"]
selected_method = st.sidebar.selectbox("Select Ensemble Method", ensemble_methods)

conf_thr = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thr = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def draw_boxes(image, boxes, scores, labels, class_names=None):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = scores[i]
        label = int(labels[i])
        class_name = class_names[label] if class_names and label < len(class_names) else str(label)
        
        # Color logic: Green for class 0, Blue for class 1
        if label == 1:
            color = (0, 0, 255)  # Blue in RGB
        else:
            color = (0, 255, 0)  # Green in RGB
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if not selected_models_keys:
        st.warning("Please select at least one model from the sidebar.")
    else:
        st.subheader("Individual Model Performances")
        
        cols = st.columns(4)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        class_names_dict = {}
        
        for i, m_key in enumerate(selected_models_keys):
            model_path = all_available_models[m_key]
            model = YOLO(model_path)
            
            # Predict
            results = model.predict(image, conf=conf_thr, verbose=False)
            res = results[0]
            
            if not class_names_dict:
                class_names_dict = res.names
            
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            labels = res.boxes.cls.cpu().numpy()
            
            # Draw individual
            img_drawn = draw_boxes(img_np, boxes, scores, labels, class_names_dict)
            
            col_idx = i % 4
            if i > 0 and col_idx == 0:
                cols = st.columns(4)
                
            model_name_only = m_key.split(" / ")[1]
            cols[col_idx].image(img_drawn, caption=f"Model: {model_name_only}", use_container_width=True)
            
            # Normalize boxes for ensemble
            if len(boxes) > 0:
                norm_boxes = boxes.copy()
                norm_boxes[:, 0] /= w
                norm_boxes[:, 1] /= h
                norm_boxes[:, 2] /= w
                norm_boxes[:, 3] /= h
                
                all_boxes.append(norm_boxes.tolist())
                all_scores.append(scores.tolist())
                all_labels.append(labels.tolist())
            else:
                all_boxes.append([])
                all_scores.append([])
                all_labels.append([])
            
        st.subheader(f"Ensemble Result ({selected_method})")
        
        if len(all_boxes) > 0:
            if selected_method == "WBF":
                ens_boxes, ens_scores, ens_labels = weighted_boxes_fusion(
                    all_boxes, all_scores, all_labels, 
                    weights=None, iou_thr=iou_thr, skip_box_thr=0.0
                )
            elif selected_method == "NMS":
                ens_boxes, ens_scores, ens_labels = nms(
                    all_boxes, all_scores, all_labels, 
                    iou_thr=iou_thr
                )
            elif selected_method == "Soft-NMS":
                ens_boxes, ens_scores, ens_labels = soft_nms(
                    all_boxes, all_scores, all_labels, 
                    iou_thr=iou_thr
                )
            elif selected_method == "NMW":
                ens_boxes, ens_scores, ens_labels = non_maximum_weighted(
                    all_boxes, all_scores, all_labels, 
                    weights=None, iou_thr=iou_thr, skip_box_thr=0.0
                )
            
            if len(ens_boxes) > 0:
                # Filter by confidence threshold
                valid_idx = ens_scores >= conf_thr
                ens_boxes = ens_boxes[valid_idx]
                ens_scores = ens_scores[valid_idx]
                ens_labels = ens_labels[valid_idx]
                
            if len(ens_boxes) > 0:
                # Denormalize
                ens_boxes[:, 0] *= w
                ens_boxes[:, 1] *= h
                ens_boxes[:, 2] *= w
                ens_boxes[:, 3] *= h
                
                img_ens = draw_boxes(img_np, ens_boxes, ens_scores, ens_labels, class_names_dict)
                st.image(img_ens, caption=f"Ensembled with {selected_method}", use_container_width=True)
            else:
                st.info("No bounding boxes remained after ensemble.")
