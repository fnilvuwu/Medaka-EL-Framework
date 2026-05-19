import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ensemble_utils import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted

st.set_page_config(layout="wide", page_title="Medaka-EL Framework")

st.markdown(
    """
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
        padding: 6px 10px;
        font-size: 12px;
        line-height: 1.1;
        transition: all 0.15s;
        min-width: 64px;
        max-width: 96px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transform: scale(1.02);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Ensemble Models Inference Dashboard")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_DIR = os.path.join(BASE_DIR, "Ensemble Models")

# Sidebar
st.sidebar.header("Settings")

categories = [
    d for d in os.listdir(ENSEMBLE_DIR) if os.path.isdir(os.path.join(ENSEMBLE_DIR, d))
]

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
    default_selection = [
        k for k in all_available_models.keys() if k.startswith(selected_group)
    ]

selected_models_keys = st.sidebar.multiselect(
    "Select Models for Ensemble",
    options=list(all_available_models.keys()),
    default=default_selection,
)

st.sidebar.subheader("Ensemble Settings")
ensemble_methods = ["WBF", "NMS", "Soft-NMS", "NMW"]
selected_method = st.sidebar.selectbox("Select Ensemble Method", ensemble_methods)

conf_thr = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thr = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Test images gallery (3 rows x 5 cols) shown above the main image area.
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test-images")
test_images = []
if os.path.isdir(TEST_IMAGES_DIR):
    # collect files named like test-1.jpg ... test-15.jpg (support common image extensions)
    files = sorted(
        [
            f
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.lower().startswith("test-")
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    test_images = files[:15]

if "selected_test_image" not in st.session_state:
    st.session_state["selected_test_image"] = None

st.sidebar.subheader("Pick a test image")
if test_images:
    for row in range(3):
        cols = st.sidebar.columns(5)
        for col_idx in range(5):
            idx = row * 5 + col_idx
            if idx < len(test_images):
                img_name = test_images[idx]
                img_path = os.path.join(TEST_IMAGES_DIR, img_name)
                try:
                    thumb = Image.open(img_path).convert("RGB")
                except Exception:
                    thumb = None

                with cols[col_idx]:
                    if thumb is not None:
                        st.image(thumb, width=92)
                    else:
                        st.write("[missing]")
                    # button under the thumbnail (use the sidebar column directly)
                    if cols[col_idx].button(f"Test {idx+1}", key=f"select_test_{idx}"):
                        st.session_state["selected_test_image"] = img_path
                        # Streamlit automatically reruns on interaction; call guarded rerun for compatibility
                        if hasattr(st, "experimental_rerun"):
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
else:
    st.sidebar.info(
        "No test images found in test-images/ (expecting files like test-1.jpg ... test-15.jpg)"
    )


def draw_boxes(image, boxes, scores, labels, class_names=None):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = scores[i]
        label = int(labels[i])
        class_name = (
            class_names[label]
            if class_names and label < len(class_names)
            else str(label)
        )

        # Color logic: Green for class 0, Blue for class 1
        if label == 1:
            color = (0, 0, 255)  # Blue in RGB
        else:
            color = (0, 255, 0)  # Green in RGB

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(
            img, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return img


def make_square_pil(img_pil, size, bg_color=(13, 17, 23)):
    """Resize and pad a PIL image to a square of given size preserving aspect ratio."""
    # ensure RGB
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    # scale to fit within size
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    # create background and paste centered
    new_img = Image.new("RGB", (size, size), bg_color)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


selected_test = st.session_state.get("selected_test_image")
if uploaded_file is not None and selected_test is not None:
    st.session_state["selected_test_image"] = None
    selected_test = None
# Sidebar preview of selected/uploaded image
if uploaded_file is not None:
    try:
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.sidebar.subheader("Uploaded Image")
        st.sidebar.image(preview_img, width=200)
    except Exception:
        pass
elif selected_test:
    try:
        preview_img = Image.open(selected_test).convert("RGB")
        st.sidebar.subheader("Selected Image")
        st.sidebar.image(preview_img, width=200)
    except Exception:
        pass
if uploaded_file is not None or selected_test:
    # Layout: original image (left) and ensembled result (right)
    left_col, right_col = st.columns([1, 1])

    # Read image from uploaded file or selected test image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open(selected_test).convert("RGB")

    img_np = np.array(image)
    h, w, _ = img_np.shape

    # render square 1:1 images
    S = 480
    left_col.subheader("Original Image")
    if selected_test:
        left_col.caption(f"Using test image: {os.path.basename(selected_test)}")
    if uploaded_file:
        left_col.caption(f"Using uploaded image: {uploaded_file.name}")
    # image is a PIL.Image
    left_square = make_square_pil(image, S)
    left_col.image(left_square, width=S)

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
            cols[col_idx].image(
                img_drawn, caption=f"Model: {model_name_only}", width=240
            )

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

        if len(all_boxes) > 0:
            if selected_method == "WBF":
                ens_boxes, ens_scores, ens_labels = weighted_boxes_fusion(
                    all_boxes,
                    all_scores,
                    all_labels,
                    weights=None,
                    iou_thr=iou_thr,
                    skip_box_thr=0.0,
                )
            elif selected_method == "NMS":
                ens_boxes, ens_scores, ens_labels = nms(
                    all_boxes, all_scores, all_labels, iou_thr=iou_thr
                )
            elif selected_method == "Soft-NMS":
                ens_boxes, ens_scores, ens_labels = soft_nms(
                    all_boxes, all_scores, all_labels, iou_thr=iou_thr
                )
            elif selected_method == "NMW":
                ens_boxes, ens_scores, ens_labels = non_maximum_weighted(
                    all_boxes,
                    all_scores,
                    all_labels,
                    weights=None,
                    iou_thr=iou_thr,
                    skip_box_thr=0.0,
                )

            if len(ens_boxes) > 0:
                # Filter by confidence threshold (round to 2 decimal places to match display)
                valid_idx = np.round(ens_scores, 2) >= conf_thr
                ens_boxes = ens_boxes[valid_idx]
                ens_scores = ens_scores[valid_idx]
                ens_labels = ens_labels[valid_idx]

            if len(ens_boxes) > 0:
                # Denormalize
                ens_boxes[:, 0] *= w
                ens_boxes[:, 1] *= h
                ens_boxes[:, 2] *= w
                ens_boxes[:, 3] *= h

                img_ens = draw_boxes(
                    img_np, ens_boxes, ens_scores, ens_labels, class_names_dict
                )
                # convert ensembled numpy image to PIL and make square
                try:
                    pil_ens = Image.fromarray(img_ens)
                except Exception:
                    pil_ens = Image.fromarray(img_ens.astype("uint8"))
                right_col.subheader(f"Ensembled Result ({selected_method})")
                right_col.caption(f"Ensembled with {selected_method}")

                right_square = make_square_pil(pil_ens, S)
                right_col.image(
                    right_square,
                    width=S,
                )
            else:
                right_col.info("No bounding boxes remained after ensemble.")
