# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import numpy as np
import joblib
import time
import os
import requests
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Nexus AI Traffic Control", layout="wide", page_icon="🚦", initial_sidebar_state="expanded")

# ---------------- ASSETS & LOTTIE ----------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_url = "https://lottie.host/802613b5-3642-498c-859a-18e38596634c/pYVj8V87fL.json"
lottie_traffic = load_lottieurl(lottie_url)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    
    /* Background & Global styling */
    .stApp { 
        background: radial-gradient(circle at top right, #111827, #000000); 
        color: #f8fafc; 
    }
    
    /* Hide top header and default footer */
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Sleek Title */
    .main-title {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-weight: 300;
        margin-top: -20px;
        margin-bottom: 50px;
        font-size: 1.2rem;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.2);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    .metric-card h4 {
        color: #cbd5e1;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .metric-card h1 {
        color: #00f2fe;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.5);
    }
    
    /* Traffic Lights UI */
    .traffic-pole {
        background: #1e293b;
        width: 120px;
        border-radius: 30px;
        padding: 20px 10px;
        margin: 0 auto;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8), 0 10px 30px rgba(0,0,0,0.5);
        border: 2px solid #334155;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }
    
    .light-bulb {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #0f172a;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.9);
        border: 2px solid #000;
        transition: all 0.3s ease;
    }
    
    /* Active Lights with Neon Glow */
    .light-bulb.red.active { 
        background: radial-gradient(circle, #ff4b4b, #990000); 
        box-shadow: 0 0 40px #ff4b4b, inset 0 0 10px rgba(255,255,255,0.6); 
        border-color: #ff4b4b;
    }
    .light-bulb.yellow.active { 
        background: radial-gradient(circle, #fbbf24, #b45309); 
        box-shadow: 0 0 40px #fbbf24, inset 0 0 10px rgba(255,255,255,0.6); 
        border-color: #fbbf24;
    }
    .light-bulb.green.active { 
        background: radial-gradient(circle, #10b981, #047857); 
        box-shadow: 0 0 40px #10b981, inset 0 0 10px rgba(255,255,255,0.6); 
        border-color: #10b981;
    }
    
    /* Timer Display */
    .timer-display {
        font-family: 'Courier New', monospace;
        font-size: 2.5rem;
        font-weight: bold;
        color: #fff;
        background: #000;
        padding: 5px 15px;
        border-radius: 10px;
        border: 2px solid #334155;
        margin-top: 15px;
        min-width: 80px;
        text-align: center;
        text-shadow: 0 0 10px rgba(255,255,255,0.5);
    }
    
    /* Lane Box for Signal execution */
    .lane-box {
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 30px 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 10px;
    }
    
    .lane-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 20px;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    model_path = "signal_model.pkl" if os.path.exists("signal_model.pkl") else os.path.join("models", "signal_model.pkl")
    try:
        signal_model = joblib.load(model_path)
    except:
        signal_model = None 
    return yolo_model, signal_model

yolo_model, signal_model = load_models()

# ---------------- LIVE VEHICLE COUNT WITH VIDEO ----------------
def process_video_live(video_file, image_placeholder, metric_placeholder, lane_id):
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file_path = temp_file.name
    temp_file.close() 

    cap = cv2.VideoCapture(temp_file_path)
    frame_skip = 6
    frame_id = 0
    
    counted_ids = set()
    recent_cx = []
    
    try:
        target_lane_cx = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_id += 1
            
            height, width, _ = frame.shape
            if target_lane_cx is None: 
                target_lane_cx = width / 2
            
            # imgsz=736 drastically improves detection of heavily occluded vehicles in dense traffic.
            # custom_tracker.yaml uses ByteTrack logic with a 120-frame buffer to prevent ID loss without the massive CPU overhead of BotSORT's Optical Flow!
            results = yolo_model.track(frame, persist=True, tracker="custom_tracker.yaml", classes=[1, 2, 3, 5, 7, 9], conf=0.15, iou=0.75, imgsz=736, verbose=False)[0]
            
            current_frame_boxes = []
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu()
                track_ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.cpu().tolist()
                
                # Step 1: Dynamically Anchor Target Lane using Traffic Density (Histogram)
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    label = yolo_model.names[int(cls)]
                    if label in ["bicycle", "car", "bus", "truck", "motorcycle"]:
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        
                        # Hard Train Rejection
                        if w > width * 0.25 and w > h * 2.5:
                            continue
                            
                        recent_cx.append((x1 + x2) / 2)
                        if len(recent_cx) > 400: recent_cx.pop(0)
                        
                # The peak density of traffic determines the exact center of our target lane dynamically!
                if len(recent_cx) > 20:
                    hist, bins = np.histogram(recent_cx, bins=15, range=(0, width))
                    peak_bin = np.argmax(hist)
                    target_lane_cx = (bins[peak_bin] + bins[peak_bin+1]) / 2
                            
                # Step 2: Strict Geometric Filtering with Perspective!
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    label = yolo_model.names[int(cls)]
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    cx = (x1 + x2) / 2
                    
                    if label not in ["bicycle", "car", "bus", "truck", "motorcycle", "traffic light"]:
                        continue
                        
                    # Hard Train Rejection
                    if w > width * 0.25 and w > h * 2.5:
                        continue
                        
                    if label == "traffic light":
                        current_frame_boxes.append((box, track_id, label))
                        continue
                        
                    # Expanded Dynamic Margin: 
                    # If traffic is heavily jammed, it tends to spill wide. 
                    # We use a very generous 70% width margin at the bottom to ensure ZERO valid vehicles are cropped!
                    dynamic_margin = width * 0.20 + (y2 / height) * (width * 0.35)
                    is_valid_lane = abs(cx - target_lane_cx) < dynamic_margin
                                    
                    if is_valid_lane:
                        # Removed temporal persistence filter! 
                        # In heavy traffic, tracking IDs flicker constantly due to massive occlusion.
                        # Forcing a vehicle to survive 3+ frames with the SAME ID will cause massive undercounting!
                        counted_ids.add(track_id)
                        current_frame_boxes.append((box, track_id, label))
                        
            # Update UI only every N frames
            if frame_id % frame_skip == 0:
                annotated_frame = frame.copy()
                for box, track_id, label in current_frame_boxes:
                    x1, y1, x2, y2 = box
                    if label == "traffic light":
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                        cv2.putText(annotated_frame, "Traffic Light", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{label}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                metric_placeholder.markdown(f"""
                <div class='metric-card'>
                    <h4>LANE {lane_id} DETECTIONS</h4>
                    <h1>{len(counted_ids)}</h1>
                </div>
                """, unsafe_allow_html=True)
                
    finally:
        cap.release()
        time.sleep(0.2)
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass 
                
    return len(counted_ids)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    if lottie_traffic:
        st_lottie(lottie_traffic, height=180, key="traffic_anim")
    st.markdown("""
    <div style='text-align: center; margin-top: -20px;'>
        <h2 style='color: #00f2fe; font-weight: 800;'>NEXUS AI</h2>
        <p style='color: #94a3b8;'>Smart City Infrastructure</p>
    </div>
    <hr style='border-color: rgba(255,255,255,0.1);'>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### ⚙️ System Status
    🟢 **AI Core:** Online  
    🟢 **Vision Model:** YOLOv8 Nano  
    🟢 **Decision Matrix:** Random Forest  
    """)
    st.info("Upload 4 video feeds to dynamically calculate optimal traffic flow.")

# ---------------- MAIN UI ----------------
st.markdown("<div class='main-title'>NEXUS AI ADAPTIVE TRAFFIC CONTROL</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time vision processing & dynamic signal optimization</div>", unsafe_allow_html=True)

# Grid for File Uploads and Live Feeds
cols = st.columns(4)
lane_uploads = []
image_placeholders = []
metric_placeholders = []

for i, col in enumerate(cols):
    with col:
        st.markdown(f"<h3 style='text-align: center; color: #cbd5e1;'>LANE {i+1}</h3>", unsafe_allow_html=True)
        up = st.file_uploader(f"Video {i+1}", key=f"l{i}", label_visibility="collapsed")
        lane_uploads.append(up)
        
        # Placeholders for the live video and live count
        image_placeholders.append(st.empty())
        metric_placeholders.append(st.empty())

st.markdown("<br>", unsafe_allow_html=True)
initiate_btn = st.button("🚀 INITIATE SMART ANALYSIS", use_container_width=True)

if initiate_btn:
    if all(lane_uploads):
        counts = []
        
        # Step 1: Live Processing
        st.markdown("### 🔍 Live Vision Analysis")
        # Process each lane sequentially so judges can watch the YOLO feed clearly
        for i, video in enumerate(lane_uploads):
            with st.spinner(f"Analyzing Lane {i+1} Feed..."):
                count = process_video_live(video, image_placeholders[i], metric_placeholders[i], i+1)
                counts.append(count)
        
        st.success("✅ Vision Analysis Complete! Calculating Optimal Signal Flow...")
        
        # Inject JavaScript to auto-scroll to the LIVE SIGNAL EXECUTION section
        st.components.v1.html("""
            <script>
                var elements = window.parent.document.querySelectorAll('h2');
                for (var i = 0; i < elements.length; i++) {
                    if (elements[i].innerText.includes('LIVE SIGNAL EXECUTION')) {
                        elements[i].scrollIntoView({behavior: 'smooth', block: 'start'});
                        break;
                    }
                }
            </script>
        """, height=0)
        
        # Step 2: Predict Timing
        if signal_model:
            pred = signal_model.predict([counts])[0]
        else:
            total = sum(counts) if sum(counts) > 0 else 1
            pred = [max(10, (c/total)*100) for c in counts]
            
        # Step 3: Interactive Signal Simulation
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; font-weight: 800; color: #fff;'>🚦 LIVE SIGNAL EXECUTION</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        sim_placeholder = st.empty()
        
        lane_order = sorted([{"id": i, "time": int(t)} for i, t in enumerate(pred)], key=lambda x: x['time'], reverse=True)

        for active in lane_order:
            duration = active['time']
            # We add a 2-second Yellow light transition before switching to the next green
            for sec in range(duration, -3, -1):
                # Determine state: Green, Yellow, or Red
                state = "green" if sec > 0 else "yellow"
                display_sec = max(sec, 0) if state == "green" else abs(sec)
                
                with sim_placeholder.container():
                    r1 = st.columns(4)
                    
                    for i in range(4):
                        is_active = (active['id'] == i)
                        
                        # Set light states
                        is_red = not is_active
                        is_green = is_active and state == "green"
                        is_yellow = is_active and state == "yellow"
                        
                        red_class = "active" if is_red else ""
                        yellow_class = "active" if is_yellow else ""
                        green_class = "active" if is_green else ""
                        
                        timer_val = display_sec if is_active else "--"
                        timer_color = "#10b981" if is_green else ("#fbbf24" if is_yellow else "#ef4444")
                        if not is_active: timer_color = "#334155"

                        r1[i].markdown(f"""
                        <div class="lane-box">
                            <div class="lane-title">LANE {i+1}</div>
                            <div class="traffic-pole">
                                <div class="light-bulb red {red_class}"></div>
                                <div class="light-bulb yellow {yellow_class}"></div>
                                <div class="light-bulb green {green_class}"></div>
                            </div>
                            <div class="timer-display" style="color: {timer_color}; border-color: {timer_color};">{timer_val}</div>
                        </div>
                        """, unsafe_allow_html=True)
                time.sleep(1)
                
        st.balloons()
        st.success("🎉 Full Traffic Cycle Successfully Completed.")
    else:
        st.error("⚠️ Please upload video files for all 4 lanes to initiate the system.")