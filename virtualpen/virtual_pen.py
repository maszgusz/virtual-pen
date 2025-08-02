import cv2
import mediapipe as mp
import numpy as np
import math
import os
import logging

# Menyembunyikan warning TensorFlow dan MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# --- Inisialisasi ---
# Inisialisasi modul hand dari MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Set resolusi kamera untuk mendapatkan aspek rasio yang baik
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Membuat canvas hitam untuk menggambar
h, w = 720, 1280
img_canvas = np.zeros((h, w, 3), np.uint8)

# Posisi titik sebelumnya untuk menggambar garis
prev_x, prev_y = 0, 0

# Status "bolpen" dan "penghapus"
is_writing = False
is_erasing = False

# Tombol Clear dan mode buttons
clear_button_rect = (50, 50, 200, 100)
draw_mode_rect = (220, 50, 370, 100)
erase_mode_rect = (390, 50, 540, 100)

# Mode saat ini: 'draw' atau 'erase'
current_mode = 'draw'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame agar terasa seperti cermin
    frame = cv2.flip(frame, 1)
    
    # Dapatkan dimensi asli frame
    original_height, original_width = frame.shape[:2]
    
    # Hitung aspek rasio
    target_aspect = w / h  # 1280/720 = 16:9
    original_aspect = original_width / original_height
    
    # Crop frame untuk mempertahankan aspek rasio tanpa distorsi
    if original_aspect > target_aspect:
        # Frame lebih lebar, crop dari samping
        new_width = int(original_height * target_aspect)
        start_x = (original_width - new_width) // 2
        frame = frame[:, start_x:start_x + new_width]
    else:
        # Frame lebih tinggi, crop dari atas/bawah
        new_height = int(original_width / target_aspect)
        start_y = (original_height - new_height) // 2
        frame = frame[start_y:start_y + new_height, :]
    
    # Resize ke ukuran target setelah crop
    frame = cv2.resize(frame, (w, h))

    # Ubah ke format RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses deteksi tangan
    results = hands.process(rgb_frame)

    # Menggabungkan frame kamera dengan canvas
    # Ini membuat tulisan tetap ada di layar
    img_combined = cv2.addWeighted(frame, 0.5, img_canvas, 0.5, 0)
    
    # Gambar tombol-tombol UI
    # Tombol Clear All
    cv2.rectangle(img_combined, (clear_button_rect[0], clear_button_rect[1]), 
                                (clear_button_rect[2], clear_button_rect[3]), 
                                (0, 0, 255), cv2.FILLED)
    cv2.putText(img_combined, "Clear", (clear_button_rect[0] + 15, clear_button_rect[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Tombol Draw Mode
    draw_color = (0, 255, 0) if current_mode == 'draw' else (100, 100, 100)
    cv2.rectangle(img_combined, (draw_mode_rect[0], draw_mode_rect[1]), 
                                (draw_mode_rect[2], draw_mode_rect[3]), 
                                draw_color, cv2.FILLED)
    cv2.putText(img_combined, "Draw", (draw_mode_rect[0] + 20, draw_mode_rect[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Tombol Erase Mode
    erase_color = (255, 100, 0) if current_mode == 'erase' else (100, 100, 100)
    cv2.rectangle(img_combined, (erase_mode_rect[0], erase_mode_rect[1]), 
                                (erase_mode_rect[2], erase_mode_rect[3]), 
                                erase_color, cv2.FILLED)
    cv2.putText(img_combined, "Erase", (erase_mode_rect[0] + 15, erase_mode_rect[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan untuk debugging
            mp_draw.draw_landmarks(img_combined, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Dapatkan koordinat landmark yang lebih akurat
            landmarks = []
            for lm in hand_landmarks.landmark:
                h_lm, w_lm, c_lm = frame.shape
                cx, cy = int(lm.x * w_lm), int(lm.y * h_lm)
                landmarks.append([cx, cy])
            
            # Index untuk landmark tangan
            thumb_tip = landmarks[4]        # Ujung ibu jari
            thumb_ip = landmarks[3]         # Sendi ibu jari
            index_tip = landmarks[8]        # Ujung telunjuk
            index_pip = landmarks[6]        # Sendi tengah telunjuk
            middle_tip = landmarks[12]      # Ujung jari tengah
            middle_pip = landmarks[10]      # Sendi jari tengah
            ring_tip = landmarks[16]        # Ujung jari manis
            pinky_tip = landmarks[20]       # Ujung kelingking
            
            # Koordinat untuk deteksi gesture
            thumb_x, thumb_y = thumb_tip[0], thumb_tip[1]
            index_x, index_y = index_tip[0], index_tip[1]
            
            # Hitung jarak antara ujung ibu jari dan telunjuk
            distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # Deteksi gesture untuk mode selection
            # Cek apakah jari lain tertekuk (untuk gesture lebih akurat)
            fingers_up = []
            
            # Ibu jari (berbeda karena orientasi horizontal)
            if thumb_tip[0] > thumb_ip[0]:  # Untuk tangan kanan
                fingers_up.append(1)
            else:
                fingers_up.append(0)
                
            # Jari lainnya (telunjuk, tengah, manis, kelingking)
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            
            for i in range(4):
                if landmarks[finger_tips[i]][1] < landmarks[finger_pips[i]][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Gambar indikator jari aktif
            cv2.circle(img_combined, (index_x, index_y), 15, (0, 255, 0), 3)
            cv2.circle(img_combined, (thumb_x, thumb_y), 15, (255, 0, 0), 3)
            
            # Tampilkan jarak untuk debugging
            cv2.putText(img_combined, f"Distance: {int(distance)}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # --- Deteksi Click/Touch ---
            is_touching = distance < 40
            
            if is_touching:
                cv2.circle(img_combined, (index_x, index_y), 15, (0, 0, 255), cv2.FILLED)
                cv2.line(img_combined, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
            
            # --- Logika Mode Selection ---
            # Cek tombol Clear
            if (clear_button_rect[0] < index_x < clear_button_rect[2] and 
                clear_button_rect[1] < index_y < clear_button_rect[3] and is_touching):
                img_canvas = np.zeros((h, w, 3), np.uint8)
                cv2.waitKey(200)  # Delay untuk mencegah multiple click
            
            # Cek tombol Draw Mode
            if (draw_mode_rect[0] < index_x < draw_mode_rect[2] and 
                draw_mode_rect[1] < index_y < draw_mode_rect[3] and is_touching):
                current_mode = 'draw'
                cv2.waitKey(200)
            
            # Cek tombol Erase Mode
            if (erase_mode_rect[0] < index_x < erase_mode_rect[2] and 
                erase_mode_rect[1] < index_y < erase_mode_rect[3] and is_touching):
                current_mode = 'erase'
                cv2.waitKey(200)
            
            # --- Logika Drawing/Erasing ---
            # Hanya aktif jika tidak menyentuh tombol UI
            in_ui_area = ((clear_button_rect[0] < index_x < clear_button_rect[2] and 
                          clear_button_rect[1] < index_y < clear_button_rect[3]) or
                         (draw_mode_rect[0] < index_x < draw_mode_rect[2] and 
                          draw_mode_rect[1] < index_y < draw_mode_rect[3]) or
                         (erase_mode_rect[0] < index_x < erase_mode_rect[2] and 
                          erase_mode_rect[1] < index_y < erase_mode_rect[3]))
            
            if is_touching and not in_ui_area:
                if current_mode == 'draw':
                    if not is_writing:
                        prev_x, prev_y = index_x, index_y
                        is_writing = True
                    else:
                        # Gambar garis
                        cv2.line(img_canvas, (prev_x, prev_y), (index_x, index_y), (0, 255, 0), 8)
                        prev_x, prev_y = index_x, index_y
                        
                elif current_mode == 'erase':
                    if not is_erasing:
                        prev_x, prev_y = index_x, index_y
                        is_erasing = True
                    else:
                        # Hapus dengan menggambar garis hitam
                        cv2.line(img_canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), 30)
                        prev_x, prev_y = index_x, index_y
                        
                    # Gambar indikator eraser
                    cv2.circle(img_combined, (index_x, index_y), 20, (255, 100, 0), 3)
            else:
                # Reset status ketika tidak menyentuh
                is_writing = False
                is_erasing = False
    
    # Tampilkan mode saat ini
    mode_text = f"Mode: {current_mode.upper()}"
    cv2.putText(img_combined, mode_text, (10, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Tampilkan frame hasil
    cv2.imshow("Virtual Pen", img_combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()