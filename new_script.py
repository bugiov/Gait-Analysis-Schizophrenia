import os
import json
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import argparse
import math
from tqdm import tqdm  # Libreria per la progress bar

# Struttura dati per memorizzare la cronologia completa dei keypoint
keypoint_history = {}  # Dictionary per memorizzare gli ultimi valori di depth per ogni ID keypoint
previous_keypoints = {}  # Dictionary per memorizzare le coordinate del frame precedente per ogni ID keypoint
current_window = {}  # Dictionary per memorizzare i movimenti della finestra corrente per ogni ID keypoint

def rotate_point(x, y, width, height):
    """
    Ruota un punto di 90 gradi in senso orario
    x, y: coordinate originali
    width, height: dimensioni originali dell'immagine
    """
    new_x = height - y
    new_y = x
    return new_x, new_y

def get_depth_around_point(depth_frame, x, y, keypoint_id, frame_num, size=5):
    """
    Ottiene un valore di profondità medio valido nell'intorno di un punto
    utilizzando una finestra quadrata di dimensione specificata.
    Se non trova valori validi, utilizza la media degli ultimi 5 frame per quel keypoint.
    
    Args:
        depth_frame: Frame di profondità RealSense
        x, y: Coordinate del punto (già ruotate di 90 gradi in senso orario)
        keypoint_id: ID del keypoint
        frame_num: Numero del frame
        size: Dimensione della finestra intorno al punto
    """
    global keypoint_history
    
    half_size = size // 2
    depth_values = []
    
    # Ottieni le dimensioni del depth frame
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    
    # Invertiamo la rotazione di 90 gradi in senso orario per ottenere le coordinate originali
    # Se la rotazione originale è:
    # new_x = height - y
    # new_y = x
    # Allora l'inversa è:
    # orig_x = y
    # orig_y = height - x
    x_orig = y
    y_orig = height - x
    
    # Controlla i punti nell'intorno
    for dy in range(-half_size, half_size + 1):
        for dx in range(-half_size, half_size + 1):
            nx, ny = x_orig + dx, y_orig + dy
            # Verifica che le coordinate siano all'interno dell'immagine
            if 0 <= nx < width and 0 <= ny < height:
                try:
                    depth = depth_frame.get_distance(nx, ny)
                    if depth > 0:  # Solo valori validi
                        depth_values.append(depth)
                except:
                    pass
    
    # Calcola la media dei valori di profondità validi
    if depth_values:
        depth_value = sum(depth_values) / len(depth_values)
        
        # Aggiorna la storia per questo keypoint
        if keypoint_id not in keypoint_history:
            keypoint_history[keypoint_id] = []
        
        # Mantieni solo gli ultimi 5 valori
        keypoint_history[keypoint_id].append((frame_num, depth_value))
        if len(keypoint_history[keypoint_id]) > 5:
            keypoint_history[keypoint_id].pop(0)
            
        return depth_value
    
    # Se non sono stati trovati valori validi, usa la storia recente
    if keypoint_id in keypoint_history and keypoint_history[keypoint_id]:
        # Calcola la media dei valori recenti
        recent_depths = [depth for _, depth in keypoint_history[keypoint_id]]
        return sum(recent_depths) / len(recent_depths)
    
    # Se non c'è storia recente, restituisci un valore di default
    return 0.0  # Valore di default

def calculate_depth_error(depth_value, camera_info):
    """
    Calcola l'errore RMS di profondità per un dato valore di profondità
    utilizzando la formula per telecamere stereo:
    
    Depth RMS error(mm) = Distance(mm)² × Subpixel / (focal length(pixels) × Baseline(mm))
    
    dove focal length(pixels) = (1/2) × res(pixels) / tan(HFOV/2)
    
    Args:
        depth_value: Valore di profondità in metri
        camera_info: Dizionario con informazioni sulla camera (baseline, HFOV, risoluzione)
    
    Returns:
        Errore RMS in metri
    """
    # Converte profondità da metri a millimetri
    distance_mm = depth_value * 1000.0
    
    # Valori tipici per le telecamere RealSense
    # Se non sono disponibili nei dati del bag, usiamo valori di default
    baseline_mm = camera_info.get('baseline', 50.0)  # Baseline tipica in mm
    hfov_degrees = camera_info.get('hfov', 86.0)     # HFOV tipico in gradi
    width_pixels = camera_info.get('width', 1280)    # Risoluzione orizzontale tipica in pixel
    subpixel_error = camera_info.get('subpixel', 0.08)  # Errore tipico di disparità subpixel
    
    # Converte HFOV da gradi a radianti
    hfov_radians = math.radians(hfov_degrees)
    
    # Calcola la lunghezza focale in pixel usando la formula fornita
    focal_length_pixels = (width_pixels / 2.0) / math.tan(hfov_radians / 2.0)
    
    # Calcola l'errore RMS di profondità in mm
    if focal_length_pixels * baseline_mm == 0:  # Evita divisione per zero
        depth_error_mm = 0.0
    else:
        depth_error_mm = (distance_mm ** 2 * subpixel_error) / (focal_length_pixels * baseline_mm)
    
    # Converte da mm a metri per coerenza con il resto dei dati
    depth_error_m = depth_error_mm / 1000.0
    
    return depth_error_m

def calculate_3d_distance(point1, point2):
    """
    Calcola la distanza euclidea 3D tra due punti
    """
    return ((point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2 + 
            (point1['z'] - point2['z'])**2) ** 0.5

def calculate_jitter_score(current_keypoints, frame_num, frame_window=2):
    """
    Calcola uno score di jitter per il frame corrente confrontandolo con i frame precedenti.
    Restituisce uno score da 0 a 1, dove 1 indica un movimento uniforme e coerente.
    Penalizza fortemente keypoint con depth=0 o bassa visibilità.
    
    Args:
        current_keypoints: Lista di keypoint del frame corrente
        frame_num: Numero del frame corrente
        frame_window: Numero di frame consecutivi da considerare
    
    Returns:
        Tuple di (score_complessivo, dictionary di score per keypoint)
    """
    global current_window, previous_keypoints
    
    # Se non ci sono keypoint precedenti, non possiamo calcolare il jitter
    if not previous_keypoints or not current_keypoints:
        # Inizializza lo storage per questo frame
        if current_keypoints:
            previous_keypoints = {kp['id']: kp for kp in current_keypoints}
            
        # Restituisci uno score neutro
        return 0.5, {kp['id']: 0.5 for kp in current_keypoints} if current_keypoints else {}
    
    jitter_scores = {}
    all_scores = []
    depth_penalty_applied = False  # Flag per indicare se è stata applicata una penalità di profondità
    visibility_penalty = 0.0  # Penalità per visibilità bassa
    
    # Esaminare ogni keypoint corrente
    for keypoint in current_keypoints:
        kp_id = keypoint['id']
        kp_depth = keypoint['z']
        kp_visibility = keypoint.get('visibility', 1.0)  # Default a 1.0 se non presente
        
        # Applicare penalità per depth=0
        depth_penalty = 0.0
        if kp_depth == 0.0:
            depth_penalty = 0.5  # Forte penalità per profondità zero
            depth_penalty_applied = True
        
        # Applicare penalità in base alla visibilità
        visibility_penalty_factor = max(0, 1.0 - kp_visibility)  # 0 quando visibility=1, 1 quando visibility=0
        visibility_penalty += visibility_penalty_factor * 0.3  # Penalità massima di 0.3 per visibility=0
        
        # Se abbiamo il keypoint nel frame precedente, calcoliamo il jitter
        if kp_id in previous_keypoints:
            # Calcola la distanza 3D tra il keypoint corrente e quello precedente
            distance = calculate_3d_distance(keypoint, previous_keypoints[kp_id])
            
            # Inizializza la finestra corrente se necessario
            if kp_id not in current_window:
                current_window[kp_id] = []
            
            # Aggiungi la distanza alla finestra corrente
            current_window[kp_id].append(distance)
            
            # Mantieni solo le ultime 'frame_window' distanze
            if len(current_window[kp_id]) > frame_window:
                current_window[kp_id].pop(0)
            
            # Calcola la variazione del movimento rispetto alla media delle distanze precedenti
            if len(current_window[kp_id]) > 1:
                avg_distance = sum(current_window[kp_id][:-1]) / (len(current_window[kp_id]) - 1)
                
                # Evita divisione per zero
                if avg_distance == 0:
                    if distance == 0:
                        # Non c'è movimento
                        jitter_score = 1.0
                    else:
                        # C'è movimento improvviso da una posizione statica
                        jitter_score = 0.5
                else:
                    # Calcola quanto è uniforme il movimento
                    movement_ratio = distance / avg_distance
                    
                    # Score alto se il movimento è simile a quello precedente
                    # Score basso se c'è un cambio improvviso di velocità o direzione
                    if movement_ratio > 3.0:  # Movimento molto più grande (jitter alto)
                        jitter_score = 0.0
                    elif movement_ratio > 2.0:  # Movimento significativamente più grande
                        jitter_score = 0.3
                    elif movement_ratio < 0.33:  # Movimento molto più piccolo (possibile freeze)
                        jitter_score = 0.4
                    elif 0.75 <= movement_ratio <= 1.25:  # Movimento uniforme
                        jitter_score = 1.0
                    else:  # Movimento leggermente diverso
                        jitter_score = 0.8
            else:
                # Non abbiamo abbastanza storia per questo keypoint
                jitter_score = 0.5
        else:
            # Keypoint appena apparso
            jitter_score = 0.5
        
        # Applicare penalità di depth
        jitter_score = max(0.0, jitter_score - depth_penalty)
        
        # Salva lo score per questo keypoint
        jitter_scores[kp_id] = jitter_score
        all_scores.append(jitter_score)
    
    # Aggiorna i keypoint precedenti per il prossimo frame
    previous_keypoints = {kp['id']: kp for kp in current_keypoints}
    
    # Calcola lo score complessivo come media degli score dei singoli keypoint
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.5
    
    # Applica penalità globale per depth=0 e visibilità
    if depth_penalty_applied:
        # Riduci lo score globale di un fattore proporzionale al numero di keypoint con depth=0
        overall_score = overall_score * 0.7  # Penalità del 30%
    
    # Applica penalità per visibilità media bassa (normalizzato per numero di keypoint)
    if len(current_keypoints) > 0:
        visibility_penalty = visibility_penalty / len(current_keypoints)
        overall_score = max(0.0, overall_score - visibility_penalty)
    
    return overall_score, jitter_scores

def get_significant_keypoints(results, color_image):
    """
    Seleziona i keypoints significativi e applica la rotazione.
    Include tutti i keypoints principali tranne le dita delle mani e piccoli dettagli del viso.
    """
    if not results.pose_landmarks:
        return [], []

    # Definizione dei keypoints significativi
    # Include tutti i keypoints principali tranne le dita delle mani
    significant_indices = [
        0,   # Nose
        1,   # Left Eye (inner)
        2,   # Left Eye
        3,   # Left Eye (outer)
        4,   # Right Eye (inner)
        5,   # Right Eye
        6,   # Right Eye (outer)
        7,   # Left Ear
        8,   # Right Ear
        9,   # Mouth (left)
        10,  # Mouth (right)
        11,  # Left shoulder
        12,  # Right shoulder
        13,  # Left elbow
        14,  # Right elbow
        15,  # Left wrist
        16,  # Right wrist
        17,  # Left pinky knuckle
        18,  # Right pinky knuckle
        19,  # Left index knuckle
        20,  # Right index knuckle
        21,  # Left thumb
        22,  # Right thumb
        23,  # Left hip
        24,  # Right hip
        25,  # Left knee
        26,  # Right knee
        27,  # Left ankle
        28,  # Right ankle
        29,  # Left heel
        30,  # Right heel
        31,  # Left foot index
        32   # Right foot index
    ]

    height, width = color_image.shape[:2]
    rotated_landmarks = []

    for i in significant_indices:
        landmark = results.pose_landmarks.landmark[i]
        
        # Calcolo coordinate pixel originali
        x_pixel = int(landmark.x * width)
        y_pixel = int(landmark.y * height)
        
        # Ruotare le coordinate
        rotated_x_pixel, rotated_y_pixel = rotate_point(x_pixel, y_pixel, width, height)
        
        # Manteniamo le coordinate in pixel senza normalizzarle
        rotated_landmark = type('RotatedLandmark', (), {
            'x': rotated_x_pixel,
            'y': rotated_y_pixel,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
        
        rotated_landmarks.append(rotated_landmark)

    # Ritorna sia i landmark che gli indici
    return rotated_landmarks, significant_indices

def draw_skeleton(image, landmarks):
    """
    Disegna lo scheletro sui landmark
    """
    # Definizione dei legami tra i keypoints
    connections = [
        (0, 1),   # Nose to Left Eye inner
        (1, 2),   # Left Eye inner to left eye
        (2, 3),   # Left Eye to left eye outer
        (0, 4),   # Nose to right eye inner
        (4, 5),   # Right eye inner to right eye
        (5, 6),   # Right eye to right eye outer
        (3, 7),   # Left eye outer to left ear
        (6, 8),   # Right eye outer to right ear
        (0, 9),   # Nose to mouth left
        (9, 10),  # Mouth left to mouth right
        (0, 10),  # Nose to mouth right
        (11, 13), # Left shoulder to left elbow
        (13, 15), # Left elbow to left wrist
        (12, 14), # Right shoulder to right elbow
        (14, 16), # Right elbow to right wrist
        (11, 12), # Left shoulder to right shoulder
        (11, 23), # Left shoulder to left hip
        (12, 24), # Right shoulder to right hip
        (23, 24), # Left hip to right hip
        (23, 25), # Left hip to left knee
        (25, 27), # Left knee to left ankle
        (24, 26), # Right hip to right knee
        (26, 28), # Right knee to right ankle
        (27, 29), # Left ankle to left heel
        (28, 30), # Right ankle to right heel
        (27, 31), # Left ankle to left foot index
        (28, 32), # Right ankle to right foot index
        (29, 31), # Left heel to left foot index
        (30, 32), # Right heel to right foot index
        
        # Connessioni per le mani
        (15, 17), # Left wrist to left pinky knuckle
        (15, 19), # Left wrist to left index knuckle
        (15, 21), # Left wrist to left thumb
        (17, 19), # Left pinky to left index
        (16, 18), # Right wrist to right pinky knuckle
        (16, 20), # Right wrist to right index knuckle
        (16, 22), # Right wrist to right thumb
        (18, 20)  # Right pinky to right index
    ]

    # Disegnare i punti
    for lm in landmarks:
        x = int(lm.x)  # Già in pixel, non serve moltiplicare
        y = int(lm.y)  # Già in pixel, non serve moltiplicare
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Disegnare i legami
    for connection in connections:
        if connection[0] < len(landmarks) and connection[1] < len(landmarks):
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            
            start_x = int(start.x)  # Già in pixel
            start_y = int(start.y)  # Già in pixel
            end_x = int(end.x)  # Già in pixel
            end_y = int(end.y)  # Già in pixel
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    return image

def get_camera_intrinsics(pipeline):
    """
    Estrae informazioni sulle intrinsiche della camera dal pipeline RealSense
    """
    try:
        # Ottengo il profilo della pipeline
        profile = pipeline.get_active_profile()
        
        # Ottengo il device
        device = profile.get_device()
        
        # Ottengo il sensore di profondità
        depth_sensor = device.first_depth_sensor()
        
        # Ottengo la baseline per D435/D455 (conversione da m a mm)
        # Questo valore può essere diverso in base al modello specifico
        try:
            baseline = depth_sensor.get_option(rs.option.stereo_baseline) * 1000.0  # in mm
        except:
            # Valori di default se non disponibili
            baseline = 50.0  # Baseline tipica per D435 in mm
        
        # Ottengo la risoluzione e l'angolo di visione
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = depth_stream.get_intrinsics()
        width = intrinsics.width
        height = intrinsics.height
        
        # Calcola HFOV dai parametri di intrinsics
        # tan(HFOV/2) = (width/2) / fx
        hfov = 2 * math.atan(width / (2 * intrinsics.fx))
        hfov_degrees = math.degrees(hfov)
        
        # Valore tipico di errore subpixel per RealSense
        subpixel = 0.08
        
        return {
            'baseline': baseline,
            'width': width,
            'height': height,
            'hfov': hfov_degrees,
            'subpixel': subpixel,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy
        }
    except Exception as e:
        print(f"⚠️ Errore nell'estrazione delle intrinsiche della camera: {e}")
        # Ritorna valori di default per D435
        return {
            'baseline': 50.0,  # mm
            'width': 1280,
            'height': 720,
            'hfov': 86.0,  # gradi
            'subpixel': 0.08
        }

def get_bag_info(bag_path):
    """
    Estrae informazioni sul file bag, incluso il frame rate
    """
    try:
        # Configurazione
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        
        # Create a pipeline
        pipeline = rs.pipeline()
        
        # Start pipeline with config
        profile = pipeline.start(config)
        device = profile.get_device()
        playback = device.as_playback()
        
        # Get streams
        streams = profile.get_streams()
        
        # Info di default
        color_fps = 60  # Impostiamo 60 come default visto che sappiamo che è questo
        depth_fps = 60
        color_resolution = (640, 480)
        depth_resolution = (640, 480)
        
        # Cerca gli stream di colore e profondità
        for stream in streams:
            if stream.stream_type() == rs.stream.color:
                color_fps = stream.fps()
                # Converti a video_stream_profile per ottenere risoluzione
                video_stream = stream.as_video_stream_profile()
                color_resolution = (video_stream.width(), video_stream.height())
            
            elif stream.stream_type() == rs.stream.depth:
                depth_fps = stream.fps()
                # Converti a video_stream_profile per ottenere risoluzione
                video_stream = stream.as_video_stream_profile()
                depth_resolution = (video_stream.width(), video_stream.height())
        
        # Get duration if possible
        duration = None
        try:
            duration = playback.get_duration().total_seconds()
        except Exception as e:
            print(f"Non è stato possibile ottenere la durata: {e}")
        
        # Stima numero totale di frame
        total_frames = int(duration * color_fps) if duration else None
        
        # Stop pipeline
        pipeline.stop()
        
        return {
            'color_fps': color_fps,
            'depth_fps': depth_fps,
            'color_resolution': color_resolution,
            'depth_resolution': depth_resolution,
            'duration': duration,
            'total_frames': total_frames
        }
    
    except Exception as e:
        print(f"Errore durante l'estrazione delle informazioni dal bag: {e}")
        return {
            'color_fps': 60,  # Default a 60 FPS come specificato
            'depth_fps': 60,
            'color_resolution': (640, 480),
            'depth_resolution': (640, 480),
            'duration': None,
            'total_frames': None
        }

def process_bag_file(bag_path, output_dir, max_frames=None):
    """
    Elabora un file bag RealSense e estrae frame e keypoints
    
    Args:
        bag_path: Percorso del file bag
        output_dir: Cartella di output
        max_frames: Numero massimo di frame da elaborare (None per tutti)
    """
    print(f"🚀 Inizio elaborazione file bag: {os.path.basename(bag_path)}")
    is_test_mode = max_frames is not None
    
    if is_test_mode:
        print(f"⚠️ MODALITÀ TEST: Elaborazione limitata a {max_frames} frame")
    
    # Ottieni informazioni sul file bag
    print("📊 Analisi informazioni del file bag...")
    try:
        bag_info = get_bag_info(bag_path)
        
        color_fps = bag_info['color_fps']
        depth_fps = bag_info['depth_fps']
        total_frames_estimated = bag_info['total_frames']
        
        print(f"   - FPS stream colore: {color_fps}")
        print(f"   - FPS stream profondità: {depth_fps}")
        print(f"   - Risoluzione colore: {bag_info['color_resolution'][0]}x{bag_info['color_resolution'][1]}")
        print(f"   - Risoluzione profondità: {bag_info['depth_resolution'][0]}x{bag_info['depth_resolution'][1]}")
        if bag_info['duration']:
            print(f"   - Durata stimata: {bag_info['duration']:.2f} secondi")
        if total_frames_estimated:
            print(f"   - Frame totali stimati: {total_frames_estimated}")
            
        if is_test_mode and total_frames_estimated:
            percent = (max_frames / total_frames_estimated) * 100
            expected_duration = max_frames / color_fps
            print(f"   - In modalità test verrà elaborato circa il {percent:.1f}% del bag")
            print(f"   - Durata video attesa: {expected_duration:.2f} secondi")
    except Exception as e:
        print(f"⚠️ Errore durante l'analisi del file bag: {e}")
        print("   Utilizzo dei valori predefiniti (FPS=60)")
        color_fps = 60
        depth_fps = 60
        total_frames_estimated = None
    
    start_time = time.time()

    # Creare le sottocartelle per i frame
    color_frames_dir = os.path.join(output_dir, 'color_frames')
    depth_frames_dir = os.path.join(output_dir, 'depth_frames')
    keypoints_dir = os.path.join(output_dir, 'keypoints')
    skeleton_frames_dir = os.path.join(output_dir, 'skeleton_frames')
    
    os.makedirs(color_frames_dir, exist_ok=True)
    os.makedirs(depth_frames_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    os.makedirs(skeleton_frames_dir, exist_ok=True)

    print("📂 Cartelle di output create:")
    print(f"   - Color Frames: {color_frames_dir}")
    print(f"   - Depth Frames: {depth_frames_dir}")
    print(f"   - Keypoints: {keypoints_dir}")
    print(f"   - Skeleton Frames: {skeleton_frames_dir}")

    # Configurazione MediaPipe Holistic con confidenza ridotta
    print("\n🔍 Configurazione MediaPipe Holistic (confidenza ridotta per maggiore sensibilità)...")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,  # Static mode per massima precisione
        model_complexity=2,      # Massima complessità per precisione superiore
        smooth_landmarks=False,  # No smoothing per precisione puntuale
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=True,  # Migliore precisione dei landmark facciali
        min_detection_confidence=0.4,  # Confidenza ridotta a 0.4 per rilevare più keypoints
        min_tracking_confidence=0.4    # Confidenza ridotta a 0.4 per tracking più sensibile
    )

    # Configurazione per leggere il file bag
    config = rs.config()
    config.enable_device_from_file(bag_path)

    # Pipeline RealSense
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    
    # Ottenere le informazioni della camera
    camera_info = get_camera_intrinsics(pipeline)
    print("\n📏 Informazioni camera:")
    print(f"   - Baseline: {camera_info['baseline']:.2f} mm")
    print(f"   - HFOV: {camera_info['hfov']:.2f} gradi")
    print(f"   - Risoluzione: {camera_info['width']}x{camera_info['height']} pixel")
    
    # Assicurarsi che il playback avvenga alla velocità corretta
    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)  # Disabilita real-time per processare tutto
    except Exception as e:
        print(f"⚠️ Avviso: Impossibile configurare la modalità di playback: {e}")

    # Preparazione per il video di output
    frames_for_video = []
    keypoints_for_video = []

    # Contatori per i frame
    frame_counter = 0
    processed_frames = 0  # Contatore separato per i frame elaborati con successo
    first_frame = None
    last_frame = None
    first_timestamp = None
    last_timestamp = None

    print("\n🔄 Elaborazione frame in corso...")
    try:
        # Creare una barra di progresso se conosciamo il limite
        if is_test_mode:
            pbar = tqdm(total=max_frames, desc="Elaborazione", unit="frame")
            
        while True:
            # Verificare se abbiamo raggiunto il limite di frame
            if is_test_mode and frame_counter >= max_frames:
                print(f"\n🛑 Raggiunto il limite di {max_frames} frame")
                break
                
            # Leggere i frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                break

            # Memorizzare il primo e l'ultimo frame e timestamp
            if frame_counter == 0:
                first_frame = frames
                first_timestamp = frames.get_timestamp() if hasattr(frames, 'get_timestamp') else None
            last_frame = frames
            last_timestamp = frames.get_timestamp() if hasattr(frames, 'get_timestamp') else None

            # Convertire i frame in numpy array
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convertire BGR a RGB
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Ruotare i frame di 90 gradi
            color_image_rotated = cv2.rotate(color_image_rgb, cv2.ROTATE_90_CLOCKWISE)
            depth_image_rotated = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)

            # Salvare frame colore (in RGB)
            color_filename = os.path.join(color_frames_dir, f'color_frame_{frame_counter:04d}.png')
            cv2.imwrite(color_filename, cv2.cvtColor(color_image_rotated, cv2.COLOR_RGB2BGR))  # OpenCV salva in BGR

            # Salvare frame profondità
            depth_filename = os.path.join(depth_frames_dir, f'depth_frame_{frame_counter:04d}.png')
            cv2.imwrite(depth_filename, depth_image_rotated)

            # Rilevamento keypoints con MediaPipe (MediaPipe usa RGB)
            rgb_image = color_image_rgb  # Già in RGB, non serve convertire
            results = holistic.process(rgb_image)

            # Preparare dizionario per keypoints 3D
            keypoints_3d = {}
            keypoints_3d['frame'] = frame_counter  # Aggiungiamo il numero del frame

            # Estrazione keypoints corpo significativi con rotazione
            significant_landmarks, significant_indices = get_significant_keypoints(results, color_image)
            
            # Flag per indicare se abbiamo trovato keypoints validi
            has_valid_keypoints = len(significant_landmarks) > 0
            
            # Assicurarsi che abbiamo sempre un frame nel video (anche se non ci sono keypoints)
            frame_with_skeleton = color_image_rotated.copy()
            
            if has_valid_keypoints:
                body_keypoints = []
                for i, landmark in enumerate(significant_landmarks):
                    # Ottieni l'ID MediaPipe originale
                    original_index = significant_indices[i]
                    
                    # Ottenere profondità del punto dalla depth image con metodo migliorato
                    depth_pixel = (int(landmark.x), int(landmark.y))  # Già in pixel
                    
                    # Usa la nuova funzione per ottenere un valore di profondità più robusto
                    depth_value = get_depth_around_point(depth_frame, depth_pixel[0], depth_pixel[1], 
                                                      original_index, frame_counter, size=5)
                    
                    # Calcola l'errore RMS della profondità
                    depth_error = calculate_depth_error(depth_value, camera_info)
                    
                    body_keypoints.append({
                        'x': int(landmark.x),
                        'y': int(landmark.y),
                        'z': depth_value,  # Utilizziamo depth come valore z
                        'depth_error': depth_error,  # Aggiungiamo l'errore RMS della profondità
                        'visibility': landmark.visibility,
                        'id': original_index  # Aggiungiamo l'ID MediaPipe originale
                    })
                
                # Calcola lo score di jitter/affidabilità per questo frame
                overall_score, keypoint_scores = calculate_jitter_score(body_keypoints, frame_counter)
                
                # Aggiungi gli score al JSON
                keypoints_3d['reliability_score'] = overall_score
                keypoints_3d['keypoint_scores'] = keypoint_scores
                keypoints_3d['body'] = body_keypoints

                # Preparare frame per video
                frame_with_skeleton = draw_skeleton(frame_with_skeleton, significant_landmarks)
                processed_frames += 1
            else:
                keypoints_3d['body'] = []  # Array vuoto per i keypoints
                keypoints_3d['reliability_score'] = 0.0  # Score basso quando non ci sono keypoint
                keypoints_3d['keypoint_scores'] = {}
                # Aggiungi un testo al frame per indicare che non sono stati trovati keypoints
                height, width = frame_with_skeleton.shape[:2]
                cv2.putText(frame_with_skeleton, "No keypoints detected", 
                           (int(width*0.1), int(height*0.5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Salvare il frame con skeleton (in RGB)
            skeleton_filename = os.path.join(skeleton_frames_dir, f'skeleton_frame_{frame_counter:04d}.png')
            cv2.imwrite(skeleton_filename, cv2.cvtColor(frame_with_skeleton, cv2.COLOR_RGB2BGR))  # OpenCV salva in BGR
            
            # Memorizza frame per video (manteniamo in RGB)
            frames_for_video.append(frame_with_skeleton)  # Manteniamo in RGB
            
            if has_valid_keypoints:
                keypoints_for_video.append(body_keypoints)
            else:
                # Aggiungi un keypoint nullo per mantenere il sincronismo con i frame
                keypoints_for_video.append([])

            # Salvare keypoints in JSON
            keypoints_filename = os.path.join(keypoints_dir, f'keypoints_{frame_counter:04d}.json')
            with open(keypoints_filename, 'w') as f:
                json.dump(keypoints_3d, f, indent=4)

            frame_counter += 1
            
            # Aggiornare la barra di progresso se attiva
            if is_test_mode:
                pbar.update(1)
                
            # Mostrare un feedback ogni 50 frame se non c'è barra di progresso
            if not is_test_mode and frame_counter % 50 == 0:
                print(f"   Elaborati {frame_counter} frame...")

    except RuntimeError as e:
        print(f"\n❗ Fine della riproduzione del bag: {e}")
    
    finally:
        pipeline.stop()
        holistic.close()
        # Chiudere la barra di progresso se attiva
        if is_test_mode and 'pbar' in locals():
            pbar.close()

    # Generare video con scheletro
    if frames_for_video:
        print("\n🎥 Generazione video dello scheletro...")
        video_filename = os.path.join(output_dir, 'skeleton_animation.mp4')
        height, width = frames_for_video[0].shape[:2]
        
        # VERIFICA: Stampa numero di frame per il video
        print(f"   Numero di frame per il video: {len(frames_for_video)}")
        
        # Usa il FPS originale del bag per il video di output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Specifichiamo esplicitamente isColor=True per un video RGB
        out = cv2.VideoWriter(video_filename, fourcc, color_fps, (width, height), isColor=True)
        
        frame_count_in_video = 0
        for frame in tqdm(frames_for_video, desc="Scrittura video", unit="frame"):
            # Manteniamo il frame in RGB senza convertirlo
            out.write(frame)
            frame_count_in_video += 1
        
        out.release()
        
        # Verifica dei frame effettivamente scritti nel video
        print(f"   Frame scritti nel video: {frame_count_in_video}")
        
        # Calcolo durata prevista e verifica
        expected_duration = frame_count_in_video / color_fps
        print(f"   Durata prevista del video: {expected_duration:.2f} secondi")
        
        # Verifica durata effettiva del video generato
        try:
            video_cap = cv2.VideoCapture(video_filename)
            if video_cap.isOpened():
                frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                actual_duration = frame_count / video_fps
                print(f"   Durata effettiva del video (da proprietà): {actual_duration:.2f} secondi")
                print(f"   FPS effettivi del video (da proprietà): {video_fps}")
                print(f"   Frame nel video (da proprietà): {frame_count}")
                if abs(expected_duration - actual_duration) > 0.5:
                    print(f"   ⚠️ La durata del video non corrisponde a quella attesa!")
            video_cap.release()
        except Exception as e:
            print(f"   ⚠️ Impossibile verificare la durata effettiva: {e}")
        
        # Salvare keypoints del video in un file JSON globale
        video_keypoints_filename = os.path.join(output_dir, 'video_keypoints.json')
        with open(video_keypoints_filename, 'w') as f:
            json.dump(keypoints_for_video, f, indent=4)

    # Calcolo tempo totale
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / frame_counter if frame_counter > 0 else 0
    
    # Calcola durata effettiva dalla differenza di timestamp
    duration_from_timestamps = None
    if first_timestamp is not None and last_timestamp is not None:
        duration_from_timestamps = (last_timestamp - first_timestamp) / 1000.0  # ms to s

    print(f"\n✅ Elaborazione completata!")
    print(f"   Frame totali elaborati: {frame_counter}")
    print(f"   Frame con keypoints rilevati: {processed_frames}")
    
    if first_timestamp is not None and last_timestamp is not None:
        print(f"   Primo timestamp: {first_timestamp:.2f} ms")
        print(f"   Ultimo timestamp: {last_timestamp:.2f} ms")
        print(f"   Durata porzione elaborata: {duration_from_timestamps:.2f} secondi")
    
    # Controllo integrità frame solo se NON siamo in modalità test
    if not is_test_mode:
        try:
            if 'bag_info' in locals() and bag_info['duration'] and total_frames_estimated:
                print(f"   Frame attesi (basato su durata bag): circa {total_frames_estimated}")
                if frame_counter < total_frames_estimated * 0.95:
                    print(f"   ⚠️ Attenzione: Elaborati meno frame del previsto ({frame_counter}/{total_frames_estimated})")
            elif duration_from_timestamps:
                expected_frames = int(duration_from_timestamps * color_fps)
                print(f"   Frame attesi (basato su timestamp): circa {expected_frames}")
                if frame_counter < expected_frames * 0.95:
                    print(f"   ⚠️ Attenzione: Elaborati meno frame del previsto ({frame_counter}/{expected_frames})")
        except Exception as e:
            print(f"   ⚠️ Impossibile verificare integrità frame: {e}")
    else:
        # In modalità test, mostrare informazioni sul test e sul bag completo
        print(f"   ℹ️ Modalità test attiva: Elaborati {frame_counter} su circa {total_frames_estimated} frame totali stimati")
        if total_frames_estimated:
            percent = (frame_counter / total_frames_estimated) * 100
            print(f"   ℹ️ Percentuale del bag elaborata: {percent:.1f}%")
    
    print(f"   FPS originale: {bag_info['color_fps'] if 'bag_info' in locals() else 'Sconosciuto'}")
    print(f"   FPS usato per il video: {color_fps}")
    print(f"   Tempo totale di elaborazione: {total_time:.2f} secondi")
    print(f"   Tempo medio per frame: {avg_time_per_frame:.4f} secondi")
    if is_test_mode:
        # Stima tempo elaborazione completa
        if total_frames_estimated:
            estimated_total_time = (total_time / frame_counter) * total_frames_estimated
            print(f"   ℹ️ Tempo stimato per elaborazione completa: {estimated_total_time:.2f} secondi ({estimated_total_time/60:.1f} minuti)")
    
    print(f"   Video scheletro salvato in: {video_filename}")
    print(f"   Keypoints salvati in: {video_keypoints_filename}")

def main():
    # Configurazione parser per argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Elabora file bag RealSense per estrarre frame e keypoints')
    parser.add_argument('bag_path', help='Percorso del file bag da elaborare')
    parser.add_argument('output_dir', help='Cartella di output per i frame e keypoints')
    parser.add_argument('--max-frames', type=int, help='Numero massimo di frame da elaborare (predefinito: tutti)')
    parser.add_argument('--test', action='store_true', help='Modalità test (limita a 500 frame)')
    
    args = parser.parse_args()
    
    # Correzione: Se --test è attivo, imposta max_frames a 500, indipendentemente da cosa è stato specificato
    if args.test:
        args.max_frames = 500
        print("Modalità test attivata: elaborazione limitata a 500 frame")
    
    # Chiamare la funzione principale
    process_bag_file(args.bag_path, args.output_dir, args.max_frames)

# Esempio di utilizzo
if __name__ == "__main__":
    main()