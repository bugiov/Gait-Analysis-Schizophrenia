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
from scipy.signal import savgol_filter  # Per il filtro Savitzky-Golay

# Struttura dati per memorizzare la cronologia completa dei keypoint
keypoint_history = {}  # Dictionary per memorizzare gli ultimi valori di depth per ogni ID keypoint
previous_keypoints = {}  # Dictionary per memorizzare le coordinate del frame precedente per ogni ID keypoint
current_window = {}  # Dictionary per memorizzare i movimenti della finestra corrente per ogni ID keypoint

# Buffer per filtraggio Savitzky-Golay
keypoint_buffers = {}  # Dictionary per memorizzare la storia dei keypoint per il filtraggio

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
        Errore RMS in millimetri
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
    
    # Ritorniamo direttamente in millimetri
    return depth_error_mm

def calculate_3d_distance(point1, point2):
    """
    Calcola la distanza euclidea 3D tra due punti
    """
    return ((point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2 + 
            (point1['z'] - point2['z'])**2) ** 0.5

def calculate_jitter_score(current_keypoints, frame_num, frame_window=15):
    """
    Calcola uno score di jitter per il frame corrente confrontandolo con i frame precedenti.
    Restituisce uno score da 0 a 1, dove 1 indica un movimento uniforme e coerente.
    Penalizza fortemente keypoint con depth=0 o bassa visibilità.
    
    Args:
        current_keypoints: Lista di keypoint del frame corrente
        frame_num: Numero del frame corrente
        frame_window: Numero di frame consecutivi da considerare (default: 15)
    
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

def apply_adaptive_savgol_filter(keypoints_sequence, filter_params):
    """
    Applica il filtro di Savitzky-Golay con parametri adattivi
    in base alla consistenza dei keypoint
    
    Args:
        keypoints_sequence: Sequenza di keypoint per ciascun ID
        filter_params: Parametri base del filtro (window_length, poly_order, threshold, coords)
    
    Returns:
        Keypoint filtrati
    """
    if not keypoints_sequence:
        return []
    
    # Estrai parametri base
    base_window_length = filter_params.get('window_length', 15)
    base_poly_order = filter_params.get('poly_order', 2)
    base_threshold = filter_params.get('threshold', 10)
    coords_to_filter = filter_params.get('coords', ['x', 'y', 'z'])
    
    # Crea una copia dei keypoint originali
    filtered_keypoints = keypoints_sequence.copy()
    
    # Per ogni ID di keypoint
    all_ids = set()
    for frame_kps in keypoints_sequence:
        for kp in frame_kps:
            if 'id' in kp:
                all_ids.add(kp['id'])
    
    print(f"   ℹ️ Analisi di {len(all_ids)} keypoints unici")
    
    # Per ogni ID di keypoint, estrai le coordinate e applica il filtro
    for kp_id in all_ids:
        # Estrai le coordinate per questo ID
        coords_data = {coord: [] for coord in coords_to_filter}
        visibility_data = []
        frame_indices = []
        
        # Raccogli dati per questo keypoint
        for i, frame_kps in enumerate(keypoints_sequence):
            for kp in frame_kps:
                if kp.get('id') == kp_id:
                    for coord in coords_to_filter:
                        coords_data[coord].append(kp.get(coord, 0))
                    visibility_data.append(kp.get('visibility', 0))
                    frame_indices.append(i)
                    break
        
        # Se abbiamo abbastanza punti, applica il filtro
        if len(frame_indices) >= base_window_length:
            # Analizzare la qualità dei keypoint per questo ID
            mean_visibility = sum(visibility_data) / len(visibility_data) if visibility_data else 0
            
            # Calcolare variabilità delle coordinate
            variance = {}
            for coord in coords_to_filter:
                if coords_data[coord]:
                    coord_array = np.array(coords_data[coord])
                    if len(coord_array) > 1:
                        diffs = np.abs(np.diff(coord_array))
                        variance[coord] = np.mean(diffs)
                    else:
                        variance[coord] = 0
                else:
                    variance[coord] = 0
            
            # Determinare la complessità di filtraggio necessaria
            avg_variance = sum(variance.values()) / len(variance) if variance else 0
            
            # Adattare i parametri del filtro in base alla qualità
            adaptive_window = base_window_length
            adaptive_poly = base_poly_order
            adaptive_threshold = base_threshold
            
            # Alta variabilità + bassa visibilità = sequenza problematica
            if avg_variance > 20 and mean_visibility < 0.7:
                # Sequenza molto problematica: finestra grande, polinomio basso
                adaptive_window = min(21, len(frame_indices) - 1)  # Finestra massima 21 o disponibile -1
                adaptive_poly = max(1, base_poly_order - 1)  # Polinomio più rigido
                adaptive_threshold = base_threshold * 0.5  # Più sensibile
                print(f"   ⚠️ Keypoint {kp_id}: Sequenza molto problematica, parametri adattati: win={adaptive_window}, poly={adaptive_poly}")
            elif avg_variance > 10 or mean_visibility < 0.8:
                # Sequenza problematica: aumenta finestra
                adaptive_window = min(17, len(frame_indices) - 1)
                adaptive_threshold = base_threshold * 0.7
                print(f"   ⚙️ Keypoint {kp_id}: Sequenza problematica, parametri adattati: win={adaptive_window}")
            
            # Verifica che la finestra sia valida (deve essere dispari)
            if adaptive_window % 2 == 0:
                adaptive_window += 1
            
            # Verifica che poly_order < window_length
            adaptive_poly = min(adaptive_poly, adaptive_window - 1)
                
            # Applica il filtro per ogni coordinata
            for coord in coords_to_filter:
                coord_array = np.array(coords_data[coord])
                
                # Calcola le differenze per rilevare jitter
                if len(coord_array) > 1:
                    diffs = np.abs(np.diff(coord_array))
                    max_diff = np.max(diffs) if len(diffs) > 0 else 0
                    
                    # Verifica se è necessario filtrare questa coordinata
                    needs_filtering = np.any(diffs > adaptive_threshold) or max_diff > adaptive_threshold * 2
                    
                    if needs_filtering and len(coord_array) >= adaptive_window:
                        try:
                            # Applica il filtro Savitzky-Golay con parametri adattivi
                            filtered_values = savgol_filter(coord_array, adaptive_window, adaptive_poly)
                            
                            # Aggiorna i valori nei keypoint filtrati
                            for i, frame_idx in enumerate(frame_indices):
                                for kp in filtered_keypoints[frame_idx]:
                                    if kp.get('id') == kp_id:
                                        kp[coord] = filtered_values[i]
                                        break
                        except Exception as e:
                            print(f"   ❌ Errore nel filtraggio per keypoint {kp_id}, coordinata {coord}: {e}")
    
    return filtered_keypoints

# Nuova funzione per creare un video comparativo
def create_comparison_video(original_frames, filtered_frames, output_path, fps):
    """
    Crea un video che mette a confronto i frame originali e quelli filtrati
    
    Args:
        original_frames: Lista di frame originali
        filtered_frames: Lista di frame filtrati
        output_path: Percorso del file video di output
        fps: Frame rate del video
    """
    if not original_frames or not filtered_frames:
        print("❌ Non ci sono frame sufficienti per creare il video comparativo")
        return
    
    # Assicurarsi che entrambe le liste abbiano la stessa lunghezza
    min_frames = min(len(original_frames), len(filtered_frames))
    
    # Ottenere dimensioni del frame
    height, width, _ = original_frames[0].shape
    
    # Creare spazio per il video comparativo (affiancato)
    comparison_width = width * 2
    comparison_height = height
    
    # Configurare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comparison_width, comparison_height), isColor=True)
    
    # Creare i frame comparativi
    for i in tqdm(range(min_frames), desc="Creazione video comparativo", unit="frame"):
        # Creare frame comparativo
        comparison_frame = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
        
        # Aggiungere frame originale a sinistra
        comparison_frame[0:height, 0:width] = original_frames[i]
        
        # Aggiungere frame filtrato a destra
        comparison_frame[0:height, width:width*2] = filtered_frames[i]
        
        # Aggiungere etichette
        cv2.putText(comparison_frame, "Originale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison_frame, "Filtrato", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Scrivere frame nel video
        out.write(comparison_frame)
    
    # Rilasciare video writer
    out.release()
    
    print(f"   ✅ Video comparativo creato con successo: {output_path}")
    print(f"   ℹ️ Durata: {min_frames / fps:.2f} secondi ({min_frames} frames)")

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

def draw_skeleton(image, landmarks, point_color=(0, 255, 0), line_color=(255, 0, 0)):
    """
    Disegna lo scheletro sui landmark con colori personalizzabili
    
    Args:
        image: Immagine su cui disegnare
        landmarks: Keypoints da disegnare
        point_color: Colore per i punti (default: verde)
        line_color: Colore per le linee (default: blu)
    
    Returns:
        Immagine con lo scheletro disegnato
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
        cv2.circle(image, (x, y), 5, point_color, -1)

    # Disegnare i legami
    for connection in connections:
        if connection[0] < len(landmarks) and connection[1] < len(landmarks):
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            
            start_x = int(start.x)  # Già in pixel
            start_y = int(start.y)  # Già in pixel
            end_x = int(end.x)  # Già in pixel
            end_y = int(end.y)  # Già in pixel
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), line_color, 2)

    return image

# Aggiungere la funzione per generare frames con scheletri sovrapposti
def create_overlaid_skeletons(original_landmarks, filtered_landmarks, color_image):
    """
    Crea un'immagine con entrambi gli scheletri sovrapposti in colori diversi
    
    Args:
        original_landmarks: Keypoints originali
        filtered_landmarks: Keypoints filtrati
        color_image: Immagine a colori su cui disegnare
    
    Returns:
        Immagine con entrambi gli scheletri
    """
    # Crea una copia dell'immagine
    overlaid_image = color_image.copy()
    
    # Disegna lo scheletro originale (verde/blu)
    if original_landmarks:
        overlaid_image = draw_skeleton(overlaid_image, original_landmarks, 
                                       point_color=(0, 255, 0), line_color=(0, 0, 255))
    
    # Disegna lo scheletro filtrato (giallo/rosso)
    if filtered_landmarks:
        overlaid_image = draw_skeleton(overlaid_image, filtered_landmarks, 
                                       point_color=(255, 255, 0), line_color=(255, 0, 0))
    
    # Aggiungi una legenda per distinguere i due scheletri
    cv2.putText(overlaid_image, "Originale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlaid_image, "Filtrato", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return overlaid_image

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

def process_bag_file(bag_path, output_dir, max_frames=None, use_filter=False, filter_window=15, 
                     filter_poly=2, filter_threshold=10, create_overlay=False):
    """
    Elabora un file bag RealSense e estrae frame e keypoints
    
    Args:
        bag_path: Percorso del file bag
        output_dir: Cartella di output
        max_frames: Numero massimo di frame da elaborare (None per tutti)
        use_filter: Attiva il filtro Savitzky-Golay
        filter_window: Dimensione della finestra per il filtro
        filter_poly: Ordine del polinomio per il filtro
        filter_threshold: Soglia per applicare il filtro
        create_overlay: Genera frames con scheletri sovrapposti
    """
    print(f"🚀 Inizio elaborazione file bag: {os.path.basename(bag_path)}")
    is_test_mode = max_frames is not None
    
    if is_test_mode:
        print(f"⚠️ MODALITÀ TEST: Elaborazione limitata a {max_frames} frame")
    
    if use_filter:
        print(f"   Video scheletro filtrato salvato in: {os.path.join(output_dir, 'skeleton_animation_filtered.mp4')}")
        print(f"   Video comparativo salvato in: {os.path.join(output_dir, 'skeleton_comparison.mp4')}")
        print(f"   Keypoints filtrati salvati in: {os.path.join(output_dir, 'keypoints_filtered')}")

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
    keypoints_filtered_dir = os.path.join(output_dir, 'keypoints_filtered')
    skeleton_frames_dir = os.path.join(output_dir, 'skeleton_frames')
    skeleton_frames_filtered_dir = os.path.join(output_dir, 'skeleton_frames_filtered')  # Nuova cartella

    os.makedirs(color_frames_dir, exist_ok=True)
    os.makedirs(depth_frames_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    os.makedirs(keypoints_filtered_dir, exist_ok=True)
    os.makedirs(skeleton_frames_dir, exist_ok=True)
    os.makedirs(skeleton_frames_filtered_dir, exist_ok=True)  # Creiamo la nuova cartella
    # Aggiungere cartella per frames sovrapposti se richiesto
    if create_overlay and use_filter:
        skeleton_overlay_dir = os.path.join(output_dir, 'skeleton_overlay')
        os.makedirs(skeleton_overlay_dir, exist_ok=True)
        print(f"   - Skeleton Overlay: {skeleton_overlay_dir}")

    print("📂 Cartelle di output create:")
    print(f"   - Color Frames: {color_frames_dir}")
    print(f"   - Depth Frames: {depth_frames_dir}")
    print(f"   - Keypoints: {keypoints_dir}")
    print(f"   - Keypoints Filtered: {keypoints_filtered_dir}")
    print(f"   - Skeleton Frames: {skeleton_frames_dir}")
    print(f"   - Skeleton Frames Filtered: {skeleton_frames_filtered_dir}")

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
    filtered_frames_for_video = []  # Nuovo array per i frame filtrati
    keypoints_for_video = []
    all_keypoints_for_filtering = []  # Buffer per filtraggio

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
                
                # Aggiungi i keypoint al buffer per il filtraggio
                all_keypoints_for_filtering.append(body_keypoints)
            else:
                keypoints_3d['body'] = []  # Array vuoto per i keypoints
                keypoints_3d['reliability_score'] = 0.0  # Score basso quando non ci sono keypoint
                keypoints_3d['keypoint_scores'] = {}
                # Aggiungi un testo al frame per indicare che non sono stati trovati keypoints
                height, width = frame_with_skeleton.shape[:2]
                cv2.putText(frame_with_skeleton, "No keypoints detected", 
                           (int(width*0.1), int(height*0.5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Aggiungi una lista vuota al buffer per mantenere la sincronizzazione
                all_keypoints_for_filtering.append([])
            
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
    
    # Applicare filtro Savitzky-Golay se richiesto
    if use_filter and len(all_keypoints_for_filtering) > 0:
        print("\n🧹 Applicazione filtro Savitzky-Golay adattivo ai keypoint...")
        
        # Definire i parametri di base del filtro
        filter_params = {
            'window_length': filter_window,
            'poly_order': filter_poly,
            'threshold': filter_threshold,
            'coords': ['x', 'y', 'z']
        }
        
        # Applicare il filtro
        try:
            # Usa il filtro adattivo invece di quello standard
            filtered_keypoints = apply_adaptive_savgol_filter(all_keypoints_for_filtering, filter_params)
            
            # Aggiornare i keypoint del video
            filtered_keypoints_for_video = filtered_keypoints
            
            # Ricalcolare gli score di jitter per i keypoint filtrati e creare i frame filtrati
            filtered_frames_for_video = []

            for i, frame_keypoints in enumerate(filtered_keypoints):
                if frame_keypoints and i < len(frames_for_video):
                    # Calcola lo score di jitter/affidabilità per questo frame
                    overall_score, keypoint_scores = calculate_jitter_score(frame_keypoints, i)
                    
                    # Crea JSON per keypoints filtrati
                    keypoints_3d_filtered = {
                        'frame': i,
                        'reliability_score': overall_score,
                        'keypoint_scores': keypoint_scores,
                        'body': frame_keypoints,
                        'filtered': True
                    }
                    
                    # Salva nella cartella dei keypoints filtrati
                    keypoints_filtered_filename = os.path.join(keypoints_filtered_dir, f'keypoints_{i:04d}.json')
                    with open(keypoints_filtered_filename, 'w') as f:
                        json.dump(keypoints_3d_filtered, f, indent=4)
                    
                    # Creiamo i landmark filtrati per il disegno
                    filtered_landmarks = []
                    for kp in frame_keypoints:
                        landmark = type('RotatedLandmark', (), {
                            'x': kp['x'],
                            'y': kp['y'],
                            'z': kp['z'],
                            'visibility': kp['visibility']
                        })
                        filtered_landmarks.append(landmark)
                    
                    # Prendiamo l'immagine originale a colori
                    color_filename = os.path.join(color_frames_dir, f'color_frame_{i:04d}.png')
                    if os.path.exists(color_filename):
                        # Leggiamo l'immagine originale
                        original_color_image = cv2.imread(color_filename)
                        # Convertiamo da BGR a RGB
                        original_color_image_rgb = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2RGB)
                        
                        # Disegniamo lo scheletro filtrato sull'immagine pulita
                        filtered_frame = original_color_image_rgb.copy()
                        filtered_frame = draw_skeleton(filtered_frame, filtered_landmarks, 
                                 point_color=(255, 255, 0), line_color=(255, 0, 0))
                        
                        # Aggiungiamo il frame al video
                        filtered_frames_for_video.append(filtered_frame)
                        
                        # Salviamo il frame filtrato
                        filtered_skeleton_filename = os.path.join(skeleton_frames_filtered_dir, f'skeleton_frame_{i:04d}.png')
                        cv2.imwrite(filtered_skeleton_filename, cv2.cvtColor(filtered_frame, cv2.COLOR_RGB2BGR))
                    else:
                        print(f"   ⚠️ File originale non trovato: {color_filename}")
                        # Creiamo un frame vuoto come fallback
                        if filtered_frames_for_video:
                            filtered_frame = filtered_frames_for_video[-1].copy()
                        else:
                            filtered_frame = frames_for_video[i].copy()
                        
                        filtered_frames_for_video.append(filtered_frame)
                        
                        # Salviamo il frame filtrato
                        filtered_skeleton_filename = os.path.join(skeleton_frames_filtered_dir, f'skeleton_frame_{i:04d}.png')
                        cv2.imwrite(filtered_skeleton_filename, cv2.cvtColor(filtered_frame, cv2.COLOR_RGB2BGR))
                else:
                    # Keypoints vuoti o indice fuori range
                    if i < len(frames_for_video):
                        # Creiamo JSON vuoto
                        keypoints_3d_filtered = {
                            'frame': i,
                            'reliability_score': 0.0,
                            'keypoint_scores': {},
                            'body': [],
                            'filtered': True
                        }
                        
                        keypoints_filtered_filename = os.path.join(keypoints_filtered_dir, f'keypoints_{i:04d}.json')
                        with open(keypoints_filtered_filename, 'w') as f:
                            json.dump(keypoints_3d_filtered, f, indent=4)
                        
                        # Salviamo il frame originale
                        color_filename = os.path.join(color_frames_dir, f'color_frame_{i:04d}.png')
                        if os.path.exists(color_filename):
                            original_color_image = cv2.imread(color_filename)
                            original_color_image_rgb = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2RGB)
                            
                            filtered_frame = original_color_image_rgb.copy()
                            height, width = filtered_frame.shape[:2]
                            cv2.putText(filtered_frame, "No keypoints detected", 
                                    (int(width*0.1), int(height*0.5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            filtered_frames_for_video.append(filtered_frame)
                            
                            # Salviamo il frame filtrato
                            filtered_skeleton_filename = os.path.join(skeleton_frames_filtered_dir, f'skeleton_frame_{i:04d}.png')
                            cv2.imwrite(filtered_skeleton_filename, cv2.cvtColor(filtered_frame, cv2.COLOR_RGB2BGR))
                        else:
                            print(f"   ⚠️ File originale non trovato: {color_filename}")
                            if filtered_frames_for_video:
                                filtered_frame = filtered_frames_for_video[-1].copy()
                                filtered_frames_for_video.append(filtered_frame)
                            else:
                                blank = np.zeros_like(frames_for_video[i])
                                filtered_frames_for_video.append(blank)
                            
                            # Salviamo il frame filtrato
                            filtered_skeleton_filename = os.path.join(skeleton_frames_filtered_dir, f'skeleton_frame_{i:04d}.png')
                            cv2.imwrite(filtered_skeleton_filename, cv2.cvtColor(filtered_frames_for_video[-1], cv2.COLOR_RGB2BGR))
            
            # Salvare keypoints filtrati in un file JSON globale
            filtered_keypoints_filename = os.path.join(output_dir, 'video_keypoints_filtered.json')
            with open(filtered_keypoints_filename, 'w') as f:
                json.dump(filtered_keypoints_for_video, f, indent=4)
            
            print(f"   ✅ Filtro applicato con successo a {len(filtered_keypoints)} frame")
            print(f"   ✅ Keypoints filtrati salvati in: {filtered_keypoints_filename}")
            
            if use_filter and create_overlay:
                print("\n🎨 Generazione frames con scheletri sovrapposti...")
                overlay_frames = []
                
                # Per ogni frame, crea un'immagine con entrambi gli scheletri
                for i in range(min(len(all_keypoints_for_filtering), len(filtered_keypoints))):
                    original_kps = all_keypoints_for_filtering[i]
                    filtered_kps = filtered_keypoints[i]
                    
                    # Carica immagine a colori originale
                    color_filename = os.path.join(color_frames_dir, f'color_frame_{i:04d}.png')
                    if os.path.exists(color_filename):
                        # Leggi l'immagine originale
                        original_color_image = cv2.imread(color_filename)
                        # Converti da BGR a RGB
                        original_color_image_rgb = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2RGB)
                        
                        # Crea landmarks originali
                        original_landmarks = []
                        if original_kps:
                            for kp in original_kps:
                                # Assicuriamoci che abbiamo le chiavi necessarie
                                if 'x' in kp and 'y' in kp:
                                    landmark = type('RotatedLandmark', (), {
                                        'x': kp['x'],
                                        'y': kp['y'],
                                        'z': kp.get('z', 0),
                                        'visibility': kp.get('visibility', 1.0)
                                    })
                                    original_landmarks.append(landmark)
                        
                        # Crea landmarks filtrati
                        filtered_landmarks = []
                        if filtered_kps:
                            for kp in filtered_kps:
                                # Assicuriamoci che abbiamo le chiavi necessarie
                                if 'x' in kp and 'y' in kp:
                                    landmark = type('RotatedLandmark', (), {
                                        'x': kp['x'],
                                        'y': kp['y'],
                                        'z': kp.get('z', 0),
                                        'visibility': kp.get('visibility', 1.0)
                                    })
                                    filtered_landmarks.append(landmark)
                        
                        # Crea immagine sovrapposta solo se abbiamo landmarks
                        if original_landmarks or filtered_landmarks:
                            overlay_frame = create_overlaid_skeletons(original_landmarks, filtered_landmarks, original_color_image_rgb)
                            overlay_frames.append(overlay_frame)
                            
                            # Salva frame con overlay
                            overlay_filename = os.path.join(skeleton_overlay_dir, f'skeleton_overlay_{i:04d}.png')
                            cv2.imwrite(overlay_filename, cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
                        else:
                            # Se non abbiamo landmarks, usa l'immagine originale
                            overlay_frames.append(original_color_image_rgb)
                            overlay_filename = os.path.join(skeleton_overlay_dir, f'skeleton_overlay_{i:04d}.png')
                            cv2.imwrite(overlay_filename, original_color_image)
                    else:
                        print(f"   ⚠️ File originale non trovato: {color_filename}")
                
                # Genera video con overlay
                if overlay_frames:
                    print("\n🎥 Generazione video con scheletri sovrapposti...")
                    overlay_video_filename = os.path.join(output_dir, 'skeleton_overlay.mp4')
                    height, width = overlay_frames[0].shape[:2]
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(overlay_video_filename, fourcc, color_fps, (width, height), isColor=True)
                    
                    frame_count = 0
                    for frame in tqdm(overlay_frames, desc="Scrittura video overlay", unit="frame"):
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        frame_count += 1
                    
                    out.release()
                    print(f"   ✅ Video con scheletri sovrapposti salvato in: {overlay_video_filename}")

            # Generare video con scheletro filtrato
            if filtered_frames_for_video:
                print("\n🎥 Generazione video dello scheletro filtrato...")
                filtered_video_filename = os.path.join(output_dir, 'skeleton_animation_filtered.mp4')
                
                if filtered_frames_for_video[0].shape == frames_for_video[0].shape:
                    height, width = filtered_frames_for_video[0].shape[:2]
                    
                    # Usa il FPS originale del bag per il video di output
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(filtered_video_filename, fourcc, color_fps, (width, height), isColor=True)
                    
                    frame_count_in_video = 0
                    for frame in tqdm(filtered_frames_for_video, desc="Scrittura video filtrato", unit="frame"):
                        out.write(frame)
                        frame_count_in_video += 1
                    
                    out.release()
                    print(f"   ✅ Video scheletro filtrato salvato in: {filtered_video_filename}")
                
                # Creare video comparativo
                print("\n🎥 Generazione video comparativo (originale vs filtrato)...")
                comparison_video_filename = os.path.join(output_dir, 'skeleton_comparison.mp4')
                create_comparison_video(frames_for_video, filtered_frames_for_video, 
                                        comparison_video_filename, color_fps)
        except Exception as e:
            print(f"   ❌ Errore nell'applicazione del filtro: {e}")

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
    if use_filter and create_overlay:
        print(f"   Frames con scheletri sovrapposti salvati in: {os.path.join(output_dir, 'skeleton_overlay')}")
        print(f"   Video con scheletri sovrapposti salvato in: {os.path.join(output_dir, 'skeleton_overlay.mp4')}")

def main():
    # Configurazione parser per argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Elabora file bag RealSense per estrarre frame e keypoints')
    parser.add_argument('bag_path', help='Percorso del file bag da elaborare')
    parser.add_argument('output_dir', help='Cartella di output per i frame e keypoints')
    parser.add_argument('--max-frames', type=int, help='Numero massimo di frame da elaborare (predefinito: tutti)')
    parser.add_argument('--test', action='store_true', help='Modalità test (limita a 500 frame)')
    
    # Parametri per il filtro Savitzky-Golay con valori predefiniti migliorati
    parser.add_argument('--filter', action='store_true', help='Attiva il filtro Savitzky-Golay adattivo')
    parser.add_argument('--filter-window', type=int, default=15, help='Dimensione base finestra del filtro (default: 15)')
    parser.add_argument('--filter-poly', type=int, default=2, help='Ordine polinomiale base del filtro (default: 2)')
    parser.add_argument('--filter-threshold', type=float, default=10.0, 
                        help='Soglia di base per applicare il filtro (default: 10.0)')
    
    # Nuovo parametro per generare frames con scheletri sovrapposti
    parser.add_argument('--overlay', action='store_true', 
                       help='Genera frames e video con scheletri originali e filtrati sovrapposti')
    
    args = parser.parse_args()
    
    # Correzione: Se --test è attivo, imposta max_frames a 500, indipendentemente da cosa è stato specificato
    if args.test:
        args.max_frames = 500
        print("Modalità test attivata: elaborazione limitata a 500 frame")
    
    # Chiamare la funzione principale con i parametri del filtro e l'opzione overlay
    process_bag_file(args.bag_path, args.output_dir, args.max_frames, 
                     args.filter, args.filter_window, args.filter_poly, args.filter_threshold,
                     args.overlay)

# Esempio di utilizzo
if __name__ == "__main__":
    main()