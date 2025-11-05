#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VBT desde .mp4 con YOLOv11-pose (Ultralytics) — Deadlift/Squat/Biceps
VERSIÓN OPTIMIZADA PARA MÁXIMA DETECCIÓN DE REPETICIONES
"""

import argparse, time, threading, queue, platform, warnings
from collections import deque
import numpy as np, pandas as pd, cv2
from dataclasses import dataclass
from scipy.signal import savgol_filter
from ultralytics import YOLO

# ---------------- Constantes ----------------
COCO_KP = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]

@dataclass
class RTSettings:
    subject_height_m: float
    g: float = 9.81
    v_thr_mps_min: float = 0.008  # UMBRAL REDUCIDO para máxima sensibilidad
    min_rep_time_s: float = 0.4   # TIEMPO MÍNIMO REDUCIDO
    min_conc_time_s: float = 0.15 # TIEMPO CONCÉNTRICA REDUCIDO
    sg_win_ms: int = 91           # VENTANA MÁS CORTA para mejor respuesta
    sg_poly: int = 2              # POLINOMIO REDUCIDO
    height_window: int = 121
    max_queue: int = 4
    skip_rate: int = 0
    print_every: float = 1.0
    kp_conf_thr: float = 0.25     # CONFIANZA REDUCIDA para más keypoints
    slow_mode: bool = False
    v_thr_enter_factor: float = 0.80   # HISTÉRESIS REDUCIDA - entra antes
    v_thr_exit_factor: float  = 0.30   # HISTÉRESIS REDUCIDA - sale antes
    hold_ms: int = 500                  # HOLD AUMENTADO para más estabilidad
    rom_min_m: float = 0.02             # ROM MÍNIMA REDUCIDA para reps cortas
    view: str = 'auto'

# ---------------- Utilidades pose ----------------
def select_main_person(result):
    if result.boxes is None or len(result.boxes)==0: return None
    boxes = result.boxes.xyxy
    if boxes is None or len(boxes)==0: return None
    boxes = boxes.detach().cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return int(np.argmax(areas))

def extract_keypoints_xy(result, person_idx):
    if result.keypoints is None or result.keypoints.data is None or person_idx is None:
        return None
    if len(result.keypoints.data) <= person_idx: return None
    kps = result.keypoints.data[person_idx].detach().cpu().numpy()
    out={}
    for i,name in enumerate(COCO_KP):
        x,y,c = kps[i]
        out[name] = (float(x), float(y), float(c))
    return out

def weighted_y(kpd, names, conf_thr=0.3):
    vals, confs = [], []
    for n in names:
        if n in kpd:
            _,y,c = kpd[n]
            if c>=conf_thr and not np.isnan(y):
                vals.append(y); confs.append(c)
    if not vals: return np.nan
    vals, confs = np.array(vals,float), np.array(confs,float)
    w = confs / (np.sum(confs)+1e-12)
    return float(np.sum(w*vals))

# ---------------- Escala px→m robusta ----------------
class OnlineHeightScaler:
    """
    Estima m/px con cabeza↔tobillos; fallback: altura del bounding box (y2-y1).
    Usa mediana deslizante + rechazo por MAD.
    """
    def __init__(self, subject_height_m: float, window: int = 121):
        self.h_m = float(subject_height_m)
        self.win = max(11, int(window)|1)
        self.buffer = deque(maxlen=self.win)
        self.last_bbox_h = None

    def update_bbox(self, result):
        try:
            if result.boxes is not None and len(result.boxes)>0:
                box = result.boxes.xyxy[0].detach().cpu().numpy()
                y1,y2 = box[1], box[3]
                self.last_bbox_h = max(1.0, float(y2-y1))
        except Exception:
            pass

    def update_and_get_scale(self, kpd: dict):
        hpx = None
        if kpd:
            y_head = np.nanmedian([
                kpd.get('nose',(np.nan,np.nan,0))[1],
                kpd.get('left_eye',(np.nan,np.nan,0))[1],
                kpd.get('right_eye',(np.nan,np.nan,0))[1],
                kpd.get('left_ear',(np.nan,np.nan,0))[1],
                kpd.get('right_ear',(np.nan,np.nan,0))[1],
            ])
            y_ank = np.nanmedian([
                kpd.get('left_ankle',(np.nan,np.nan,0))[1],
                kpd.get('right_ankle',(np.nan,np.nan,0))[1],
            ])
            if not np.isnan(y_head) and not np.isnan(y_ank):
                hpx = max(1.0, y_ank - y_head)

        if hpx is None and self.last_bbox_h is not None:
            # fallback si no hay KP suficientes
            hpx = float(self.last_bbox_h)

        if hpx is None:
            return self.current_scale()

        self.buffer.append(hpx)
        return self.current_scale()

    def current_scale(self):
        if len(self.buffer) < max(11, self.win//4): return None
        arr = np.array(self.buffer,float)
        med = np.median(arr); mad = np.median(np.abs(arr-med))+1e-9
        z = 0.6745*(arr-med)/mad
        arr = arr[np.abs(z) < 3.5] if np.any(np.abs(z) < 3.5) else arr
        height_px = float(np.median(arr))
        return self.h_m / height_px

# ---------------- Derivadas ----------------
class OnlineDerivatives:
    def __init__(self, fps: float, win_ms: int = 91, poly: int = 2, slow=False):
        self.fps = max(1e-6, float(fps))
        self.dt = 1.0/self.fps
        base = int(max(5, (win_ms/1000.0)*self.fps))
        if slow: base = int(base*1.5)
        self.win = int(base//2*2+1)
        if self.win <= poly: self.win = poly + 2 + (poly % 2 == 0)
        self.poly = poly
        self.buf = deque(maxlen=self.win)
    def push(self, y): self.buf.append(float(y))
    def get(self):
        arr = np.array(self.buf,float)
        if len(arr) < self.win: return None, None, None
        y_s = savgol_filter(arr, self.win, self.poly, mode='interp')
        v = np.gradient(y_s, self.dt); v_s = savgol_filter(v, self.win, self.poly, mode='interp')
        a = np.gradient(v_s, self.dt); a_s = savgol_filter(a, self.win, self.poly, mode='interp')
        return y_s[-1], v_s[-1], a_s[-1]

# ---------------- Segmentación Deadlift ----------------
class RepSegmenterDeadlift:
    """
    VERSIÓN OPTIMIZADA: Umbrales reducidos para máxima detección
    """
    def __init__(self, g=9.81, v_thr_base=0.008, enter_factor=0.80, exit_factor=0.30,
                 min_rep_time_s=0.4, min_conc_time_s=0.15, fps=30.0, rom_min_m=0.02):
        self.g=g
        self.v_thr_base=v_thr_base
        self.enter_factor=enter_factor
        self.exit_factor=exit_factor
        self.min_rep=min_rep_time_s
        self.min_conc=min_conc_time_s
        self.fps=fps
        self.rom_min_m = float(rom_min_m)

        self.t=[]; self.y=[]; self.v=[]; self.a=[]
        self.in_up=False; self.up_start=None
        self.pending_rep_start=None; self.last_down=None
        self.reps=[]; self.last_rep_metrics=None; self.last_rep_time=0.0

        # auto-thr - OPTIMIZADO: factor reducido
        self.init_samples=int(2.0*fps); self.vbuf=[]
        self.v_thr=v_thr_base
        self.v_thr_enter=self.v_thr*self.enter_factor
        self.v_thr_exit =self.v_thr*self.exit_factor

        # extremos locales (ventana aumentada → mejor detección)
        self.win_ext=5  # AUMENTADO para mejor detección de picos
        self.idx_last_top=None
        self.idx_last_bottom=None
        
        # Debug
        self._last_debug = 0.0

    def _update_thr(self):
        if len(self.vbuf) >= self.init_samples:
            p95 = np.percentile(np.abs(self.vbuf), 95)
            # FACTOR REDUCIDO: de 0.08 a 0.04 para mayor sensibilidad
            self.v_thr = max(self.v_thr_base, 0.04*float(p95))
            self.v_thr_enter=self.v_thr*self.enter_factor
            self.v_thr_exit =self.v_thr*self.exit_factor
            self.vbuf=[]

    def _is_local_extreme(self, seq, i, kind='max'):
        # MEJORADO: ventana adaptativa más robusta
        win_size = min(self.win_ext, i, len(seq)-i-1)
        if win_size < 1: return False
        
        # Verificar que sea un extremo significativo
        window = seq[i-win_size:i+win_size+1]
        if len(window) < 3: return False
        
        if kind == 'max':
            is_extreme = seq[i] == np.max(window)
            # Verificar que sea al menos X% mayor que el promedio local
            local_avg = np.mean(window)
            significance = (seq[i] - local_avg) / (local_avg + 1e-9) > 0.05  # 5% de significancia
            return is_extreme and significance
        else:
            is_extreme = seq[i] == np.min(window)
            local_avg = np.mean(window)
            significance = (local_avg - seq[i]) / (local_avg + 1e-9) > 0.05
            return is_extreme and significance

    def _t(self, i0, i1): return self.t[i1]-self.t[i0]

    def push(self, t, y, v, a):
        i=len(self.t)
        self.t.append(t); self.y.append(y); self.v.append(v); self.a.append(a)

        if len(self.v) <= self.init_samples:
            self.vbuf.append(v)
            self._update_thr()

        # MEJORADO: Detección más robusta de extremos
        if self._is_local_extreme(self.y, i, 'min'):
            self.idx_last_bottom = i
        if self._is_local_extreme(self.y, i, 'max'):
            self.idx_last_top = i

        is_up_enter = v > +self.v_thr_enter
        is_up_keep  = v > +self.v_thr_exit
        is_down_any = v < -self.v_thr

        if is_down_any: 
            self.last_down = i

        # MEJORADO: Condición de inicio más sensible pero estable
        if (not self.in_up) and is_up_enter:
            # Verificar que no sea un falso positivo por ruido
            if len(self.v) > 3 and all(vv > -self.v_thr for vv in self.v[-3:]):
                self.in_up = True
                self.up_start = i
                self.pending_rep_start = self.last_down if self.last_down is not None else max(0, i-1)

        # MEJORADO: Lógica de cierre más robusta
        if self.in_up and (not is_up_keep):
            up_end = i
            
            # ESTRATEGIA MEJORADA de cierre:
            rep_end = None
            
            # 1. Buscar descenso claro en ventana ampliada
            look_ahead = min(15, len(self.v) - up_end - 1)  # Ventana más amplia
            for j in range(up_end, up_end + look_ahead):
                if j < len(self.v) and self.v[j] < -self.v_thr:
                    rep_end = j
                    break
            
            # 2. Si no hay descenso claro, usar máximo local reciente
            if rep_end is None and self.idx_last_top is not None:
                # Ventana más permisiva para máximo local
                lo = max(0, self.up_start)
                hi = min(len(self.t)-1, up_end + 8)  # Más permisivo
                if lo <= self.idx_last_top <= hi:
                    rep_end = self.idx_last_top
            
            # 3. Último recurso: usar fin de fase concéntrica + margen
            if rep_end is None:
                rep_end = up_end + 2  # Pequeño margen
            
            if rep_end is not None and self.up_start is not None:
                i0 = self.pending_rep_start if self.pending_rep_start is not None else self.up_start
                i1 = min(rep_end, len(self.t)-1)
                
                # VERIFICACIÓN MEJORADA DE ROM
                if i1 > i0:
                    y_segment = self.y[i0:i1+1]
                    if len(y_segment) > 0:
                        rom_candidate = float(np.nanmax(y_segment) - np.nanmin(y_segment))
                        
                        # MEJORADO: Verificación más inteligente de ROM
                        min_samples = int(self.min_rep * self.fps * 0.5)  # Al menos 50% del tiempo mínimo
                        time_ok = self._t(i0, i1) >= self.min_rep
                        conc_time_ok = self._t(self.up_start, up_end) >= self.min_conc
                        rom_ok = rom_candidate >= self.rom_min_m
                        enough_samples = (i1 - i0) >= min_samples
                        
                        if time_ok and conc_time_ok and rom_ok and enough_samples:
                            self._emit(i0, i1, self.up_start, up_end)
                        else:
                            # DEBUG: Mostrar por qué se rechazó
                            if time.time() - self._last_debug > 2.0:
                                reasons = []
                                if not time_ok: reasons.append(f"time({self._t(i0,i1):.2f}<{self.min_rep})")
                                if not conc_time_ok: reasons.append(f"conc({self._t(self.up_start,up_end):.2f}<{self.min_conc})")
                                if not rom_ok: reasons.append(f"rom({rom_candidate:.3f}<{self.rom_min_m})")
                                if not enough_samples: reasons.append(f"samples({i1-i0}<{min_samples})")
                                print(f"⚠️ Rep rechazada: {', '.join(reasons)}")
                                self._last_debug = time.time()
                
                self.in_up = False
                self.up_start = None


    
    def _emit(self, i0, i1, s_up, e_up):
        y = np.array(self.y[i0:i1+1], float)
        v_conc = np.array(self.v[s_up:e_up+1], float)
        a_conc = np.array(self.a[s_up:e_up+1], float)

        t_total = float(self.t[i1]-self.t[i0])
        rom = float(np.nanmax(y)-np.nanmin(y)) - 0.08  # Corrección por ruido
        
        # ✅ NUEVA CONDICIÓN: Ignorar repeticiones con ROM < 0.20 m
        if rom < 0.19:
            print(f"❌ Rep descartada: ROM={rom:.3f} m < 0.20 m")
            return
        
        # CÁLCULO DE VMAX MEDIANTE ECUACIÓN DE MICHAELIS-MENTEN EN FASE CONCÉNTRICA
        if len(v_conc) > 0:
            # Velocidad máxima tradicional
            vmax_trad = float(np.nanmax(v_conc))
            
            # Aplicar ecuación de Michaelis-Menten a la fase concéntrica
            # v = vmax * t / (Km + t) donde t es el tiempo normalizado en la fase concéntrica
            t_conc_normalized = np.linspace(0, 1, len(v_conc))
            
            # Ajuste no lineal usando Michaelis-Menten
            try:
                from scipy.optimize import curve_fit
                
                def michaelis_menten(t, vmax, Km):
                    return vmax * t / (Km + t)
                
                # Estimación inicial de parámetros
                p0 = [vmax_trad, 0.5]  # vmax estimada, Km inicial 0.5
                
                # Ajuste de curva (evitar divisiones por cero)
                t_fit = t_conc_normalized + 1e-9
                popt, _ = curve_fit(michaelis_menten, t_fit, v_conc, p0=p0, 
                                bounds=([0, 1e-9], [vmax_trad * 2, 10]))
                
                vmax_mm = float(popt[0])  # vmax de Michaelis-Menten
                Km = float(popt[1])       # Constante de Michaelis
                
                print(f"🔬 Michaelis-Menten: Vmax={vmax_mm:.3f} m/s, Km={Km:.3f}")
                
            except Exception as e:
                print(f"⚠️  Ajuste Michaelis-Menten falló: {e}, usando Vmax tradicional")
                vmax_mm = vmax_trad
        else:
            vmax_mm = np.nan
            vmax_trad = np.nan

        # MPV propulsiva: hasta primer a < -g
        idx = np.where(a_conc < -self.g)[0]
        prop_end = (s_up + int(idx[0])) if len(idx) else e_up
        mpv = float(np.nanmean(self.v[s_up:prop_end+1])) if prop_end > s_up else np.nan

        rep = dict(
            t_ini=float(self.t[i0]), t_fin=float(self.t[i1]),
            t_total_s=t_total,
            t_concentrica_s=float(self.t[e_up]-self.t[s_up]) if e_up>s_up else np.nan,
            ROM_m=rom, 
            VMED_prop_mps=mpv, 
            VMAX_mps=vmax_mm,  # ← USAR VMAX DE MICHAELIS-MENTEN
            VMAX_trad_mps=vmax_trad  # ← Guardar también la tradicional para comparación
        )
        self.reps.append(rep); self.last_rep_metrics=rep; self.last_rep_time=self.t[i1]
        print(f"✅ REP @ {self.t[i1]:.2f}s | ROM={rom:.3f} m | MPV={0.0 if np.isnan(mpv) else mpv:.3f} m/s | Vmax_MM={0.0 if np.isnan(vmax_mm) else vmax_mm:.3f} m/s | t={t_total:.2f}s")
    def _emit1(self, i0, i1, s_up, e_up):
        y = np.array(self.y[i0:i1+1], float)
        v_conc = np.array(self.v[s_up:e_up+1], float)
        a_conc = np.array(self.a[s_up:e_up+1], float)

        t_total = float(self.t[i1]-self.t[i0])
        rom = float(np.nanmax(y)-np.nanmin(y))
        
        # ✅ NUEVA CONDICIÓN: Ignorar repeticiones con ROM < 0.35 m
        if rom < 0.20:
            print(f"❌ Rep descartada: ROM={rom:.3f} m < 0.20 m")
            return
        
        vmax = float(np.nanmax(v_conc)) if len(v_conc) else np.nan

        # MPV propulsiva: hasta primer a < -g
        idx = np.where(a_conc < -self.g)[0]
        prop_end = (s_up + int(idx[0])) if len(idx) else e_up
        mpv = float(np.nanmean(self.v[s_up:prop_end+1])) if prop_end > s_up else np.nan

        rep = dict(
            t_ini=float(self.t[i0]), t_fin=float(self.t[i1]),
            t_total_s=t_total,
            t_concentrica_s=float(self.t[e_up]-self.t[s_up]) if e_up>s_up else np.nan,
            ROM_m=rom, VMED_prop_mps=mpv, VMAX_mps=vmax
        )
        self.reps.append(rep); self.last_rep_metrics=rep; self.last_rep_time=self.t[i1]
        print(f"✅ REP @ {self.t[i1]:.2f}s | ROM={rom:.3f} m | MPV={0.0 if np.isnan(mpv) else mpv:.3f} m/s | Vmax={0.0 if np.isnan(vmax) else vmax:.3f} m/s | t={t_total:.2f}s")
    def flush_close(self):
        """MEJORADO: Forzado al terminar el video: cierra rep abierta con criterios más permisivos"""
        if self.in_up and self.up_start is not None:
            up_end = len(self.t) - 1
            # MEJORADO: Estrategia más agresiva para cerrar reps al final
            candidates = []
            
            # Opción 1: Último máximo local
            if self.idx_last_top is not None and self.idx_last_top >= self.up_start:
                candidates.append(self.idx_last_top)
            
            # Opción 2: Punto donde la velocidad se estabiliza
            for i in range(up_end, max(self.up_start, up_end-10), -1):
                if i < len(self.v) and abs(self.v[i]) < self.v_thr_exit:
                    candidates.append(i)
                    break
            
            # Opción 3: Simplemente usar el final
            candidates.append(up_end)
            
            for rep_end in candidates:
                i0 = self.pending_rep_start if self.pending_rep_start is not None else self.up_start
                i1 = rep_end
                
                if i1 > i0:
                    y_segment = self.y[i0:i1+1]
                    if len(y_segment) > 0:
                        rom_candidate = float(np.nanmax(y_segment) - np.nanmin(y_segment))
                        time_ok = self._t(i0, i1) >= (self.min_rep * 0.8)  # 80% del tiempo mínimo
                        conc_time_ok = self._t(self.up_start, up_end) >= (self.min_conc * 0.8)
                        rom_ok = rom_candidate >= (self.rom_min_m * 0.7)  # 70% del ROM mínimo
                        
                        if time_ok and conc_time_ok and rom_ok:
                            self._emit(i0, i1, self.up_start, up_end)
                            print(f"✅ REP FINAL @ {self.t[i1]:.2f}s (flush)")
                            break
            
            self.in_up = False
            self.up_start = None

# ---------------- Captura sin pérdida ----------------
class FrameSourceThread(threading.Thread):
    def __init__(self, video_path, q: queue.Queue, max_queue: int, skip_rate: int):
        super().__init__(daemon=True)
        self.video_path=video_path; self.q=q
        self.max_queue=max_queue; self.skip_rate=max(0,skip_rate)
        self.stop=False; self.fps=None; self.frame_count=0
        self.duration=None; self.frame_size=None
    def run(self):
        cap=cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total= cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration=(total/self.fps) if total and total>0 else None
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size=(w,h)
        idx=0
        while not self.stop:
            ret, frame = cap.read()
            if not ret: break
            if self.skip_rate>0 and (idx % (self.skip_rate+1) != 0):
                idx+=1; continue
            self.q.put((idx, frame), block=True)   # ← bloqueante → no se pierden frames
            idx+=1; self.frame_count=idx
        cap.release(); self.q.put(None)

# ---------------- Inferencia + métricas ----------------
class InferenceAndMetricsThread(threading.Thread):
    def __init__(self, q, model: YOLO, settings: RTSettings,
                 imgsz:int, conf:float, iou:float, device:str, half:bool,
                 exercise:str, dominant:str, save_video:str, frame_size, fps_cap: float,
                 draw_pose: bool, disp_q: queue.Queue=None):
        super().__init__(daemon=True)
        self.q=q; self.model=model; self.settings=settings
        self.imgsz=imgsz; self.conf=conf; self.iou=iou
        self.device=device; self.half=half
        self.exercise=exercise.lower(); self.dominant=dominant.lower()
        self.save_video=save_video; self.frame_size=frame_size; self.fps_cap=fps_cap
        self.draw_pose=draw_pose; self.disp_q=disp_q

        self.eff_fps = fps_cap / max(1, (self.settings.skip_rate+1))
        self.scaler=None; self.deriv=None; self.segmenter=None
        self.m_per_px=None; self.results_df=None
        self.live_y=np.nan; self.live_v=np.nan; self.live_a=np.nan
        self.kp_debug=False; self.writer=None
        self.last_print=0.0

        # "hold" del marcador si cae la confianza
        self.last_y_px=None; self.last_y_ts=None

        # --- estado de vista en tiempo real (auto → frontal/lateral) con suavizado ---
        self.view_runtime = (self.settings.view or 'auto').lower()
        self._view_score = 0.0   # EMA: >0 → frontal, <0 → lateral
        self._ema_alpha = 0.15   # suavizado para no "flippear"

        if self.save_video and self.frame_size and self.fps_cap:
            fourcc=cv2.VideoWriter_fourcc(*'mp4v')
            self.writer=cv2.VideoWriter(self.save_video, fourcc, self.eff_fps,
                                        (self.frame_size[0], self.frame_size[1]))

    def _warn_no_kp(self):
        now=time.time()
        if not hasattr(self,"_last_kp_warn") or (now - getattr(self,"_last_kp_warn",0)) > 2.0:
            print("⚠️  Sin keypoints en este frame. Revisa luz/encuadre o usa --device cpu y un yolo11*-pose.pt")
            self._last_kp_warn = now

    # --------- Inferencia de orientación (auto) ---------
    def _infer_view(self, kpd):
        """
        Decide 'frontal' o 'lateral' a partir de la VISIBILIDAD relativa de lados (izq/der).
        Heurística robusta + EMA para estabilidad.
        """
        if not kpd:
            return self.view_runtime

        thr = max(0.15, 0.8 * self.settings.kp_conf_thr)

        def side_score(prefix):
            s = 0.0
            for kp in [f'{prefix}_wrist', f'{prefix}_elbow', f'{prefix}_shoulder',
                       f'{prefix}_hip', f'{prefix}_knee', f'{prefix}_ankle']:
                if kp in kpd:
                    c = float(kpd[kp][2])
                    if not np.isnan(c):
                        s += max(0.0, c)
            return s

        sL, sR = side_score('left'), side_score('right')
        s_sum = sL + sR + 1e-9
        bal = abs(sL - sR) / s_sum

        wrL = float(kpd.get('left_wrist',  (np.nan, np.nan, 0.0))[2])
        wrR = float(kpd.get('right_wrist', (np.nan, np.nan, 0.0))[2])
        shL = float(kpd.get('left_shoulder',  (np.nan, np.nan, 0.0))[2])
        shR = float(kpd.get('right_shoulder', (np.nan, np.nan, 0.0))[2])

        wrists_ok    = (wrL >= thr) and (wrR >= thr)
        shoulders_ok = (shL >= thr) and (shR >= thr)

        if wrists_ok and shoulders_ok and bal < 0.25:
            want = 'frontal'
        elif bal > 0.45:
            want = 'lateral'
        else:
            want = self.view_runtime if self.view_runtime in ('frontal','lateral') else 'frontal'

        target = +1.0 if want == 'frontal' else -1.0
        self._view_score = (1 - self._ema_alpha)*self._view_score + self._ema_alpha*target

        new_view = self.view_runtime
        if self._view_score > +0.20: new_view = 'frontal'
        if self._view_score < -0.20: new_view = 'lateral'
        return new_view

    # --------- Marcador "vista-aware" ---------
    def _marker_y_px(self, kpd, view_override=None):
        """
        DEADLIFT:
          - FRONTAL: muñeca más baja (mayor y) entre izquierda/derecha; fallback caderas/tobillos.
          - LATERAL: usa el LADO VISIBLE. Prioridad: muñeca → codo → cadera → tobillo del MISMO lado.
        SQUAT:
          - caderas (media ponderada izq/der).
        BICEPS:
          - muñeca del dominante si está; si no, media ponderada de ambas.
        """
        if not kpd:
            return np.nan

        view = (view_override or self.view_runtime or self.settings.view or 'auto').lower()

        def yz(name):
            if name not in kpd:
                return (np.nan, 0.0)
            return (kpd[name][1], kpd[name][2])  # y, conf

        def choose_best_side():
            # Elige lado por suma de confs (wrist/elbow/shoulder/hip/knee/ankle)
            def side_score(prefix):
                s, valid = 0.0, 0
                for kp in [f'{prefix}_wrist', f'{prefix}_elbow', f'{prefix}_shoulder',
                           f'{prefix}_hip', f'{prefix}_knee', f'{prefix}_ankle']:
                    if kp in kpd:
                        c = float(kpd[kp][2])
                        if not np.isnan(c):
                            s += max(0.0, c); valid += 1
                return s, valid

            sL, nL = side_score('left')
            sR, nR = side_score('right')
            if nL == 0 and nR == 0:
                return 'none'
            # si una muñeca es claramente más confiable, usa ese lado
            _, cL = yz('left_wrist')
            _, cR = yz('right_wrist')
            if cL >= (cR + 0.10): return 'left'
            if cR >= (cL + 0.10): return 'right'
            # si no, por score total
            return 'left' if sL >= sR else 'right'

        if self.exercise == 'deadlift':
            if view == 'lateral':
                side = choose_best_side()
                prefs = {
                    'left' : ['left_wrist','left_elbow','left_hip','left_ankle'],
                    'right': ['right_wrist','right_elbow','right_hip','right_ankle'],
                }
                for kp in prefs.get(side, []) + ['left_wrist','right_wrist','left_elbow','right_elbow',
                                                 'left_hip','right_hip','left_ankle','right_ankle']:
                    yy, cc = yz(kp)
                    if not np.isnan(yy) and cc >= self.settings.kp_conf_thr:
                        return yy
                return np.nan
            else:
                # frontal/auto → muñeca más baja (mayor y)
                yl, cl = yz('left_wrist')
                yr, cr = yz('right_wrist')
                cand=[]
                if not np.isnan(yl) and cl >= self.settings.kp_conf_thr: cand.append(yl)
                if not np.isnan(yr) and cr >= self.settings.kp_conf_thr: cand.append(yr)
                if cand:
                    return float(np.max(cand))
                # fallback: caderas → tobillos
                y = weighted_y(kpd, ['left_hip','right_hip'], self.settings.kp_conf_thr)
                if np.isnan(y):
                    y = weighted_y(kpd, ['left_ankle','right_ankle'], self.settings.kp_conf_thr)
                return y

        elif self.exercise == 'squat':
            return weighted_y(kpd, ['left_hip','right_hip'], self.settings.kp_conf_thr)

        elif self.exercise == 'biceps':
            if self.dominant == 'right':
                y, c = yz('right_wrist')
                if not np.isnan(y) and c >= self.settings.kp_conf_thr: return y
            elif self.dominant == 'left':
                y, c = yz('left_wrist')
                if not np.isnan(y) and c >= self.settings.kp_conf_thr: return y
            return weighted_y(kpd, ['left_wrist','right_wrist'], self.settings.kp_conf_thr)

        return np.nan

    # --------- Hold + conversión a metros ---------
    def _marker_y_m_with_hold(self, kpd, t_now, view_for_frame=None):
        y_px = self._marker_y_px(kpd, view_override=view_for_frame) if kpd else np.nan
        if np.isnan(y_px):
            if self.last_y_px is not None and self.last_y_ts is not None:
                if (t_now - self.last_y_ts) <= (self.settings.hold_ms/1000.0):
                    y_px = self.last_y_px
        else:
            self.last_y_px = y_px; self.last_y_ts = t_now

        if np.isnan(y_px) or self.m_per_px is None:
            return None
        return -y_px * self.m_per_px  # invertir eje: arriba positivo

    def _draw_overlay(self, frame, reps, fps_proc, t_now):
        x0,y0,lh=12,24,22; col=(255,255,255)
        ov=frame.copy()
        cv2.rectangle(ov,(5,5),(600,210),(0,0,0),-1)
        cv2.addWeighted(ov,0.35,frame,0.65,0,frame)
        # Vista en tiempo real
        cv2.putText(frame,f"Exercise: {self.exercise.upper()} | View: {self.view_runtime.upper()}",
                    (x0,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)
        cv2.putText(frame,f"Reps: {reps}",(x0,y0+lh),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)
        cv2.putText(frame,f"m/px: {0.0 if self.m_per_px is None else self.m_per_px:.6f}",
                    (x0,y0+2*lh),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)
        cv2.putText(frame,f"v_inst (m/s): {np.nan_to_num(self.live_v,nan=0.0):+.3f}",
                    (x0,y0+3*lh),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)
        cv2.putText(frame,f"FPS_proc: {fps_proc:.1f}",(x0,y0+4*lh),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)

        seg = self.segmenter
        if seg is not None and seg.last_rep_metrics is not None and (t_now - seg.last_rep_time) <= 3.0:
            rep = seg.last_rep_metrics
            text=(f"LAST REP  ROM={rep.get('ROM_m',np.nan):.3f} m  "
                  f"MPV={rep.get('VMED_prop_mps',np.nan):.3f} m/s  "
                  f"Vmax={rep.get('VMAX_mps',np.nan):.3f} m/s  "
                  f"t={rep.get('t_total_s',np.nan):.2f}s")
            H=frame.shape[0]
            ov2=frame.copy()
            cv2.rectangle(ov2,(5,H-42),(frame.shape[1]-5,H-6),(0,0,0),-1)
            cv2.addWeighted(ov2,0.35,frame,0.65,0,frame)
            cv2.putText(frame,text,(12,H-16),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
        return frame

    def run(self):
        start_wall=time.time(); processed=0
        # Inicializaciones con FPS efectivo
        self.deriv = OnlineDerivatives(self.eff_fps, self.settings.sg_win_ms, self.settings.sg_poly, slow=self.settings.slow_mode)
        self.segmenter = RepSegmenterDeadlift(
            g=self.settings.g, v_thr_base=self.settings.v_thr_mps_min,
            enter_factor=self.settings.v_thr_enter_factor, exit_factor=self.settings.v_thr_exit_factor,
            min_rep_time_s=self.settings.min_rep_time_s, min_conc_time_s=self.settings.min_conc_time_s,
            fps=self.eff_fps, rom_min_m=self.settings.rom_min_m
        )
        self.scaler = OnlineHeightScaler(self.settings.subject_height_m, self.settings.height_window)

        while True:
            item=self.q.get()
            if item is None: break
            idx, frame = item
            t_now = idx / self.eff_fps

            # Inferencia
            rlist = self.model.predict(
                frame, imgsz=self.imgsz, conf=max(self.conf,0.30), iou=self.iou,
                device=self.device, half=self.half, classes=[0], max_det=1, verbose=False
            )
            result = rlist[0]

            kp_none = (getattr(result,"keypoints",None) is None or
                       getattr(result.keypoints,"data",None) is None or
                       len(result.keypoints.data)==0)
            if kp_none: self._warn_no_kp()

            if self.draw_pose and not kp_none:
                try: frame = result.plot()
                except Exception: pass

            pidx = select_main_person(result) if not kp_none else None
            kpd  = extract_keypoints_xy(result, pidx) if (pidx is not None and not kp_none) else None

            # Escala (actualiza bbox fallback + KP)
            self.scaler.update_bbox(result)
            m_per_px = self.scaler.update_and_get_scale(kpd)
            if m_per_px is not None: self.m_per_px = m_per_px

            # --- vista en tiempo real ---
            if (self.settings.view or 'auto').lower() == 'auto':
                self.view_runtime = self._infer_view(kpd)
            else:
                self.view_runtime = (self.settings.view or 'frontal').lower()

            # Marcador con hold + derivadas + segmentación
            y_m = self._marker_y_m_with_hold(kpd, t_now, view_for_frame=self.view_runtime)
            if y_m is not None:
                self.deriv.push(y_m)
                y_s, v_s, a_s = self.deriv.get()
                if y_s is not None:
                    self.segmenter.push(t_now, y_s, v_s, a_s)
                    processed += 1
                    self.live_y, self.live_v, self.live_a = y_s, v_s, a_s

            # Overlay
            reps = len(self.segmenter.reps) if self.segmenter is not None else 0
            fps_proc = processed / max(1e-6, (time.time()-start_wall))
            frame = self._draw_overlay(frame, reps, fps_proc, t_now)

            # Display / Guardado
            if self.disp_q is not None:
                self.disp_q.put(frame, block=True)
            if self.writer is not None:
                self.writer.write(frame)

            if (time.time()-self.last_print) >= self.settings.print_every:
                print(f"[RT {self.exercise.upper()}] idx={idx} | reps={reps} | FPS_in_eff={self.eff_fps:.2f} | m/px={self.m_per_px or 0:.6f} | v_thr≈{self.segmenter.v_thr:.3f}")
                self.last_print=time.time()

        # flush final por si quedó una rep abierta
        self.segmenter.flush_close()
        self.results_df = pd.DataFrame(self.segmenter.reps if self.segmenter is not None else [])
        if self.writer is not None: self.writer.release()

# ---------------- Validación modelo ----------------
def _validate_pose_model(path_or_model):
    try:
        m = path_or_model if isinstance(path_or_model, YOLO) else YOLO(path_or_model)
        if getattr(m,"task",None) != "pose":
            warnings.warn("El modelo cargado no es de tipo 'pose'. Usa un yolo11*-pose.pt")
        return m
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo pose: {e}")

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="VBT con YOLOv11-pose + overlays y CSV - VERSIÓN OPTIMIZADA")
    ap.add_argument('--exercise', choices=['deadlift','squat','biceps'], required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--height', type=float, required=True)
    ap.add_argument('--model', default='yolo11n-pose.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--imgsz', type=int, default=448)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--skip_rate', type=int, default=0)
    ap.add_argument('--max_queue', type=int, default=4)
    ap.add_argument('--min_rep_s', type=float, default=0.4)
    ap.add_argument('--min_conc_s', type=float, default=0.15)
    ap.add_argument('--sg_win_ms', type=int, default=91)
    ap.add_argument('--sg_poly', type=int, default=2)
    ap.add_argument('--print_every', type=float, default=1.0)
    ap.add_argument('--dominant', choices=['left','right','auto'], default='auto')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--save_video', default=None)
    ap.add_argument('--draw_pose', action='store_true')
    ap.add_argument('--kp_conf', type=float, default=0.25)
    ap.add_argument('--slow_mode', action='store_true')
    ap.add_argument('--rom_min_m', type=float, default=0.02, help='ROM mínima (m) para validar/cerrar la repetición')
    ap.add_argument('--view', choices=['auto','frontal','lateral'], default='auto',
                    help='Orientación del sujeto: auto|frontal|lateral')
    # AÑADIR el argumento faltante
    ap.add_argument('--v_thr_mps_min', type=float, default=0.008, help='Umbral mínimo de velocidad para detección')
    return ap.parse_args()

# ---------------- Main ----------------
# ---------------- Main ----------------
def main1():
    args = parse_args()

    # Device por defecto → CPU en macOS por bug MPS Pose
    if args.device == 'auto':
        args.device = 'cpu' if platform.system().lower()=='darwin' else 'cpu'
    if args.device != 'cuda':
        args.half = False

    model = _validate_pose_model(args.model)

    cap0=cv2.VideoCapture(args.video)
    fps_cap = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w0=int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)); h0=int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap0.release()

    rts = RTSettings(
        subject_height_m=args.height, g=9.81,
        v_thr_mps_min=args.v_thr_mps_min, min_rep_time_s=args.min_rep_s, min_conc_time_s=args.min_conc_s,
        sg_win_ms=args.sg_win_ms, sg_poly=args.sg_poly, height_window=121, max_queue=args.max_queue,
        skip_rate=max(0,args.skip_rate), print_every=args.print_every, kp_conf_thr=args.kp_conf,
        slow_mode=args.slow_mode, rom_min_m=args.rom_min_m, view=args.view
    )

    q = queue.Queue(maxsize=max(2, args.max_queue))
    t_cap=FrameSourceThread(args.video, q, max_queue=rts.max_queue, skip_rate=rts.skip_rate)

    disp_q = queue.Queue(maxsize=1) if args.show else None
    t_inf=InferenceAndMetricsThread(
        q, model, rts, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, half=args.half, exercise=args.exercise, dominant=args.dominant,
        save_video=args.save_video, frame_size=(w0,h0), fps_cap=fps_cap,
        draw_pose=args.draw_pose, disp_q=disp_q
    )

    t_cap.start(); t_inf.start()

    # Previsualización temporizada por FPS efectivo (sin acelerarse)
    if args.show:
        cv2.namedWindow("VBT RT + Overlays", cv2.WINDOW_NORMAL)
        start_show = time.time()
        eff_fps = t_inf.eff_fps
        idx_shown = -1
        while t_inf.is_alive() or not q.empty() or not disp_q.empty():
            try:
                frame_vis = disp_q.get(timeout=0.5)
                idx_shown += 1
                target = start_show + (idx_shown / max(1e-6, eff_fps))
                sleep_t = target - time.time()
                if sleep_t > 0: time.sleep(sleep_t)
                cv2.imshow("VBT RT + Overlays", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except queue.Empty:
                # Continuar procesando aunque la cola de display esté vacía
                if not t_inf.is_alive() and q.empty():
                    break
        cv2.destroyAllWindows()
    else:
        # Esperar a que se procesen TODOS los frames, no solo que el hilo esté vivo
        while t_inf.is_alive() or not q.empty():
            time.sleep(0.01)

    # Asegurar que se procesen todos los frames pendientes antes de hacer join
    print("🔄 Procesando frames finales...")
    time.sleep(1.0)  # Dar tiempo extra para procesamiento final
    
    t_cap.stop = True  # Forzar parada si no se ha detenido
    t_cap.join(timeout=2.0)
    t_inf.join(timeout=3.0)

    # Forzar flush final del segmentador si es necesario
    if hasattr(t_inf, 'segmenter') and t_inf.segmenter:
        t_inf.segmenter.flush_close()
        print("✅ Flush final completado")

    df = t_inf.results_df if t_inf.results_df is not None else pd.DataFrame()
    out_csv = args.out or f"{args.exercise}_metrics.csv"
    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"\n✅ CSV guardado en: {out_csv}")
        print(f"📊 Total de repeticiones detectadas: {len(df)}")
        if len(df) > 0:
            print(f"📈 ROM promedio: {df['ROM_m'].mean():.3f} m")
            print(f"⚡ Velocidad máxima promedio: {df['VMAX_mps'].mean():.3f} m/s")
    else:
        print("\n⚠️  No se generaron repeticiones. Prueba ajustando --view lateral (si es de perfil) o --kp_conf 0.20")

    if total:
        dur = total/(fps_cap if fps_cap>0 else 1.0)
        print(f"\n🎥 Duración video: {dur:.2f}s | FPS entrada: {fps_cap:.2f} | FPS efectivo: {t_inf.eff_fps:.2f}")
        print("✔ Métricas y preview sincronizadas al tiempo del video (idx/fps).")
def main():
    args = parse_args()

    # Device por defecto → CPU en macOS por bug MPS Pose
    if args.device == 'auto':
        args.device = 'cpu' if platform.system().lower()=='darwin' else 'cpu'
    if args.device != 'cuda':
        args.half = False

    model = _validate_pose_model(args.model)

    cap0=cv2.VideoCapture(args.video)
    fps_cap = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w0=int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)); h0=int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap0.release()

    rts = RTSettings(
        subject_height_m=args.height, g=9.81,
        v_thr_mps_min=args.v_thr_mps_min, min_rep_time_s=args.min_rep_s, min_conc_time_s=args.min_conc_s,
        sg_win_ms=args.sg_win_ms, sg_poly=args.sg_poly, height_window=121, max_queue=args.max_queue,
        skip_rate=max(0,args.skip_rate), print_every=args.print_every, kp_conf_thr=args.kp_conf,
        slow_mode=args.slow_mode, rom_min_m=args.rom_min_m, view=args.view
    )

    q = queue.Queue(maxsize=max(2, args.max_queue))
    t_cap=FrameSourceThread(args.video, q, max_queue=rts.max_queue, skip_rate=rts.skip_rate)

    disp_q = queue.Queue(maxsize=1) if args.show else None
    t_inf=InferenceAndMetricsThread(
        q, model, rts, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, half=args.half, exercise=args.exercise, dominant=args.dominant,
        save_video=args.save_video, frame_size=(w0,h0), fps_cap=fps_cap,
        draw_pose=args.draw_pose, disp_q=disp_q
    )

    t_cap.start(); t_inf.start()

    # Previsualización temporizada por FPS efectivo (sin acelerarse)
    if args.show:
        cv2.namedWindow("VBT RT + Overlays", cv2.WINDOW_NORMAL)
        start_show = time.time()
        eff_fps = t_inf.eff_fps
        idx_shown = -1
        while t_inf.is_alive():
            try:
                frame_vis = disp_q.get(timeout=0.5)
                idx_shown += 1
                target = start_show + (idx_shown / max(1e-6, eff_fps))
                sleep_t = target - time.time()
                if sleep_t > 0: time.sleep(sleep_t)
                cv2.imshow("VBT RT + Overlays", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except queue.Empty:
                pass
        cv2.destroyAllWindows()
    else:
        while t_inf.is_alive(): time.sleep(0.01)

    t_cap.join(); t_inf.join()

    df = t_inf.results_df if t_inf.results_df is not None else pd.DataFrame()
    out_csv = args.out or f"{args.exercise}_metrics.csv"
    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"\n✅ CSV guardado en: {out_csv}")
        print(f"📊 Total de repeticiones detectadas: {len(df)}")
        if len(df) > 0:
            print(f"📈 ROM promedio: {df['ROM_m'].mean():.3f} m")
            print(f"⚡ Velocidad máxima promedio: {df['VMAX_mps'].mean():.3f} m/s")
    else:
        print("\n⚠️  No se generaron repeticiones. Prueba ajustando --view lateral (si es de perfil) o --kp_conf 0.20")

    if total:
        dur = total/(fps_cap if fps_cap>0 else 1.0)
        print(f"\n🎥 Duración video: {dur:.2f}s | FPS entrada: {fps_cap:.2f} | FPS efectivo: {t_inf.eff_fps:.2f}")
        print("✔ Métricas y preview sincronizadas al tiempo del video (idx/fps).")

if __name__ == "__main__":
    main()
