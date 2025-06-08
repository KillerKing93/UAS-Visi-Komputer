import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import pytube
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
import gdown
import json
import hashlib
import shutil

# --- Konfigurasi dan Manajemen File ---
CONFIG_DIR = "config"
MODEL_DIR = "models"
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
USERS_FILE = os.path.join(CONFIG_DIR, "users.json")

# PERBAIKAN YAML: Menambahkan constructor untuk menangani tag pathlib.PosixPath
def posix_path_constructor(loader, node):
    """Mengonstruksi path sebagai string, bukan sebagai objek PosixPath."""
    seq = loader.construct_sequence(node)
    return os.path.join(*[str(s) for s in seq])

yaml.add_constructor('!python/object/apply:pathlib.PosixPath', posix_path_constructor, Loader=yaml.UnsafeLoader)

def hash_password(password):
    """Mengenkripsi password menggunakan SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Memuat data pengguna dari file JSON."""
    if not os.path.exists(USERS_FILE):
        default_users = {"admin": {"password": hash_password("admin123"), "role": "admin"}}
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f)
        return default_users
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Menyimpan data pengguna ke file JSON."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def load_config():
    """Memuat konfigurasi path model dari file JSON."""
    if not os.path.exists(CONFIG_FILE):
        return {"weights_path": None, "yaml_path": None}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(weights_path, yaml_path):
    """Menyimpan konfigurasi path model ke file JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"weights_path": weights_path, "yaml_path": yaml_path}, f, indent=4)

# --- Fungsi Inti Aplikasi ---

def format_timestamp(seconds):
    """Mengubah detik menjadi format MM:SS."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def download_video(video_url, progress):
    """Mengunduh video dari YouTube atau Google Drive."""
    progress(0, desc="Mengunduh Video...")
    filename = "input_video.mp4"
    if os.path.exists(filename):
        os.remove(filename)

    if "youtube.com" in video_url or "youtu.be" in video_url:
        yt = pytube.YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream is None:
            raise Exception("Tidak ada stream video MP4 yang tersedia.")
        return stream.download(filename=filename)
    elif "drive.google.com" in video_url:
        return gdown.download(video_url, filename, quiet=False, fuzzy=True)
    else:
        raise Exception("URL tidak didukung. Harap masukkan URL YouTube atau Google Drive.")

def cleanup_temp_files():
    """Menghapus file video dan chart sementara."""
    files_to_delete = ["input_video.mp4", "output_video.mp4", "stats_chart.png"]
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)

def process_video(video_url, progress=gr.Progress()):
    """
    Fungsi utama untuk memproses video menggunakan model yang telah dikonfigurasi.
    """
    # FITUR BARU: Hapus file lama sebelum memulai proses baru
    cleanup_temp_files()
    
    config = load_config()
    weights_path = config.get("weights_path")
    yaml_path = config.get("yaml_path")

    if not all([video_url, weights_path, yaml_path]):
        return None, None, "---", "### Kesalahan: Admin harus mengonfigurasi model di tab 'Manajemen Model'."

    if not os.path.exists(weights_path) or not os.path.exists(yaml_path):
        return None, None, "---", f"### Kesalahan: File model tidak ditemukan. Harap unggah model yang valid di tab 'Manajemen Model'. Path saat ini: `{weights_path}`"

    try:
        input_video_path = download_video(video_url, progress)
        if not input_video_path or not os.path.exists(input_video_path):
             raise Exception(f"Gagal mengunduh video dari URL: {video_url}")

        progress(0.1, desc="Memuat Model...")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader) 
            if 'names' in data:
                class_names = data['names']
            elif 'dataset_config_content' in data and 'names' in data['dataset_config_content']:
                class_names = data['dataset_config_content']['names']
            else:
                raise Exception("File YAML tidak mengandung daftar 'names' yang diperlukan.")

        model = YOLO(weights_path)
        
        output_video_path = "output_video.mp4"
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0: raise Exception("Video tidak dapat dibaca.")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        detection_stats, key_moments = defaultdict(int), defaultdict(list)
        last_moment_time = defaultdict(lambda: -5)

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            progress(0.1 + (0.7 * (frame_count / total_frames)), desc=f"Mendeteksi... Frame {frame_count+1}/{total_frames}")

            results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')
            annotated_frame = results[0].plot()

            for det in results[0].boxes.data.cpu().numpy():
                class_name = class_names[int(det[5])]
                detection_stats[class_name] += 1
                current_time = (frame_count + 1) / fps
                if current_time - last_moment_time[class_name] > 2:
                    key_moments[class_name].append(current_time)
                    last_moment_time[class_name] = current_time
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        progress(0.85, desc="Menghasilkan Statistik...")
        stats_chart_path = None
        if detection_stats:
            df_stats = pd.DataFrame(list(detection_stats.items()), columns=['Objek', 'Total Deteksi']).sort_values('Total Deteksi', ascending=False)
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(df_stats['Objek'], df_stats['Total Deteksi'], color='skyblue')
            ax.set_ylabel('Total Deteksi'); ax.set_title('Statistik Deteksi Objek')
            plt.xticks(rotation=45, ha='right'); ax.bar_label(bars, fmt='%d')
            plt.tight_layout()
            stats_chart_path = "stats_chart.png"
            fig.savefig(stats_chart_path); plt.close(fig)

        progress(0.95, desc="Menyusun Hasil Akhir...")
        key_moments_md = "### ⏱️ Momen Kunci\n\n" + "\n\n".join([f"**{label}:** {', '.join([f'`{format_timestamp(t)}`' for t in timestamps[:15]])}{' ...' if len(timestamps) > 15 else ''}" for label, timestamps in sorted(key_moments.items(), key=lambda item: len(item[1]), reverse=True)]) if key_moments else "Tidak ada momen kunci."

        unique_detections = {label: len(timestamps) for label, timestamps in key_moments.items()}
        num_people = unique_detections.get('Person', 0)
        hardhat_count = unique_detections.get('Hardhat', 0)
        no_hardhat_count = unique_detections.get('NO-Hardhat', 0)
        total_hardhat_status = hardhat_count + no_hardhat_count
        safety_compliance = (hardhat_count / total_hardhat_status * 100) if total_hardhat_status > 0 else 0
        
        summary_text = f"""### 📊 Ringkasan Analisis
- **Total Durasi:** `{format_timestamp(total_frames / fps)}`
- **Total Orang:** `{num_people}`
- **Kepatuhan Helm:** `{safety_compliance:.2f}%` (`{hardhat_count}` dengan helm, `{no_hardhat_count}` tanpa helm)"""

        if os.path.exists(input_video_path): os.remove(input_video_path)

        progress(1, desc="Selesai!")
        return output_video_path, stats_chart_path, summary_text, key_moments_md
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return None, None, "---", f"### ❌ Terjadi Kesalahan:\n`{e}`"

# --- Antarmuka Gradio ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    auth_state = gr.State(value={"logged_in": False, "username": None, "role": None})
    
    with gr.Column(visible=True) as login_ui:
        gr.Markdown("# 🔑 Login Dasbor")
        with gr.Row():
            username_login = gr.Textbox(label="Username", placeholder="admin")
            password_login = gr.Textbox(label="Password", type="password", placeholder="admin123")
        login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Markdown()

    with gr.Column(visible=False) as main_app_ui:
        with gr.Row():
            gr.Markdown("# 👷 Dasbor Deteksi Keselamatan Konstruksi")
            current_user_display = gr.Markdown()
            logout_btn = gr.Button("Logout")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("🔎 Deteksi Video"):
                with gr.Row():
                    with gr.Column():
                        video_url = gr.Textbox(label="URL Video (YouTube atau Google Drive)", placeholder="Tempel URL di sini...")
                        analyze_btn = gr.Button("Analisis Video", variant="primary")
                    with gr.Column():
                        output_video = gr.Video(label="Video dengan Anotasi")

            with gr.TabItem("📊 Hasil Analisis"):
                summary_text = gr.Markdown(label="Ringkasan")
                with gr.Row():
                    stats_chart = gr.Image(label="Grafik Deteksi Objek")
                    key_moments_md = gr.Markdown(label="Momen Kunci")
            
            with gr.TabItem("⚙️ Manajemen Model", visible=False, id="admin_model_tab") as admin_model_panel:
                gr.Markdown("Unggah file bobot dan metadata baru untuk menggantikan model yang ada.")
                with gr.Row():
                    new_weights_file = gr.File(label="Unggah Bobot Model Baru (.pt)", file_types=['.pt'])
                    new_yaml_file = gr.File(label="Unggah Metadata Baru (.yaml)", file_types=['.yaml'])
                save_model_btn = gr.Button("Simpan & Ganti Model")
                model_status = gr.Markdown()
                
            with gr.TabItem("👥 Manajemen Pengguna", visible=False, id="admin_user_tab") as admin_user_panel:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Tambah Pengguna Baru")
                        new_username = gr.Textbox(label="Username Baru")
                        new_password = gr.Textbox(label="Password Baru", type="password")
                        new_role = gr.Radio(label="Peran", choices=["operator", "admin"], value="operator")
                        add_user_btn = gr.Button("Tambah Pengguna")
                        user_add_status = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### Edit atau Hapus Pengguna")
                        user_select_for_edit = gr.Dropdown(label="Pilih Pengguna", choices=list(load_users().keys()))
                        edit_username = gr.Textbox(label="Username Baru (biarkan kosong jika tidak berubah)")
                        edit_password = gr.Textbox(label="Password Baru (biarkan kosong jika tidak berubah)", type="password")
                        edit_role = gr.Radio(label="Peran Baru", choices=["operator", "admin"])
                        with gr.Row():
                            update_user_btn = gr.Button("Perbarui Pengguna")
                            delete_user_btn = gr.Button("Hapus Pengguna", variant="stop")
                        user_edit_status = gr.Markdown()
    
    def login_user(username, password, current_state):
        users = load_users()
        hashed_pass = hash_password(password)
        if username in users and users[username]["password"] == hashed_pass:
            role = users[username]["role"]
            new_state = {"logged_in": True, "username": username, "role": role}
            is_admin = (role == "admin")
            return (
                new_state,
                gr.update(visible=False), gr.update(visible=True),
                gr.update(value=f"Login sebagai: **{username}** ({role})"),
                gr.update(visible=is_admin), gr.update(visible=is_admin)
            )
        else:
            return current_state, gr.update(), gr.update(), gr.update(value="<p style='color:red;'>Username atau password salah.</p>"), gr.update(), gr.update()
            
    def logout_user():
        # FITUR BARU: Hapus file sementara saat logout
        cleanup_temp_files()
        return (
            {"logged_in": False, "username": None, "role": None}, 
            gr.update(visible=True), gr.update(visible=False), "", 
            gr.update(visible=False), gr.update(visible=False),
            # Hapus output saat logout
            None, None, "", ""
        )

    def save_new_model(weights_file, yaml_file, current_state):
        if current_state["role"] != "admin": return "Akses ditolak."
        if weights_file is None or yaml_file is None: return "Harap unggah kedua file."
            
        config = load_config()
        if config.get("weights_path") and os.path.exists(config["weights_path"]):
            os.remove(config["weights_path"])
        if config.get("yaml_path") and os.path.exists(config["yaml_path"]):
            os.remove(config["yaml_path"])

        weights_dest = os.path.join(MODEL_DIR, os.path.basename(weights_file.name))
        yaml_dest = os.path.join(MODEL_DIR, os.path.basename(yaml_file.name))
        shutil.move(weights_file.name, weights_dest)
        shutil.move(yaml_file.name, yaml_dest)
        
        save_config(weights_dest, yaml_dest)
        return f"✅ Model berhasil diperbarui. Menggunakan `{os.path.basename(weights_dest)}`."
        
    def add_new_user(username, password, role, current_state):
        if current_state["role"] != "admin": return "Akses ditolak."
        if not username or not password: return "Username dan password tidak boleh kosong."
            
        users = load_users()
        if username in users: return f"❌ Username '{username}' sudah ada."
            
        users[username] = {"password": hash_password(password), "role": role}
        save_users(users)
        return f"✅ Pengguna '{username}' dengan peran '{role}' berhasil ditambahkan.", gr.update(choices=list(users.keys()))

    def update_user_details(selected, new_user, new_pass, new_role, current_state):
        if current_state["role"] != "admin": return "Akses ditolak.", gr.update()
        if not selected: return "Pilih pengguna untuk diperbarui.", gr.update()
        
        users = load_users()
        user_data = users.pop(selected) 
        
        final_username = new_user if new_user else selected
        if final_username != selected and final_username in users:
            users[selected] = user_data 
            return f"❌ Username '{final_username}' sudah ada.", gr.update(choices=list(users.keys()))

        user_data["password"] = hash_password(new_pass) if new_pass else user_data["password"]
        user_data["role"] = new_role
        
        users[final_username] = user_data
        save_users(users)
        
        return f"✅ Pengguna '{selected}' berhasil diperbarui menjadi '{final_username}'.", gr.update(choices=list(users.keys()), value=None)

    def delete_user_by_name(username_to_delete, current_state):
        if current_state["role"] != "admin": return "Akses ditolak.", gr.update()
        if not username_to_delete: return "Pilih pengguna untuk dihapus.", gr.update()
        
        users = load_users()
        admin_count = sum(1 for u in users.values() if u['role'] == 'admin')
        
        if users[username_to_delete]['role'] == 'admin' and admin_count <= 1:
            return "❌ Tidak dapat menghapus admin terakhir.", gr.update()

        del users[username_to_delete]
        save_users(users)
        return f"✅ Pengguna '{username_to_delete}' berhasil dihapus.", gr.update(choices=list(users.keys()), value=None)
        
    def refresh_user_list():
        return gr.update(choices=list(load_users().keys()))

    login_btn.click(login_user, inputs=[username_login, password_login, auth_state], outputs=[auth_state, login_ui, main_app_ui, current_user_display, admin_model_panel, admin_user_panel])
    # PERBAIKAN: Menambahkan output untuk membersihkan UI saat logout
    logout_btn.click(logout_user, inputs=[], outputs=[auth_state, login_ui, main_app_ui, current_user_display, admin_model_panel, admin_user_panel, output_video, stats_chart, summary_text, key_moments_md])
    analyze_btn.click(process_video, inputs=[video_url], outputs=[output_video, stats_chart, summary_text, key_moments_md])
    
    save_model_btn.click(save_new_model, inputs=[new_weights_file, new_yaml_file, auth_state], outputs=[model_status])
    add_user_btn.click(add_new_user, inputs=[new_username, new_password, new_role, auth_state], outputs=[user_add_status, user_select_for_edit])
    update_user_btn.click(update_user_details, inputs=[user_select_for_edit, edit_username, edit_password, edit_role, auth_state], outputs=[user_edit_status, user_select_for_edit])
    delete_user_btn.click(delete_user_by_name, inputs=[user_select_for_edit, auth_state], outputs=[user_edit_status, user_select_for_edit])
    
    admin_user_panel.select(refresh_user_list, [], [user_select_for_edit])
    
if __name__ == "__main__":
    demo.launch(debug=True)
