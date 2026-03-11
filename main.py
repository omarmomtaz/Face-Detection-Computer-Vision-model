# pip install opencv-python numpy matplotlib pillow requests

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load image from a local path OR a URL
# ─────────────────────────────────────────────────────────────────────────────
def load_image(source):
    """
    Load an image from:
      - A local file path  (e.g. '/content/photo.jpg')
      - A URL             (e.g. 'https://example.com/photo.jpg')

    Returns a BGR numpy array (same as cv2.imread), or None on failure.
    """
    # ── URL ──────────────────────────────────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):
        try:
            import urllib.request
            print(f"  Downloading image from URL...")
            req = urllib.request.Request(
                source,
                headers={"User-Agent": "Mozilla/5.0"}   # some servers block plain Python
            )
            resp = urllib.request.urlopen(req, timeout=15)
            arr  = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print("  Error: URL responded but the content is not a valid image.")
            return img
        except Exception as e:
            print(f"  Error downloading image: {e}")
            return None

    # ── Local file ───────────────────────────────────────────────────────────
    img = cv2.imread(source)
    if img is None:
        print(f"  Error: Could not load file '{source}'")
        print("  Make sure the path is correct and the file is a valid image.")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# FaceDetector class
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    """Face detector using OpenCV's Haar Cascade"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

    def detect_faces(self, image, detect_eyes=False, detect_smile=False):
        """Detect faces (and optionally eyes / smile) in a BGR image."""
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            face_info = {'bbox': (x, y, w, h), 'eyes': [], 'smile': None}

            if detect_eyes:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                face_info['eyes'] = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]

            if detect_smile:
                roi_gray = gray[y+h//2:y+h, x:x+w]   # lower half only
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0:
                    face_info['smile'] = True

            results.append(face_info)
        return results

    def draw_detections(self, image, detections, draw_eyes=False):
        """Return a copy of the image with bounding boxes drawn."""
        out = image.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 3)
            label = "Face (Smiling)" if det.get('smile') else "Face"
            cv2.putText(out, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if draw_eyes:
                for (ex, ey, ew, eh) in det['eyes']:
                    cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 – detect from a local path OR a URL
# ─────────────────────────────────────────────────────────────────────────────
def detect_from_image(source, detect_eyes=False, detect_smile=False):
    """
    Detect faces from:
      • a local file path  →  '/content/photo.jpg'
      • a URL             →  'https://example.com/photo.jpg'
    """
    print("=" * 70)
    print(f"DETECTING FACES IN IMAGE: {source}")
    print("=" * 70)

    image = load_image(source)
    if image is None:
        return

    detector   = FaceDetector()
    print("\nDetecting faces...")
    detections = detector.detect_faces(image, detect_eyes, detect_smile)
    print(f"Found {len(detections)} face(s)!")

    output_image = detector.draw_detections(image, detections, draw_eyes=detect_eyes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(image,        cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image',       fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected: {len(detections)} Face(s)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('face_detection_result.png', dpi=150, bbox_inches='tight')
    print("\nResult saved to 'face_detection_result.png'")
    plt.show()

    print("\nDetection Details:")
    print("-" * 70)
    for i, det in enumerate(detections, 1):
        x, y, w, h = det['bbox']
        print(f"Face {i}:")
        print(f"  Position : (x={x}, y={y})")
        print(f"  Size     : {w}x{h} pixels")
        if detect_eyes:
            print(f"  Eyes     : {len(det['eyes'])} detected")
        if detect_smile and det.get('smile'):
            print(f"  Smile    : Yes")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 – webcam
# ─────────────────────────────────────────────────────────────────────────────
def detect_from_webcam():
    """
    Detect faces from the laptop webcam in real time.

    Tries several camera indices (0-3) so it works across different
    operating systems and setups. Falls back with a clear error message
    if no camera is found (e.g. running on a remote server / Colab).
    """
    print("=" * 70)
    print("WEBCAM FACE DETECTION")
    print("=" * 70)

    # ── Find a working camera index ──────────────────────────────────────────
    cap   = None
    index = None
    for i in range(4):                          # try indices 0, 1, 2, 3
        test = cv2.VideoCapture(i)
        if test.isOpened():
            ret, _ = test.read()               # confirm we can actually read
            if ret:
                cap   = test
                index = i
                break
        test.release()

    if cap is None or not cap.isOpened():
        print("\n⚠  No webcam found.")
        print("   Possible reasons:")
        print("   • Running on a remote server (Colab, cloud VM) — no physical camera.")
        print("   • Camera driver not installed or camera is in use by another app.")
        print("   • On Colab: use the JavaScript snippet below to capture from your")
        print("     browser camera, save as an image, then use Mode 1 with the path.\n")
        print("─" * 70)
        print("COLAB BROWSER-CAMERA SNIPPET (paste in a new cell):")
        print("─" * 70)
        print("""
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np, cv2

def take_photo(filename='webcam_shot.jpg', quality=0.8):
    js = Javascript('''
      async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = '📸 Capture';
        div.appendChild(capture);
        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();
        await new Promise((r) => capture.onclick = r);
        const canvas = document.createElement('canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();
        return canvas.toDataURL('image/jpeg', quality);
      }
    ''')
    display(js)
    data = eval_js(f'takePhoto({quality})')
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    print(f"Photo saved as '{filename}'")
    return filename

path = take_photo()           # opens your browser camera
detect_from_image(path, detect_eyes=True, detect_smile=True)
""")
        return

    # ── Webcam loop ──────────────────────────────────────────────────────────
    print(f"\n✓ Webcam found at index {index}")
    print("Press 'q' to quit  |  Press 's' to save screenshot")
    print("-" * 70)

    detector    = FaceDetector()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: lost camera feed.")
            break

        detections   = detector.detect_faces(frame, detect_eyes=True)
        output_frame = detector.draw_detections(frame, detections, draw_eyes=True)

        cv2.putText(output_frame, f"Faces: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, "q = quit   s = screenshot", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Face Detection – Webcam', output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting webcam...")
            break
        elif key == ord('s'):
            filename = f'webcam_capture_{frame_count}.jpg'
            cv2.imwrite(filename, output_frame)
            print(f"Screenshot saved: {filename}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 – test image generator
# ─────────────────────────────────────────────────────────────────────────────
def create_test_image():
    """Create a simple synthetic face and return its path."""
    print("\nCreating test image...")
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.circle(img, (200, 200), 100, (200, 150, 100), -1)   # head
    cv2.circle(img, (170, 180),  15, ( 50,  50,  50), -1)   # left eye
    cv2.circle(img, (230, 180),  15, ( 50,  50,  50), -1)   # right eye
    cv2.ellipse(img, (200, 230), (40, 20), 0, 0, 180, (50, 50, 50), 3)  # smile
    cv2.imwrite('test_face.jpg', img)
    print("Test image saved as 'test_face.jpg'")
    return 'test_face.jpg'


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 – batch detection
# ─────────────────────────────────────────────────────────────────────────────
def batch_detect_faces(path_input):
    """
    Detect faces in:
      • A folder path  →  every JPG/PNG/BMP inside is processed
      • A single file  →  treated as a one-image batch (common mistake — handled gracefully)
    """
    print("=" * 70)

    # ── Smart path handling ──────────────────────────────────────────────────
    # FIX: user often passes a file path to the batch mode by mistake.
    # If it's a file, just process it as a single image instead of failing.
    if os.path.isfile(path_input):
        print(f"Note: '{path_input}' is a file, not a folder.")
        print("Running single-image detection instead...")
        print("=" * 70)
        detect_from_image(path_input, detect_eyes=True, detect_smile=True)
        return

    if not os.path.isdir(path_input):
        print(f"Error: '{path_input}' is not a valid folder or file path.")
        return

    print(f"BATCH FACE DETECTION IN FOLDER: {path_input}")
    print("=" * 70)

    exts        = ['*.jpg', '*.jpeg', '*.png', '*.bmp',
                   '*.JPG', '*.JPEG', '*.PNG', '*.BMP']   # case-insensitive on Linux too
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(path_input, ext)))
    image_files = list(set(image_files))    # deduplicate

    if not image_files:
        print(f"No images found in '{path_input}'")
        print("Supported formats: JPG, JPEG, PNG, BMP")
        return

    print(f"\nFound {len(image_files)} image(s)")
    print("-" * 70)

    detector   = FaceDetector()
    total      = 0
    results    = []

    for img_path in sorted(image_files):
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ✗ {os.path.basename(img_path)}: could not read file")
                continue
            detections = detector.detect_faces(image)
            count      = len(detections)
            total     += count
            results.append((img_path, count))
            print(f"  ✓ {os.path.basename(img_path)}: {count} face(s)")
        except Exception as e:
            print(f"  ✗ {os.path.basename(img_path)}: {e}")

    print("-" * 70)
    print(f"Total images processed : {len(results)}")
    print(f"Total faces detected   : {total}")

    # Optional: save annotated copies of every image
    save = input("\nSave annotated copies of all images? (y/n): ").strip().lower()
    if save == 'y':
        out_dir = os.path.join(path_input, 'annotated')
        os.makedirs(out_dir, exist_ok=True)
        for img_path, _ in results:
            image      = cv2.imread(img_path)
            detections = detector.detect_faces(image, detect_eyes=True)
            annotated  = detector.draw_detections(image, detections, draw_eyes=True)
            out_path   = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
        print(f"Annotated images saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  FACE DETECTION SYSTEM  —  Omar Momtaz")
    print("=" * 70)
    print("  1 · Detect from image file OR URL")
    print("  2 · Detect from laptop webcam (real-time)")
    print("  3 · Generate test image and detect")
    print("  4 · Batch detect across a folder")
    print("  ↵ · Quick demo (auto test image)")
    print("=" * 70)

    choice = input("\nEnter choice (1-4) or press Enter: ").strip()

    if choice == "1":
        source = input(
            "\nEnter a local file path OR a full image URL:\n"
            "  e.g.  /content/photo.jpg\n"
            "  e.g.  https://example.com/photo.jpg\n"
            "> "
        ).strip()
        detect_from_image(source, detect_eyes=True, detect_smile=True)

    elif choice == "2":
        detect_from_webcam()

    elif choice == "3":
        test_img = create_test_image()
        detect_from_image(test_img, detect_eyes=True)

    elif choice == "4":
        folder = input(
            "\nEnter folder path (or a single image path — both work):\n> "
        ).strip()
        batch_detect_faces(folder)

    else:
        print("\nRunning quick demo...")
        test_img = create_test_image()
        detect_from_image(test_img, detect_eyes=True)

    print("\n" + "=" * 70)
    print("✓ Face detection complete!")
    print("=" * 70)
    print("\nQuick integration example:")
    print("  detector = FaceDetector()")
    print("  image    = load_image('https://example.com/photo.jpg')  # or local path")
    print("  faces    = detector.detect_faces(image)")
    print("  result   = detector.draw_detections(image, faces)")


if __name__ == "__main__":
    main()