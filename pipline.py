import matplotlib
import os
import shutil
import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ollama_file import analyze_image_for_text


class InteractiveLabelingTool:
    def __init__(self, image_path, predictions, parent_window=None, text_display=None):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.predictions = predictions
        self.boxes = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        self.current_box = None
        self.drawing = False
        self.selected_box = None
        self.detected_texts = []

        # Create GUI window
        self.window = tk.Toplevel(parent_window) if parent_window else tk.Tk()
        self.window.title("Bounding Box Correction Tool")

        # Convert OpenCV image to PhotoImage
        self.photo = self.convert_cv2_to_photoimage(self.image)

        # Create main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas
        self.canvas = tk.Canvas(main_frame, width=self.photo.width(), height=self.photo.height())
        self.canvas.pack(side=tk.TOP)

        # Create text display if not provided
        self.text_display = text_display if text_display else tk.Text(main_frame, height=10, width=50)
        if not text_display:
            self.text_display.pack(side=tk.BOTTOM, padx=5, pady=5)

        # Create a frame for buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # Remove box button
        remove_box_button = tk.Button(button_frame, text="Remove Box", command=self.remove_box)
        remove_box_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Remove all boxes button
        remove_all_boxes_button = tk.Button(button_frame, text="Remove All Boxes", command=self.remove_all_boxes)
        remove_all_boxes_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Save annotations button
        save_button = tk.Button(button_frame, text="Save Annotations", command=self.save_annotations)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Detect Text button
        detect_text_button = tk.Button(button_frame, text="Detect Text", command=self.detect_text_in_boxes)
        detect_text_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Display the image on the canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Draw initial predictions
        self.draw_boxes()

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # If not using parent window, start mainloop
        if not parent_window:
            self.window.mainloop()
    def convert_cv2_to_photoimage(self, cv2_image):
        """Convert OpenCV image to Tkinter PhotoImage."""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(image=pil_image)

    def draw_boxes(self):
        """Draw bounding boxes on the canvas."""
        self.canvas.delete("box")  # Clear existing boxes
        for i, box in enumerate(self.boxes):
            # Ensure integer coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Draw box with unique tag
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="red", width=2, tags=(f"box{i}", "box"))

    def on_click(self, event):
        """Handle mouse click events."""
        # Check if clicking on an existing box
        clicked_box = self.canvas.find_withtag("box")
        clicked_box = [box for box in clicked_box if self.is_point_in_box(
            event.x, event.y, box)]

        if clicked_box:
            # If a box is clicked, prepare for potential removal
            self.selected_box = clicked_box[0]
        else:
            # Start drawing a new box
            self.current_box = [event.x, event.y, event.x, event.y]
            # Now we can use append since self.boxes is a list
            self.boxes.append(self.current_box)
            self.draw_boxes()

    def on_drag(self, event):
        """Update the current bounding box during drag."""
        if self.current_box:
            # Update the last box (most recently added)
            self.boxes[-1][2] = event.x
            self.boxes[-1][3] = event.y
            self.draw_boxes()

    def on_release(self, event):
        """Finish drawing or selecting the current bounding box."""
        if self.current_box:
            self.current_box[2] = event.x
            self.current_box[3] = event.y
            self.current_box = None

    def is_point_in_box(self, x, y, box_id):
        """Check if a point is inside a given box."""
        box_coords = self.canvas.coords(box_id)
        return (box_coords[0] <= x <= box_coords[2] and
                box_coords[1] <= y <= box_coords[3])

    def remove_box(self):
        """Remove the selected bounding box."""
        if self.selected_box:
            # Find the index of the box to remove
            box_tags = self.canvas.gettags(self.selected_box)
            box_index = int(box_tags[0][3:])  # Extract index from tag

            # Remove the box from the list
            del self.boxes[box_index]

            # Redraw boxes
            self.draw_boxes()

            # Reset selected box
            self.selected_box = None

    def remove_all_boxes(self):
        """Remove all bounding boxes."""
        self.boxes.clear()  # Clear the list of boxes
        self.draw_boxes()  # Redraw the canvas without any boxes

    
    
    def detect_text_in_boxes(self):
        """Detect text in the current bounding boxes."""
        # Clear previous text
        self.text_display.delete(1.0, tk.END)
        
        # Crop and detect text in the current boxes
        save_dir = os.path.join(r"D:\new_pipe\dataset\cropped", os.path.splitext(os.path.basename(self.image_path))[0])
        
        # Detect texts using the current boxes
        self.detected_texts = crop_and_save_bounding_boxes(
            self.image, 
            [{"boxes": self.boxes}], 
            save_dir
        )

        # Display detected texts
        self.text_display.insert(tk.END, "Detected Texts:\n")
        for i, text in enumerate(self.detected_texts, 1):
            self.text_display.insert(tk.END, f"{i}. {text}\n")

    def save_annotations(self):
        """Save annotations and close the window."""
        print("Annotations saved:", self.boxes)
        
        # Optionally, save detected texts
        if self.detected_texts:
            print("Detected Texts:", self.detected_texts)
        
        self.window.destroy()



def save_yolo_annotations(image_path, boxes):
    """Save annotations in YOLO format."""
    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Prepare label filename and directory
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Prepare labels directory
    labels_dir = r"D:\new_pipe\dataset\labels\train"
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, base_filename + '.txt')

    # Write annotations in YOLO format
    with open(label_path, 'w') as f:
        for box in boxes:
            # Convert to YOLO format: <class> <x_center> <y_center> <width> <height>
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = abs(x2 - x1) / width
            bbox_height = abs(y2 - y1) / height

            # Assuming class 0, modify if you have multiple classes
            f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")


def yolo_predict(image_path, model):
    """Run YOLO prediction on an image and return bounding boxes."""
    results = model(image_path)

    # Ensure results are non-empty and contain bounding box data
    if not results or not hasattr(results[0], 'boxes'):
        raise ValueError("No bounding boxes found in YOLO results.")

    # Extract bounding boxes
    # Bounding box coordinates in [x1, y1, x2, y2]
    boxes = results[0].boxes.xyxy.cpu().numpy()

    return boxes

def crop_and_save_bounding_boxes(image, results, save_dir="save"):
    """
    Crops and saves bounding box regions from the image based on YOLO results.

    Parameters:
        image (numpy.ndarray): The original image.
        results: List of dictionaries containing bounding boxes.
        save_dir (str): Directory to save cropped images. Defaults to "save".

    Returns:
        list: Detected texts from the image
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    detected_texts = []
    
    # If results is a list of dictionaries with 'boxes' key
    if isinstance(results, list) and results and 'boxes' in results[0]:
        boxes = results[0]['boxes']
    # If results is a numpy array of boxes
    elif isinstance(results, np.ndarray):
        boxes = results
    else:
        print("Unexpected results format")
        return detected_texts

    # Convert boxes to a list if it's a numpy array
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    # Sort boxes by y1 (top) first, then by x1 (left to right)
    box_data = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        box_data.append({
            'index': i,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        })

    sorted_boxes = sorted(box_data, key=lambda x: (x['y1'], x['x1']))

    print(f"Total number of detected text regions: {len(sorted_boxes)}")

    for seq_num, box_info in enumerate(sorted_boxes, 1):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']
        
        
        print("Original image shape:", image.shape)
        # Crop the region
        cropped_region = image[y1:y2, x1:x2]

        # check if the cropped_region is not empty
        print("Cropped Region:",cropped_region)
        
        if cropped_region.size == 0:
            print("Cropped region is empty!")
            return
        
        # Save the cropped region with sequential naming
        save_path = os.path.join(save_dir, f"text_region00_{seq_num:03d}.png")
        cv2.imwrite(save_path, cropped_region)
        print(f"Saved cropped region to: {save_path}")

        # Analyze text in the cropped region
        try:
            text = analyze_image_for_text(save_path)
            detected_texts.append(text)
            print(f"Detected text in region {seq_num}: {text}")
        except Exception as e:
            print(f"Text detection error in region {seq_num}: {e}")
            detected_texts.append("No text detected")

    print("Detected texts:", detected_texts)
    return detected_texts


def iterative_pipeline(input_path, model_path, iterations=5):
    # Ensure dataset directories exist
    dataset_dir = r'/home/omen/Documents/Shiv/dataset'
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)

    # Create or update data.yaml
    data_yaml_path = r"D:\new_pipe\dataset\data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(f"""
train: {r"/home/omen/Documents/Shiv/dataset/images/train"}
val: {r"/home/omen/Documents/Shiv/dataset/images/val"}
nc: 1
names: ['word']""")

    # Load the initial model
    model = YOLO(model_path)

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")

        # Get list of images
        images = [os.path.join(input_path, img) for img in os.listdir(
            input_path) if img.endswith((".png", ".jpg"))]

        for img_path in images:
            print(f"Processing: {img_path}")

            # Predict bounding boxes
            try:
                results = model(img_path)
                print("Result type:::",type(results), results)

                # Ensure results are not empt
                if len(results) == 0 or len(results[0].boxes) == 0:
                    print(f"No bounding boxes detected in {img_path}")
                    continue

                # Extract bounding box coordinates
                predictions = results[0].boxes.xyxy.cpu().numpy()

            except Exception as e:
                print(f"Error predicting bounding boxes: {e}")
                continue


            labeling_tool = InteractiveLabelingTool(
                image_path=img_path, 
                predictions=predictions
            )

            # Validate bounding boxes
            if not labeling_tool.boxes or len(labeling_tool.boxes) == 0:
                print(f"No bounding boxes created for {img_path}")
                continue

            # Save updated bounding boxes
            save_yolo_annotations(img_path, labeling_tool.boxes)

            # Save cropped bounding box regions after corrections
            image = cv2.imread(img_path)
            save_dir = os.path.join("/home/omen/Documents/Shiv/dataset/cropped", os.path.splitext(
                os.path.basename(img_path))[0])
            detected_texts = crop_and_save_bounding_boxes(
                image, [{"boxes": labeling_tool.boxes}], save_dir)

        # Fine-tune and save the model
        try:
            results = model.train(
                data=data_yaml_path,
                epochs=20,
                imgsz=640,
                patience=3,
                batch=8,
                device='cpu'
            )

            # Save the updated model
            updated_model_path = "best.pt"
            model.save(updated_model_path)

            # For next iteration, load the updated model
            model = YOLO(updated_model_path)
        except Exception as e:
            print(f"Error during model training: {e}")

    print("Pipeline completed. Model improved iteratively.")

# Example usage
input_path = "/home/omen/Documents/Shiv/dataset/train"
model_path = "best.pt"
iterations = 1
iterative_pipeline(input_path, model_path, iterations)
