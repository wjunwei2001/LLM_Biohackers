import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import AutoProcessor, AutoModelForCausalLM

# from huggingface_hub import login
# login()

processor_path = "./git-base-pokemon/git-base-pokemon/checkpoint-740/"
processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)

def upload_button_click():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize the image to fit in the window
        image = ImageTk.PhotoImage(image)
        uploaded_image_label.configure(image=image)
        uploaded_image_label.image = image
        print(f"Uploaded image: {file_path}")

def on_button_click():
    generated_text = generate_text()
    result_label.config(text=generated_text)

def generate_text(inputs):
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    input_ids = processor(pixel_values=pixel_values, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]

    return generated_text


root = tk.Tk()
label = tk.Label(root, text="LLM Image-to-Text")
label.pack()
root.geometry("532x632")

upload_button = tk.Button(root, text="Upload photo!", command=upload_button_click)
upload_button.pack()

uploaded_image_label = ttk.Label(root)
uploaded_image_label.pack()

generate_button = tk.Button(root, text="Generate Text!", command=on_button_click)
generate_button.pack()

result_label = ttk.Label(root, wraplength=400, justify='center')
result_label.pack()

root.mainloop()