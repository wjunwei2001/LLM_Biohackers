import tkinter as tk
from tkinter import ttk
from query_GAN import load_model, generate_images

def on_dropdown_selected(event):
    selected_value = dropdown_var.get()
    print("Selected:", selected_value)

def on_button_click():
    print("Button clicked!")
    model = load_model()
    generate_images(16, model)


root = tk.Tk()
label = tk.Label(root, text="Ultrasound Generative AI")
label.pack(side="top", pady=10)
root.geometry("532x632")

# Create sections
# section1 = tk.Label(root, text="Section 1", bg="red", fg="white")
# section2 = tk.Label(root, text="Section 2", bg="blue", fg="white")
# section3 = tk.Label(root, text="Section 3", bg="green", fg="white")

# # Add sections using pack layout manager
# section1.pack(fill=tk.BOTH, expand=True)  # Expands both horizontally and vertically
# section2.pack(fill=tk.X)  # Expands horizontally
# section3.pack(fill=tk.Y)  

options = ["Malignant Breast Cancer"]
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(root, textvariable=dropdown_var, values=options)
dropdown_var.set(options[0])
dropdown.bind("<<ComboboxSelected>>", on_dropdown_selected)
dropdown.pack(side="top", pady=10)

button = tk.Button(root, text="Click Me!", command=on_button_click)

# Pack the button (or you can use grid or place layout manager)
button.pack(side="top", pady=20)

root.mainloop()