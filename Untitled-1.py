from tkinter import Canvas, PhotoImage, Tk
import os

# Ensure the image file is in the same directory as this script
file_path = "lion2.jpg"
if not os.path.isfile(file_path):
    print(f"Error: '{file_path}' not found.")
else:
    tk = Tk()
    tk.title("Hello Tkinter")

    canvas = Canvas(tk, width=500, height=500)
    canvas.pack()

    my_image = PhotoImage(file=file_path)
    canvas.create_image(0, 0, anchor='nw', image=my_image)

    tk.mainloop()

