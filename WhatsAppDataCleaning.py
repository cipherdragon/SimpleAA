import re
import tkinter as tk
from tkinter import filedialog

def select_and_read_file():
    try:
        not_Deelaka_texts_List = []
        file_path1 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        with open(file_path1, 'r', encoding='utf-8') as file:
            for line in file:
                # Regular expression pattern
                pattern = re.compile(r"\[.*\] ([^:]+): (.*)")
                matches = pattern.findall(line.strip())
                # filtering process
                for match in matches:
                    sender = match[0]
                    message = match[1]
                    if sender != "Deelaka Dias":
                        final_sender = sender
                        not_Deelaka_texts_List.append(message)

        not_Deelaka_texts_List = [s for s in not_Deelaka_texts_List if not s.startswith('\u200E')]
        file_path2 = f"{final_sender} - Cleaned_Chat.txt"
        with open(file_path2, 'w', encoding='utf-8') as file:
            for item in not_Deelaka_texts_List:
                file.write(f"{item}\n")
                print("Data Cleaned")

    except FileNotFoundError:
        print("You need to select a file")
        print("Terminated")

# Create the main application window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Call the function to select and read a file
selected_sender = select_and_read_file()

# Start the Tkinter event loop
root.mainloop()