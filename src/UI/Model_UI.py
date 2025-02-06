from tkinter import *
from tkinter.ttk import *
import time
import threading
import requests
import pandas as pd
from tkinter import filedialog
from tkcalendar import DateEntry
from io import StringIO

# Model UI is a class that creates a window for the user to input the model parameters (start and end dates for training and testing) and outputs csv file with the model output
class ModelUI:
    def __init__(self, master):
        self.master = master
        master.title("Dream11 Model Parameters")
        master.geometry("600x400+{}+{}".format(int(master.winfo_screenwidth() * 0.3), int(master.winfo_screenheight() * 0.3)))


        self.label_training = Label(master, text="Training Parameters", font=("Helvetica", 14))
        self.label_training.grid(row=0, column=0, padx=60, pady=40)

        self.label_testing = Label(master, text="Testing Parameters", font=("Helvetica", 14))
        self.label_testing.grid(row=0, column=1, padx=60, pady=40)

        self.label_start_date_training = Label(master, text="Start Date (Training):")
        self.label_start_date_training.grid(row=1, column=0, padx=10, pady=5)

        self.start_date_training = DateEntry(master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern="YYYY-MM-DD")
        self.start_date_training.grid(row=2, column=0, padx=10, pady=5)

        self.label_end_date_training = Label(master, text="End Date (Training):")
        self.label_end_date_training.grid(row=3, column=0, padx=10, pady=5)

        self.end_date_training = DateEntry(master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern="YYYY-MM-DD")
        self.end_date_training.grid(row=4, column=0, padx=10, pady=5)

        self.label_start_date_testing = Label(master, text="Start Date (Testing):")
        self.label_start_date_testing.grid(row=1, column=1, padx=10, pady=5)

        self.start_date_testing = DateEntry(master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern="YYYY-MM-DD")
        self.start_date_testing.grid(row=2, column=1, padx=10, pady=5)

        self.label_end_date_testing = Label(master, text="End Date (Testing):")
        self.label_end_date_testing.grid(row=3, column=1, padx=10, pady=5)

        self.end_date_testing = DateEntry(master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern="YYYY-MM-DD")
        self.end_date_testing.grid(row=4, column=1, padx=10, pady=5)

        self.submit_button = Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.submit_label = Label(master, text="")
        self.submit_label.grid(row=8, column=0, columnspan=2, pady=10)

    def submit(self):
        start_date_training = self.start_date_training.get()
        end_date_training = self.end_date_training.get()
        start_date_testing = self.start_date_testing.get()
        end_date_testing = self.end_date_testing.get()
        status = [0, '']
        api_thread = threading.Thread(target=self.call_api, args=(start_date_training, end_date_training, start_date_testing, end_date_testing, status), name="API call")
        api_thread.start()

        self.submit_button['state'] = 'disabled'
        self.submit_label['text'] = ""
        self.progress = Progressbar(self.master, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress.grid(row=7, column=0, columnspan=2, pady=10)
        self.progress['value'] = 0
        self.master.update_idletasks()
        time_period = 30
        for i in range(500):
            if api_thread.is_alive() == False:
                break

            self.progress['value'] += 100/500
            self.master.update_idletasks()
            time.sleep(time_period/500)

            if i > 0.90*500 and api_thread.is_alive():
                self.submit_label['text'] = "Processing is taking longer than expected. Please wait..."
                self.master.update_idletasks()
                api_thread.join()
                break

        if status[0] == -1:
            self.submit_label['text'] = "API call failed"
        elif status[0] == 1:
            self.submit_label['text'] = "API call successful"
            # make a download button to download the csv file
            download_button = Button(self.master, text="Download Model Output", command=lambda: self.save_to_csv(status[1]))
            download_button.grid(row=9, column=0, columnspan=2, pady=10)

        self.progress.destroy()
        self.submit_button['state'] = 'normal'

    def call_api(self, start_date_training, end_date_training, start_date_testing, end_date_testing, status):
        url = "http://localhost:8000/api/modelui"
        data = {
            "start_date_training": start_date_training,
            "end_date_training": end_date_training,
            "start_date_testing": start_date_testing,
            "end_date_testing": end_date_testing
        }
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            if response.status_code == 200:
                print("API call successful")
                status[0] = 1
                status[1] = response.json()
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            status[0] = -1
            status[1] = "API call failed"

    def save_to_csv(self, data):
        df = pd.read_json(StringIO(data))
        # convert the first column from milliseconds to date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            self.submit_label['text'] = f"Model output saved to file: {file_path}"
        # remove the download button
        self.master.winfo_children()[-1].destroy()

if __name__ == "__main__":
    root = Tk()
    app = ModelUI(root)
    root.mainloop()
