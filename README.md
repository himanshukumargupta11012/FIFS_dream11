## Project Repository Structure
This repository is structured to streamline data processing, modeling, and UI integration. Below is an overview of each folder and its purpose.


**🚀 Main Application**

**main_app.py:** The main entry point for running the project. This script leverages various modules from the src directory to function.

````
├── README.md                   <- Project overview and usage instructions



├── data                        <- Data folder with all stages of data
│   ├── interim                 <- Intermediate data files generated during processing
│   ├── processed               <- Finalized datasets ready for modeling
│   └── raw                     <- Original data as downloaded
│       ├── cricksheet_data     <- Raw data from Cricksheet
│       └── additional_data     <- Raw data from other sources, if any




├── data_processing             <- Scripts to process data
│   ├── data_download.py        <- Download all project data using this script. All raw data sources are processed here before further use.
│   └── feature_engineering.py  <- Handles all data manipulation and feature engineering for the project.



├── docs                        <- Documentation and project demo
│   └── video_demo              <- Walk-through video, covering setup, UI, and functionality




├── model                       <- Modeling scripts for training and prediction
│   ├── train_model.py          <- Model training script
│   └── predict_model.py        <- Prediction script with trained models



├── model_artifacts             <- Storage for trained models
│                             (Includes pre-trained model for Product UI and models from Model UI)



├── out_of_sample_data          <- Sample dummy data for evaluation matches, After submission is done we will put testing data here (4th - 14th Dec)
                                in the same format as the sample data provided. This folder should be well integrated with Model UI where it will
                                automatically append the new data with already avalaible data from cricksheet.


├── rest                        <- For any miscellaneous requirements not covered by other folders 

└── UI                          <- All files related to the user interface 
````


## Project Setup
To run the project, follow the steps below:

### Data & ML Model
- Install poetry through the following command:
```bash
pip install poetry
```
- In the root directory of the project, run the following command to install all dependencies:
```bash
poetry install
poetry shell
```
- Install parallel (linux library) using:
```bash
sudo apt install parallel
```
- Go to `src/data_processing` folder and run the following command to download all data and process it, do change the num_threads to whatever threads you want to dedicate to the process:
```bash
./data_process.sh ../data/raw/cricksheet/ ../data/interim/ ../data/processed/ 15 <num_threads>
```

### Backend
- Check installation of poetry using :
```bash
pip install poetry
```
- Verify that all data is present in the `src/data` folder, and the `.env` file is present in the root directory.
- Navigate to the root directory of the project and run the following 3 commands to install all dependencies, activate the virtual environment, and start the backend server:
```bash
poetry install
poetry shell
uvicorn main_app:app --reload
```

### Frontend
#### Product UI
- Make sure you have Flutter installed on your machine. If not, follow the instructions [here](https://flutter.dev/docs/get-started/install).
- Verify that flutter is installed by running the following command:
```bash
flutter doctor
```
- Navigate to the `src/UI/product_ui` folder and run the following commands to install all dependencies and start the frontend server:
```bash
flutter pub get
flutter run -d chrome --web-port 8080
```

#### Model UI
- Make sure the backend server is running.
- Navigate to the `src/UI` folder and run the following command to run Model UI:
```bash
python3 Model_UI.py
```
