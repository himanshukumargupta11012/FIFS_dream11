from datetime import date, datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import re
import json
import os
import pandas as pd
import random
from src.model.predict_model import forward, true_forward
from src.model.model_ui_train import train_and_test_model
import httpx
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path_players = './src/data/raw/additional_data/player_data.csv'
file_path_players_1 = './src/data/raw/cricksheet/names.csv'
file_path_teams = './src/data/raw/cricksheet/data/teams.csv'
data_players = pd.read_csv(file_path_players)
data_players_1 = pd.read_csv(file_path_players_1)
data_teams = pd.read_csv(file_path_teams)
with open("./src/data/raw/cricksheet/data/team_data.json", "r") as file:
    team_data = json.load(file)

@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.get("/matches/{date}/{format}")
async def get_matches(date: str, format: Literal["T20", "ODI", "Test"]):
    # verify the date format to YYYY-MM-DD
    if not re.match(r"\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    try:
        matches = pd.read_csv("./src/data/interim/matches_info.csv")
        matches = matches[matches['date'] == date]

        formats = {
            "T20": ["IT20", "T20"],
            "ODI": ["ODI", "ODM"],
            "Test": ["Test", "MDM"]
        }

        res_matches = []
        for match in matches.values:
            with open('./src/data/raw/cricksheet/data/all_json/' + match[4] + '.json') as f:
                json_data = f.read()


            data = json.loads(json_data)
            if data['info']['match_type'] not in formats[format]:
                continue
            registry = data['info']['registry']
            
            res_matches.append({
                "teamA": match[5].split(" vs ")[0],
                "teamB": match[5].split(" vs ")[1],
                "venue": data['info']['venue'],
                "format": format,
                "registry": registry['people'],
                "players": data['info']['players'],
                "match_id": match[4],
                "date": date
            })

        return res_matches
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the request", )

@app.get("/logo/{name}")
async def get_images(name: str):
    """
    Serve an image file from the specified directory.
    """
    base_dir = './src/data/raw/additional_data/logo'
    
    image_path = os.path.join(base_dir, name + ".png")
    
    if not os.path.exists(image_path):
        print("NOT FOUND")
        placeholder = os.path.join(base_dir, "placeholder.png")
        return FileResponse(placeholder, media_type='image/jpeg')

    return FileResponse(image_path, media_type='image/jpeg')
  
@app.get("/players/search")
async def get_players_name(q: str = Query(..., description="Name to search for")):
    """
    Search for player names containing the query string (case-insensitive).
    """
    matching_players = data_players_1[data_players_1['name'].str.contains(q, case=False, na=False)]
    
    results = matching_players[['identifier', 'name']].dropna().to_dict(orient="records")
    
    return {"matching_players": results}

@app.get("/teams/search")
async def get_teams_name(q: str = Query(..., description="Name to search for")):
    """
    Search for team names containing the query string (case-insensitive).
    """
    try:
        matching_teams = data_teams[data_teams['Team'].str.contains(q, case=False, na=False)]
        results = matching_teams[['Team']].to_dict(orient="records")
        return {"matching_teams": results}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.get("/team/{name}")
async def get_team_data(name: str):
    # Search for the team
    for team in team_data:
        if team["team_name"].lower() == name.lower():
            return team

    raise HTTPException(status_code=404, detail="Team not found")

@app.get("/cdn/{id}")
async def get_images(id: str):
    """
    Serve an image file from the specified directory.
    """
    base_dir = './src/data/raw/additional_data/images'
    
    image_path = os.path.join(base_dir, id + ".jpg")
    
    if not os.path.exists(image_path):
        placeholder = os.path.join(base_dir, "placeholder.png")
        return FileResponse(placeholder, media_type='image/jpeg')

    return FileResponse(image_path, media_type='image/jpeg')

DATA_DIR = Path("./src/data/upcoming")

@app.get("/matches/upcoming", response_class=JSONResponse)
async def get_upcoming_matches():
    now = datetime.now()
    file_name = f"{now.strftime('%Y-%m-%d-%H')}.json" 
    file_path = DATA_DIR / file_name
    
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Corrupted JSON data.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")
    else:
        external_api_url = os.getenv("UPCOMING_MATCH_URL")
        external_api_url = 'https://api.cricapi.com/v1/cricScore?apikey=3f63f4e2-cb60-4020-a0f6-0bd11ab0c1c2'
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(external_api_url, timeout=10.0)
                response.raise_for_status()  
                data = response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to external API: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error response {e.response.status_code} from external API.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Invalid JSON received from external API.")
        
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating data directory: {str(e)}")
        
        try:
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")
        
        return data

@app.post("/api/modelui")
async def get_model_output(data: dict):
    start_date_training = data['start_date_training']
    end_date_training = data['end_date_training']
    start_date_testing = data['start_date_testing']
    end_date_testing = data['end_date_testing']
    print(f"Received POST request with data: {data}")
    output_csv_path = f"src/data/processed/model_ui_output_{end_date_testing}.csv"
    output_df_json = train_and_test_model(start_date_training, end_date_training, start_date_testing, end_date_testing, output_csv_path)
    return output_df_json

@app.post("/api/predict")
async def get_prediction(data: dict):
    date = data['date']
    if not re.match(r"\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    format = data['format']
    if format == "ODI":
        format = "OD"
    players_id_list = data['players_id_list']
    match_id = str(data['match_id']) if data['match_id'] else None
    output = forward(date, format, players_id_list, match_id)
    res = {}
    try:
        for i in range(22):
            res[output[2][i]] = {
                "name": data_players[data_players['identifier'] == output[2][i]]['full_name'].values[0],
                "score": output[0][i],
                "explanation": output[1][i]
            }
    except Exception as e:
        print(e)
        return res
    return res

@app.post("/api/true_predict")
async def get_true_prediction(data: dict):
    date = data['date']
    if not re.match(r"\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    format = data['format']
    if format == "ODI":
        format = "OD"
    players_id_list = data['players_id_list']
    match_id = str(data['match_id']) if data['match_id'] else None
    output = true_forward(date, format, players_id_list, match_id)
    res = {}
    try:
        for i in range(22):
            res[output[2][i]] = {
                "name": data_players[data_players['identifier'] == output[2][i]]['full_name'].values[0],
                "score": output[0][i],
                "explanation": output[1][i]
            }
    except Exception as e:
        print(e)
        return res
    return res

@app.get("/api/player/{player_id}")
def get_player_info(player_id: str):
    players_data = pd.read_csv("src/data/raw/additional_data/player_data.csv")
    player = players_data[players_data['identifier'] == player_id]
    print(player)
    player = player.where(pd.notnull(player), None)
    return player.to_dict(orient="records")[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)