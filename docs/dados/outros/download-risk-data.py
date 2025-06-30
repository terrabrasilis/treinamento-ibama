from datetime import date
from pathlib import Path
import requests

API_URL="https://terrabrasilis.dpi.inpe.br/stac-api/v1/"
START_DATE=date(2023,12,16)
END_DATE=date(2023,12,16)
DOWNLOAD_DIR="/tmp/risk/download"
dynamic_variables=['ArDS', 'A7Q', 'AcAr', 'CtDS', 'DeAI', 'DeAr', 'NuAI', 'Nuvem', 'OcDS', 'PtDG', 'PtEM', 'Pr']


def get_variables(endpoint, params=None):
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()["features"]
    print(f"error: {response.text}")
    return None

def download_asset(url, download_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"downloaded in {download_path}")
        return True
    print(f"error downloading file {url}: {response.text}")
    return False

endpoint = f"{API_URL}/search"
datetime_range=f"{START_DATE.strftime("%Y-%m-%d")}/{END_DATE.strftime("%Y-%m-%d")}"
params = {
    "collections": "collection1",
    "datetime_range": datetime_range
}
variables = get_variables(endpoint=endpoint, params=params)

assert variables is not None

for variable in variables:
    dynamic_vars=[]
    for name, values in variable['assets'].items():
        dynamic_vars.append(name)
        url = values['href']
        download_dir = Path(f"{DOWNLOAD_DIR}/{name}")
        download_dir.mkdir(exist_ok=True, parents=True)
        download_path = Path(f"{download_dir}/{Path(url).name}")
        if not Path.is_file(download_path):
            print(f"downloading {url} ...")
            download_asset(url=url, download_path=download_path)