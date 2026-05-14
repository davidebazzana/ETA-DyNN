import requests


def upload(file_path:str, server_ip:str="192.168.0.68"):
    url = f"http://{server_ip}:8000/upload"

    # "/mnt/datasets/prin/video_raw/v2/varroa_free/100 2024-08-22 10-50-30.mkv"
    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f}
        )

    print(response.json())
