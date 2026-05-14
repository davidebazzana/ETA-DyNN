import requests


def upload(file_path:str, server_ip:str="192.168.0.68"):
    url = f"http://{server_ip}:8000/upload"

    # "/mnt/datasets/prin/video_raw/v2/varroa_free/100 2024-08-22 10-50-30.mkv"
    with open(file_path, "rb") as f:
        try:
            response = requests.post(
                url,
                files={"file": f}
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)

        except requests.exceptions.ConnectionError:
            print("Connection failed")

        except requests.exceptions.Timeout:
            print("Request timed out")

        except requests.exceptions.RequestException as e:
            print("Other request error:", e)

