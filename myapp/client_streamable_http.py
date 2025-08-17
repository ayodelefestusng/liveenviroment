# myapp/client_streamable_http.py

import requests

def stream_data():
    url = "https://example.com/stream"
    response = requests.get(url, stream=True)
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
    return "Stream completed"