import requests


def inference_call(api_url, file_path, model_in_use, sparrow_key):
    with open(file_path, "rb") as file:
        # Prepare the payload
        files = {
            'file': (file.name, file, 'image/jpeg')
        }

        data = {
            'image_url': '',
            'model_in_use': model_in_use,
            'sparrow_key': sparrow_key
        }

        response = requests.post(api_url, data=data, files=files)
    if response.status_code != 200:
        print('Request failed with status code:', response.status_code)
        print('Response:', response.text)

    return response.text
