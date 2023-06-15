from google.cloud import vision


def init_google_vision_engine():
    return vision.ImageAnnotatorClient()


def detect_text(client, image_path):
    # Read the image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object
    image = vision.Image(content=content)

    # Perform OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations

    return response
