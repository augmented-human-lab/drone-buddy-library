from google.cloud import vision


def init_google_vision_engine():
    """
    Initializes the Google Cloud Vision client for image annotation.

    Returns:
        vision.ImageAnnotatorClient: The initialized Vision client.

    Example:
        vision_client = init_google_vision_engine()
    """
    return vision.ImageAnnotatorClient()


def detect_text(client, image_path):
    """
     Detects text in an image using the Google Cloud Vision client.

     Args:
         client (vision.ImageAnnotatorClient): The Vision client instance.
         image_path (str): The path to the image file to be analyzed.

     Returns:
         google.protobuf.json_format.MessageToJson: The response containing detected text annotations.

     Example:
         vision_client = init_google_vision_engine()
         text_response = detect_text(vision_client, 'image.jpg')
         print(text_response)
     """
    # Read the image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object
    image = vision.Image(content=content)

    # Perform OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations

    return response
