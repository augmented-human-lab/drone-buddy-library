import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class S3ImageUploader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name='us-east-1'):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = bucket_name

    def upload_image(self, file_path, s3_key):
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"Upload Successful: {file_path} to bucket: {self.bucket_name} with key: {s3_key}")
        except FileNotFoundError:
            print(f"The file was not found: {file_path}")
        except NoCredentialsError:
            print("Credentials not available")
        except ClientError as e:
            print(f"Client error: {e}")


# Usage example
if __name__ == "__main__":
    aws_access_key_id = 'your_aws_access_key_id'
    aws_secret_access_key = 'your_aws_secret_access_key'
    bucket_name = 'drone-buddy'

    uploader = S3ImageUploader(aws_access_key_id, aws_secret_access_key, bucket_name)

    file_path = 'path_to_your_image.jpg'
    s3_key = 'your_s3_key.jpg'

    uploader.upload_image(file_path, s3_key)
