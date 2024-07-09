import boto3
import os 

# Create an S3 client
s3 = boto3.client('s3')


def download_directory_from_s3(bucket, s3_directory, local_directory):
    s3 = boto3.client('s3')
    
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=s3_directory):
        if 'Contents' in result:
            for file in result['Contents']:
                s3_path = file['Key']
                if not s3_path.endswith('/'):  # Skip directories
                    local_path = os.path.join(local_directory, os.path.relpath(s3_path, s3_directory))
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    print(f"Downloading {s3_path} to {local_path}")
                    s3.download_file(bucket, s3_path, local_path)

# Download a file
# bucket_name = 'mind-central'
# file_name = 'datasets/info_small.json'
# s3.download_file(bucket_name, file_name, "test.json")


# Usage
bucket_name = 'mind-central'
s3_dir = 'datasets'
local_dir = './datasets'

download_directory_from_s3(bucket_name, s3_dir, local_dir)