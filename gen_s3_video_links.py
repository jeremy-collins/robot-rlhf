import os
import boto3
import pandas as pd
import numpy as np
from random import shuffle
from botocore.exceptions import ClientError
from botocore.client import Config
import urllib


def create_presigned_url(bucket_name, object_name, expiration=3600):
    s3_resource = boto3.resource('s3', config=Config(signature_version='s3v4'))
    bucket = s3_resource.Bucket(bucket_name)
    obj = bucket.Object(object_name)
    url = obj.meta.client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=expiration)
    return url

def upload_to_s3(file_path, bucket_name, object_name):
    # s3_client = boto3.client('s3')
    s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
    s3_client.upload_file(file_path, bucket_name, object_name)
    # s3_client.upload_file(file_path, bucket_name, object_name, ExtraArgs={'ACL': 'public-read'})

    return f'https://{bucket_name}.s3.amazonaws.com/{object_name}'

def upload_videos_and_generate_csv(input_folder, bucket_name, output_csv):
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    shuffle(video_files)

    n = len(video_files)
    half_n = n // 2

    data = {
        'video_name_1': [],
        'video_url_1': [],
        'video_name_2': [],
        'video_url_2': []
    }

    for i, video_file in enumerate(video_files):
        file_path = os.path.join(input_folder, video_file)
        object_name = f'videos_6/{video_file}'
        video_url = upload_to_s3(file_path, bucket_name, object_name)
        print(f'Uploaded {video_file} to {video_url}')

        if i < half_n:
            data['video_name_1'].append(video_file)
            data['video_url_1'].append(video_url)
        else:
            data['video_name_2'].append(video_file)
            data['video_url_2'].append(video_url)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f'Generated {output_csv}')

if __name__ == '__main__':
    input_folder = 'data/videos_random_policy'
    bucket_name = 'hri-rlhf'
    output_csv = 's3_output_6.csv'

    upload_videos_and_generate_csv(input_folder, bucket_name, output_csv)