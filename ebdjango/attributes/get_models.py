import boto3
from django.conf import settings

def list_objects(Bucket):

	s3 = boto3.client(
	    's3',
	    aws_access_key_id=settings.AWS_ACCESS_KEY,
	    aws_secret_access_key=settings.AWS_SECRET_KEY,
	    region_name=settings.AWS_S3_REGION_NAME
	    )

	objs = s3.list_objects(
	    Bucket=Bucket,
	    )
	return objs

def download_file(Bucket, Key, Filename):
	s3 = boto3.client(
	    's3',
	    aws_access_key_id=settings.AWS_ACCESS_KEY,
	    aws_secret_access_key=settings.AWS_SECRET_KEY,
	    region_name=settings.AWS_S3_REGION_NAME
	    )

	s3.download_file(Bucket=Bucket, Key=Key, Filename=Filename)

if __name__ == "__main__":
	list_buckets()

