import boto3
import pytest


# function to check bucket exist or not
def s3Bucket_isEmpty(bucket_name):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name)
    return "Contents" not in response


# test case to check bucket exist or not
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("bucket", [("delivery-time-estimate-data")])
def test_s3_bucket_exist(bucket):
    assert not s3Bucket_isEmpty(bucket), f"Bucket {bucket} not exist"


# function to check data exist in bucket or not
def s3_file_exist(bucket, file):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket)
    return file in [i["Key"] for i in response["Contents"]]


# test case to check data exist in bucket or not
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("bucket, file", [("delivery-time-estimate-data", "train.csv")])
def test_s3_bucket_has_data(bucket, file):
    assert s3_file_exist(bucket, file), f"Data {file} not exist"
