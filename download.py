from minio import Minio

minio_client = Minio(
    "168.131.152.80",  # MinIO server endpoint
    access_key="ZrbpazySscP6K3L5ZkRK",
    secret_key="AuiaLr3X3Cr8EfzT1PAzxwYIF0iUWG8BwX9lWkwh",
    secure=True,  # Set to True if using HTTPS,
    cert_check=False
)


# for item in minio_client.list_objects("bucket.name",recursive=True):
#     minio_client.fget_object("bucket.name",item.object_name,item.object_name)
# minio_client.fget_object("temporary","ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip","/home/hdd2/jo/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip")
# minio_client.fget_object("temporary","ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip","/home/hdd2/jo/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip")