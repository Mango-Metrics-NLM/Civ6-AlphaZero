from google.cloud import aiplatform

def find_jobs():
    aiplatform.init(project="hrmtrain", location="us-central1")
    jobs = aiplatform.CustomJob.list(
        filter='display_name="civ6az-tournament-51models"',
        order_by="create_time desc",
    )
    for job in jobs:
        print(f"Name: {job.display_name}")
        print(f"ID: {job.name}")
        print(f"Resource Name: {job.resource_name}")
        print(f"State: {job.state}")
        print(f"Create Time: {job.create_time}")
        print("-" * 20)

if __name__ == "__main__":
    find_jobs()
