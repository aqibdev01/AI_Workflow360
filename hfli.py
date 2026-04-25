from huggingface_hub import HfApi

api = HfApi()


# Upload large folder
api.upload_large_folder(
    folder_path="AI_Workflow360/ai-server/model_weights/decomposition",
    repo_id="aqibdev01/flan-t5-pm-decomposition",
    repo_type="model",
    ignore_patterns=["checkpoint-*"],
)

print("Upload complete!")