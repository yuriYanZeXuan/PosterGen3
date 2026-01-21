
PROXY_PORT=51958
PROXY_ENDPOINT="https://runway.devops.rednote.life/openai/google/v1:generateContent"
PROXY_API_KEY=""   # optional; set if your upstream requires it

# Pipeline args (copied from PosterGen2_0/run.sh; adjust paths to your machine)
uv run python -m src.workflow.pipeline \
  --poster_width 54 --poster_height 36 \
  --paper_path /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/paper.pdf \
  --text_model gemini-2.5-pro \
  --vision_model gemini-2.5-pro \
  --logo /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/logo.png \
  --aff_logo /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/aff.png

