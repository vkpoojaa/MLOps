"""Run the prediction API with Uvicorn: python -m mlops_pipeline.serving."""
import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "mlops_pipeline.serving.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        access_log=True,
    )
