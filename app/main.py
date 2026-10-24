from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
from app.recommender import recommender

app = FastAPI(title="Z Video Recommendation Service", version="1.0.0")

class RankRequest(BaseModel):
    user_id: str
    limit: Optional[int] = 30

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "z-recommendation", "ml_trained": recommender.is_trained}

@app.post("/recommend/for-you")
def get_for_you_recommendations(req: RankRequest):
    """
    Ranks videos based on watch time, likes, comments, shares,
    recency and user interest matching.
    """
    ranked_videos = recommender.rank_for_you(user_id=req.user_id, limit=req.limit)
    return {
        "userId": req.user_id,
        "count": len(ranked_videos),
        "results": ranked_videos
    }

@app.post("/model/train")
def trigger_training():
    """Retrain scikit-learn ranking model on latest engagement data"""
    res = recommender.train_model()
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
