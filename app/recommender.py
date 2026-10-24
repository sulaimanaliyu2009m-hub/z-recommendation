import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from app.database import get_db_connection

class VideoRecommender:
    def __init__(self):
        # Machine learning model to predict engagement score
        self.ml_model = Ridge(alpha=1.0)
        self.is_trained = False

    def get_candidate_videos(self):
        """Fetch candidate video posts from PostgreSQL"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                p.id, p.user_id, p.content, p.created_at,
                COALESCE(l.likes_count, 0) as likes_count,
                COALESCE(r.reposts_count, 0) as reposts_count,
                COALESCE(c.comments_count, 0) as comments_count,
                COALESCE(v.views_count, 0) as views_count,
                COALESCE(v.avg_completion, 0) as avg_completion_rate,
                COALESCE(array_agg(h.tag) FILTER (WHERE h.tag IS NOT NULL), '{}') as hashtags
            FROM posts p
            LEFT JOIN (SELECT post_id, COUNT(*) as likes_count FROM likes GROUP BY post_id) l ON l.post_id = p.id
            LEFT JOIN (SELECT post_id, COUNT(*) as reposts_count FROM reposts GROUP BY post_id) r ON r.post_id = p.id
            LEFT JOIN (SELECT post_id, COUNT(*) as comments_count FROM comments GROUP BY post_id) c ON c.post_id = p.id
            LEFT JOIN (
                SELECT post_id, COUNT(*) as views_count, AVG(completion_rate) as avg_completion 
                FROM video_views GROUP BY post_id
            ) v ON v.post_id = p.id
            LEFT JOIN post_hashtags ph ON ph.post_id = p.id
            LEFT JOIN hashtags h ON h.id = ph.hashtag_id
            WHERE p.post_type = 'video' AND p.media_url IS NOT NULL
            GROUP BY p.id, l.likes_count, r.reposts_count, c.comments_count, v.views_count, v.avg_completion;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.DataFrame(rows)

    def get_user_profile(self, user_id: str):
        """Fetch user interests and watch history from PostgreSQL"""
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Fetch user selected interests
        cursor.execute("SELECT interests FROM users WHERE id = %s", (user_id,))
        user_row = cursor.fetchone()
        user_interests = user_row['interests'] if user_row and user_row['interests'] else []

        # 2. Fetch posts user already watched or liked
        cursor.execute("SELECT post_id FROM video_views WHERE user_id = %s", (user_id,))
        watched_posts = set(r['post_id'] for r in cursor.fetchall())

        cursor.close()
        conn.close()
        return set(i.lower() for i in user_interests), watched_posts

    def calculate_recency_decay(self, created_at, half_life_hours=48.0):
        """Exponential decay based on hours since creation"""
        if not created_at:
            return 0.5
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        diff_hours = (now - created_at).total_seconds() / 3600.0
        return math.exp(-math.log(2) * (diff_hours / half_life_hours))

    def calculate_interest_overlap(self, user_interests, post_tags):
        """Jaccard similarity between user interests and video tags"""
        if not user_interests or not post_tags:
            return 0.1
        post_set = set(t.lower() for t in post_tags)
        intersection = user_interests.intersection(post_set)
        return len(intersection) / max(len(user_interests), 1)

    def rank_for_you(self, user_id: str, limit: int = 30):
        """Rank videos for user's For You feed"""
        df = self.get_candidate_videos()
        if df.empty:
            return []

        user_interests, watched_posts = self.get_user_profile(user_id)

        features = []
        for _, row in df.iterrows():
            views = max(float(row['views_count']), 1.0)
            
            # Feature 1: Completion Rate
            completion = float(row['avg_completion_rate'])

            # Feature 2: Engagement Rate (likes, comments, reposts weighted by views)
            engagement = (float(row['likes_count']) * 3.0 + float(row['comments_count']) * 2.0 + float(row['reposts_count']) * 4.0) / views
            engagement = min(engagement, 5.0) / 5.0

            # Feature 3: Interest Affinity Overlap
            interest_match = self.calculate_interest_overlap(user_interests, row['hashtags'])

            # Feature 4: Recency Decay
            recency = self.calculate_recency_decay(row['created_at'])

            # Penalty for already-watched videos
            seen_penalty = 0.4 if row['id'] in watched_posts else 1.0

            # --- 1. BASELINE SCORING FORMULA ---
            # Score = 30% WatchRate + 25% Engagement + 25% Interests + 20% Recency
            formula_score = (
                (0.30 * completion) + 
                (0.25 * engagement) + 
                (0.25 * interest_match) + 
                (0.20 * recency)
            ) * seen_penalty

            features.append({
                'id': row['id'],
                'completion': completion,
                'engagement': engagement,
                'interest_match': interest_match,
                'recency': recency,
                'formula_score': formula_score
            })

        feat_df = pd.DataFrame(features)

        # --- 2. MACHINE LEARNING RANKER (Scikit-Learn) ---
        if self.is_trained:
            X = feat_df[['completion', 'engagement', 'interest_match', 'recency']]
            feat_df['final_score'] = self.ml_model.predict(X)
        else:
            feat_df['final_score'] = feat_df['formula_score']

        ranked = feat_df.sort_values(by='final_score', ascending=False)
        return ranked[['id', 'final_score']].head(limit).to_dict(orient='records')

    def train_model(self):
        """Train scikit-learn Ridge model on historical watch and like data"""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT 
                vv.completion_rate,
                (CASE WHEN l.id IS NOT NULL THEN 1.0 ELSE 0.0 END) as did_like,
                EXTRACT(EPOCH FROM (vv.created_at - p.created_at)) / 3600.0 as hours_to_watch
            FROM video_views vv
            JOIN posts p ON vv.post_id = p.id
            LEFT JOIN likes l ON l.post_id = vv.post_id AND l.user_id = vv.user_id
            LIMIT 5000;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(rows) < 10:
            return {"status": "insufficient_data", "count": len(rows)}

        train_df = pd.DataFrame(rows)
        X = train_df[['completion_rate', 'hours_to_watch']]
        # Synthetic engagement label
        y = train_df['did_like'] * 0.6 + train_df['completion_rate'] * 0.4

        self.ml_model.fit(X, y)
        self.is_trained = True
        return {"status": "trained", "samples": len(train_df)}

recommender = VideoRecommender()
