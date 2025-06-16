from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Thêm import này
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import requests
import json
from collections import defaultdict
import os
import uvicorn
from contextlib import asynccontextmanager


# ===============================
# PYDANTIC MODELS
# ===============================

class CourseRecommendation(BaseModel):
    course_id: int
    course_name: str
    category: str
    difficulty: str
    duration_hours: int
    rating: float
    num_lessons: int
    ai_score: float


class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    current_level: Optional[int] = None
    learning_goal: Optional[str] = None
    study_time_per_week: Optional[int] = None
    preferred_skill: Optional[str] = None
    country: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id: str
    user_profile: UserProfile
    recommendations: List[CourseRecommendation]
    total_courses: int


# ===============================
# MODEL ARCHITECTURE (Same as training)
# ===============================

class DQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.q_network.eval()  # Set to evaluation mode

    def choose_action(self, state, epsilon=0.0):
        """Choose action with optional exploration"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()


# ===============================
# RECOMMENDATION SYSTEM CLASS
# ===============================

class EnglishLearningRecommendationSystem:
    def __init__(self, model_path="complete_english_learning_model.pkl"):
        """Initialize the recommendation system"""
        self.model_path = model_path
        self.agent = None
        self.course_data = None
        self.user_data = None
        self.mappings = None
        self.course_features = None
        self.load_model()
        self.setup_system()

    def load_model(self):
        """Load trained model and metadata"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Extract hyperparameters
            hyperparams = model_data['hyperparameters']

            # Recreate agent
            self.agent = DQNAgent(
                hyperparams['state_dim'],
                hyperparams['action_dim'],
                hyperparams['hidden_dim'],
                hyperparams['learning_rate'],
                hyperparams['gamma'],
                1.0, 0.01, 0.995
            )

            # Load model weights
            self.agent.q_network.load_state_dict(model_data['model_state_dict'])
            self.mappings = model_data['mappings']

            print("✅ Model loaded successfully!")

        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def setup_system(self):
        """Setup course and user data structures"""
        print("🔄 Fetching course and user data...")

        courses_data = self.fetch_courses_from_api()
        users_data = self.fetch_users_from_api()

        if courses_data is None or users_data is None:
            print("⚠️  API unavailable, using sample data...")
            courses_data, users_data = self.create_sample_data()

        self.course_data = courses_data
        self.user_data = users_data
        self.course_features = self.create_course_features()

        print(f"✅ System ready! {len(self.course_data)} courses, {len(self.user_data)} users")

    def fetch_courses_from_api(self):
        """Fetch courses from API"""
        try:
            url = "http://localhost:8001/api/v1/course-statistics/suggestions/get-courses"
            response = requests.post(url, timeout=5)

            if response.status_code == 200:
                courses = response.json()
                return self.process_courses_data(courses)
            return None
        except:
            return None

    def fetch_users_from_api(self):
        """Fetch users from API"""
        try:
            url = "http://localhost:8001/api/v1/course-statistics/suggestions/get-users"
            response = requests.post(url, timeout=5)

            if response.status_code == 200:
                users = response.json()
                return self.process_users_data(users)
            return None
        except:
            return None

    def process_courses_data(self, api_courses):
        """Process API course data - use real data except for category"""
        course_categories = ['Grammar', 'Vocabulary', 'Speaking', 'Listening', 'Reading', 'Writing', 'TOEIC', 'IELTS']

        courses = []
        for course in api_courses:
            course_info = {
                'course_id': course['course_id'],
                'course_name': course.get('course_name', f"Course_{course['course_id']}"),
                'category': np.random.choice(course_categories),  # Only randomize category
                'difficulty': course.get('difficulty', 'Intermediate'),  # Use API data
                'duration_hours': course.get('duration_hours', 20),  # Use API data
                'rating': course.get('rating', 4.0),  # Use API data
                'num_lessons': course.get('num_lessons', 30),  # Use API data
                'prerequisite_level': course.get('prerequisite_level', 2)  # Use API data
            }
            courses.append(course_info)

        return pd.DataFrame(courses)

    def process_users_data(self, api_users):
        """Process API user data - use real data except for category preferences"""
        course_categories = ['Grammar', 'Vocabulary', 'Speaking', 'Listening', 'Reading', 'Writing', 'TOEIC', 'IELTS']

        users = []
        for user in api_users:
            user_info = {
                'user_id': user['username'],
                'age': user.get('age', 25),  # Use API data
                'current_level': user.get('current_level', 2),  # Use API data
                'learning_goal': user.get('learning_goal', 'General'),  # Use API data
                'study_time_per_week': user.get('study_time_per_week', 10),  # Use API data
                'preferred_skill': user.get('preferred_skill', np.random.choice(course_categories)),
                # Use API data, fallback to random
                'country': user.get('country', 'Vietnam')  # Use API data
            }
            users.append(user_info)

        return pd.DataFrame(users)
    def create_sample_data(self):
        """Create sample data when API is unavailable"""
        # Sample courses
        course_categories = ['Grammar', 'Vocabulary', 'Speaking', 'Listening', 'Reading', 'Writing', 'TOEIC', 'IELTS']
        difficulty_levels = ['Beginner', 'Elementary', 'Intermediate', 'Upper-Intermediate', 'Advanced']

        courses = []
        for i in range(1, 51):  # 50 sample courses
            course = {
                'course_id': i,
                'course_name': f"English Course {i}",
                'category': np.random.choice(course_categories),
                'difficulty': np.random.choice(difficulty_levels),
                'duration_hours': np.random.randint(5, 50),
                'rating': np.random.uniform(3.0, 5.0),
                'num_lessons': np.random.randint(10, 100),
                'prerequisite_level': np.random.randint(0, 5)
            }
            courses.append(course)

        # Sample users
        users = []
        for i in range(1, 21):  # 20 sample users
            user = {
                'user_id': f"user_{i}",
                'age': np.random.randint(15, 60),
                'current_level': np.random.randint(0, 5),
                'learning_goal': np.random.choice(['Business', 'Academic', 'Travel', 'General', 'Test_Prep']),
                'study_time_per_week': np.random.randint(2, 20),
                'preferred_skill': np.random.choice(course_categories),
                'country': np.random.choice(['Vietnam', 'Korea', 'Japan', 'China', 'Thailand', 'Other'])
            }
            users.append(user)

        return pd.DataFrame(courses), pd.DataFrame(users)

    def create_course_features(self):
        """Create course feature vectors"""
        num_categories = len(self.mappings['category_mapping'])
        num_difficulties = 5

        course_features = {}

        for _, course in self.course_data.iterrows():
            course_id = course['course_id']

            # One-hot encode category
            category_feature = np.zeros(num_categories)
            if course['category'] in self.mappings['category_mapping']:
                category_idx = self.mappings['category_mapping'][course['category']]
                category_feature[category_idx] = 1

            # One-hot encode difficulty
            difficulty_feature = np.zeros(num_difficulties)
            if course['difficulty'] in self.mappings['difficulty_mapping']:
                difficulty_idx = self.mappings['difficulty_mapping'][course['difficulty']]
                difficulty_feature[difficulty_idx] = 1

            # Normalized numerical features
            duration_norm = min(course['duration_hours'] / 50.0, 1.0)
            rating_norm = course['rating'] / 5.0
            lessons_norm = min(course['num_lessons'] / 100.0, 1.0)

            # Combine features
            feature_vector = np.concatenate([
                category_feature,
                difficulty_feature,
                [duration_norm, rating_norm, lessons_norm]
            ])

            course_features[course_id] = feature_vector

        return course_features

    def encode_user_features(self, user_info):
        """Encode user features"""
        num_age_groups = 5
        num_levels = 5
        num_goals = len(self.mappings['goal_mapping'])
        num_skills = len(self.mappings['skill_mapping'])

        # Age groups
        age = user_info.get('age', 25)
        if age <= 20:
            age_group = 0
        elif age <= 30:
            age_group = 1
        elif age <= 40:
            age_group = 2
        elif age <= 50:
            age_group = 3
        else:
            age_group = 4

        age_feature = np.zeros(num_age_groups)
        age_feature[age_group] = 1

        # Level
        level = user_info.get('current_level', 2)
        level_feature = np.zeros(num_levels)
        level_feature[min(level, num_levels - 1)] = 1

        # Goal
        goal_feature = np.zeros(num_goals)
        goal = user_info.get('learning_goal', 'General')
        if goal in self.mappings['goal_mapping']:
            goal_idx = self.mappings['goal_mapping'][goal]
            goal_feature[goal_idx] = 1

        # Skill
        skill_feature = np.zeros(num_skills)
        skill = user_info.get('preferred_skill', 'Grammar')
        if skill in self.mappings['skill_mapping']:
            skill_idx = self.mappings['skill_mapping'][skill]
            skill_feature[skill_idx] = 1

        # Study time
        study_time_norm = min(user_info.get('study_time_per_week', 10) / 20.0, 1.0)

        # Combine features
        user_feature = np.concatenate([
            age_feature, level_feature, goal_feature, skill_feature, [study_time_norm]
        ])

        return user_feature

    def encode_state(self, user_info, course_id):
        """Combine user and course features into state"""
        user_feature = self.encode_user_features(user_info)
        course_feature = self.course_features.get(course_id, np.zeros(len(list(self.course_features.values())[0])))

        state = np.concatenate([user_feature, course_feature])
        return state

    def recommend_courses(self, user_info, top_k=5, exclude_courses=None):
        """Recommend top-k courses for a user"""
        if exclude_courses is None:
            exclude_courses = []

        available_courses = [cid for cid in self.course_data['course_id']
                             if cid not in exclude_courses]

        if not available_courses:
            return []

        # Score all available courses
        course_scores = []

        for course_id in available_courses:
            state = self.encode_state(user_info, course_id)

            # Get Q-value from trained model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.q_network(state_tensor)

                # Find the action index for this course
                try:
                    course_idx = available_courses.index(course_id) % self.agent.action_dim
                    score = q_values[0][course_idx].item()
                except:
                    score = q_values[0].max().item()  # Fallback

            course_scores.append((course_id, score))

        # Sort by score and return top-k
        course_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_course_ids = [cid for cid, _ in course_scores[:top_k]]

        # Get detailed course information
        recommendations = []
        for course_id in recommended_course_ids:
            course_info = self.course_data[self.course_data['course_id'] == course_id].iloc[0]
            score = next(score for cid, score in course_scores if cid == course_id)

            recommendations.append({
                'course_id': course_id,
                'course_name': course_info['course_name'],
                'category': course_info['category'],
                'difficulty': course_info['difficulty'],
                'duration_hours': course_info['duration_hours'],
                'rating': round(course_info['rating'], 2),
                'num_lessons': course_info['num_lessons'],
                'ai_score': round(score, 4)
            })

        return recommendations

    def get_user_info(self, user_id):
        """Get user information by user_id"""
        user_match = self.user_data[self.user_data['user_id'] == user_id]
        if user_match.empty:
            return None
        return user_match.iloc[0].to_dict()

    def get_all_users(self):
        """Get all available users"""
        return self.user_data['user_id'].tolist()


# ===============================
# GLOBAL VARIABLES
# ===============================
rec_system = None


# ===============================
# FASTAPI LIFECYCLE
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rec_system
    print("🚀 Starting FastAPI Course Recommendation Service...")
    try:
        rec_system = EnglishLearningRecommendationSystem()
        print("✅ Recommendation system initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize recommendation system: {e}")
        raise

    yield

    # Shutdown
    print("🔽 Shutting down recommendation service...")


# ===============================
# FASTAPI APPLICATION
# ===============================

app = FastAPI(
    title="English Learning Course Recommendation API",
    description="AI-powered course recommendation system for English learning",
    version="1.0.0",
    lifespan=lifespan
)

# ===============================
# CORS CONFIGURATION
# ===============================

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React development server
        "http://127.0.0.1:3000",     # Alternative localhost
        "http://localhost:3001",      # Alternative port
        "http://localhost:8080",      # Alternative port
        # Thêm các domain khác nếu cần
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Cho phép tất cả headers
)

# Hoặc để cho phép tất cả origins (chỉ dùng trong development):
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# ===============================
# API ENDPOINTS
# ===============================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "English Learning Course Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "cors_enabled": True,
        "allowed_origins": ["localhost:3000"],
        "endpoints": {
            "recommendations": "/recommendations/{user_id}",
            "users": "/users",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rec_system
    return {
        "status": "healthy" if rec_system is not None else "unhealthy",
        "model_loaded": rec_system is not None,
        "total_courses": len(rec_system.course_data) if rec_system else 0,
        "total_users": len(rec_system.user_data) if rec_system else 0,
        "cors_enabled": True
    }


@app.get("/users")
async def get_all_users():
    """Get all available users"""
    global rec_system
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized")

    users = rec_system.get_all_users()
    return {
        "total_users": len(users),
        "users": users
    }


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
        user_id: str,
        top_k: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return"),
        exclude_courses: Optional[str] = Query(default=None,
                                               description="Comma-separated list of course IDs to exclude")
):
    """Get course recommendations for a specific user"""
    global rec_system

    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized")

    # Get user info
    user_info = rec_system.get_user_info(user_id)
    if user_info is None:
        available_users = rec_system.get_all_users()
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"User '{user_id}' not found",
                "available_users": available_users[:10],  # Show first 10 users
                "total_available_users": len(available_users)
            }
        )

    # Parse exclude_courses
    exclude_list = []
    if exclude_courses:
        try:
            exclude_list = [int(cid.strip()) for cid in exclude_courses.split(',') if cid.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid exclude_courses format. Use comma-separated integers.")

    # Get recommendations
    try:
        recommendations = rec_system.recommend_courses(user_info, top_k=top_k, exclude_courses=exclude_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    # Create user profile
    user_profile = UserProfile(
        user_id=user_id,
        age=user_info.get('age'),
        current_level=user_info.get('current_level'),
        learning_goal=user_info.get('learning_goal'),
        study_time_per_week=user_info.get('study_time_per_week'),
        preferred_skill=user_info.get('preferred_skill'),
        country=user_info.get('country')
    )

    # Convert recommendations to response format
    course_recommendations = [
        CourseRecommendation(**rec) for rec in recommendations
    ]

    return RecommendationResponse(
        user_id=user_id,
        user_profile=user_profile,
        recommendations=course_recommendations,
        total_courses=len(course_recommendations)
    )


@app.get("/recommendations/{user_id}/simple")
async def get_simple_recommendations(
        user_id: str,
        top_k: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return")
):
    """Get simple course recommendations (course IDs only)"""
    global rec_system

    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized")

    # Get user info
    user_info = rec_system.get_user_info(user_id)
    if user_info is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    # Get recommendations
    try:
        recommendations = rec_system.recommend_courses(user_info, top_k=top_k)
        course_ids = [rec['course_id'] for rec in recommendations]

        return {
            "user_id": user_id,
            "recommended_course_ids": course_ids,
            "total_recommendations": len(course_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# ===============================
# MAIN FUNCTION
# ===============================

if __name__ == "__main__":
    print("🌟 Starting English Learning Course Recommendation API...")
    print("✅ CORS enabled for localhost:3000")
    uvicorn.run(
        "main:app",  # Change this to your actual filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )