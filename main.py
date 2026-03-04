from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson.objectid import ObjectId
from insightface.app import FaceAnalysis
import numpy as np
import faiss
import io
from PIL import Image
import qrcode
from dotenv import load_dotenv
import os
import cloudinary
import cloudinary.uploader
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import List, Optional

load_dotenv()

app = FastAPI(title="活動照片 AI 分發系統")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB 連接
client = MongoClient(os.getenv("MONGO_URI"))
db = client.photo_db
users = db.users
events = db.events

# Cloudinary 配置
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# InsightFace 初始化
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# 認證設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

def verify_password(plain_password, hashed_password):
    plain_bytes = plain_password.encode('utf-8')[:72]
    return pwd_context.verify(plain_bytes.decode('utf-8'), hashed_password)

def get_password_hash(password):
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes.decode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

# 註冊
@app.post("/register")
async def register(username: str, password: str):
    if users.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_pw = get_password_hash(password)
    users.insert_one({"username": username, "hashed_password": hashed_pw})
    return {"message": "User registered successfully"}

# 登入
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 建立活動
@app.post("/create_event")
async def create_event(event_name: str, current_user: dict = Depends(get_current_user)):
    event_data = {
        "name": event_name,
        "owner": current_user["username"],
        "photos": [],
        "embeddings": [],
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    result = events.insert_one(event_data)
    event_id = str(result.inserted_id)  # 轉 string 給前端
    
    event_url = f"http://localhost:3000/event/{event_id}"
    qr = qrcode.make(event_url)
    qr_path = f"qr_{event_id}.png"
    qr.save(qr_path)
    
    return {"event_id": event_id, "qr_path": qr_path, "event_url": event_url}

# 上傳照片
@app.post("/upload_photos/{event_id}")
async def upload_photos(
    event_id: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID 格式")
    
    if not event or event["owner"] != current_user["username"]:
        raise HTTPException(404, "活動不存在或無權限")
    
    photos = event.get("photos", [])
    embeddings_list = event.get("embeddings", [])
    
    uploaded_count = 0
    for file in files:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        rgb_img = np.array(img.convert("RGB"))
        
        faces = app_face.get(rgb_img)
        if not faces:
            continue
        
        upload_result = cloudinary.uploader.upload(content)
        photo_url = upload_result["secure_url"]
        
        for face in faces:
            embedding = face.embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # 正規化
            embedding_id = len(embeddings_list)
            photos.append({
                "url": photo_url,
                "embedding_id": embedding_id,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_count": len(faces)
            })
            embeddings_list.append(embedding.tolist())
            uploaded_count += 1
    
    events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": {"photos": photos, "embeddings": embeddings_list}}
    )
    
    return {"message": f"成功上傳 {uploaded_count} 張人臉照片"}

# 自拍找照片（公開）- 回傳所有超過閥值的照片
@app.post("/find_photos/{event_id}")
async def find_photos(event_id: str, selfie: UploadFile = File(...)):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID 格式")
    
    if not event or not event.get("is_active", True):
        raise HTTPException(404, "活動不存在或已結束")
    
    content = await selfie.read()
    img = Image.open(io.BytesIO(content))
    rgb_img = np.array(img.convert("RGB"))
    
    faces = app_face.get(rgb_img)
    if not faces:
        return {"photos": [], "message": "未偵測到人臉"}
    
    query_emb = faces[0].embedding.astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)  # 正規化
    
    embeddings = event.get("embeddings", [])
    if not embeddings:
        return {"photos": [], "message": "活動無人臉資料"}
    
    embeddings = np.array([np.array(emb) for emb in embeddings])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # 全部正規化
    
    index = faiss.IndexFlatIP(512)
    index.add(embeddings)
    
    distances, indices = index.search(query_emb.reshape(1, -1), k=len(embeddings))  # 找所有
    
    matching_urls = []
    threshold = 0.45  # cosine similarity > 0.45 才匹配（可調高到 0.55~0.65 更嚴格）
    
    print(f"自拍與圖庫照片的相似度（前5）：")
    for i in range(min(5, len(distances[0]))):
        dist = distances[0][i]
        print(f"  照片 {indices[0][i]}: {dist:.4f}")
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or dist < threshold:
            break
        photo = event["photos"][idx]
        matching_urls.append(photo["url"])
    
    if not matching_urls:
        return {"photos": [], "message": f"未找到相似照片（最高相似度 {distances[0][0]:.4f}）"}
    
    return {"photos": matching_urls, "message": f"找到 {len(matching_urls)} 張相似照片"}

# 我的活動列表
@app.get("/my_events")
async def my_events(current_user: dict = Depends(get_current_user)):
    user_events = list(events.find({"owner": current_user["username"]}))
    
    result = []
    for event in user_events:
        result.append({
            "event_id": str(event["_id"]),
            "name": event["name"],
            "created_at": event.get("created_at", "未知"),
            "photos_count": len(event.get("photos", []))
        })
    
    return result

# 取得活動詳細資訊（管理頁用）
@app.get("/get_event_info/{event_id}")
async def get_event_info(event_id: str):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID 格式")
    
    if not event:
        raise HTTPException(404, "活動不存在")
    
    if not event.get("is_active", True):
        raise HTTPException(403, "活動已結束")
    
    # 只返回參與者需要的資訊（不驗證 owner）
    photos = event.get("photos", [])
    return {
        "name": event["name"],
        "created_at": event.get("created_at", "未知"),
        "photos_count": len(photos),
        "logo_url": event.get("logo_url", "https://via.placeholder.com/180x180/3182ce/ffffff?text=活動Logo"),
        "event_url": f"http://localhost:3000/event/{event_id}",
        "is_active": event.get("is_active", True)
    }

# 刪除單張照片
@app.delete("/delete_photo/{event_id}/{photo_index}")
async def delete_photo(event_id: str, photo_index: int, current_user: dict = Depends(get_current_user)):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID")
    
    if not event or event["owner"] != current_user["username"]:
        raise HTTPException(403, "無權限")
    
    photos = event.get("photos", [])
    if photo_index < 0 or photo_index >= len(photos):
        raise HTTPException(400, "照片索引無效")
    
    photos.pop(photo_index)
    events.update_one({"_id": ObjectId(event_id)}, {"$set": {"photos": photos}})
    return {"message": "照片已刪除"}

# 結束活動
@app.post("/close_event/{event_id}")
async def close_event(event_id: str, current_user: dict = Depends(get_current_user)):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID")
    
    if not event or event["owner"] != current_user["username"]:
        raise HTTPException(403, "無權限")
    
    events.update_one({"_id": ObjectId(event_id)}, {"$set": {"is_active": False}})
    return {"message": "活動已結束"}

# 刪除所有照片
@app.delete("/delete_all_photos/{event_id}")
async def delete_all_photos(event_id: str, current_user: dict = Depends(get_current_user)):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID")
    
    if not event or event["owner"] != current_user["username"]:
        raise HTTPException(403, "無權限")
    
    events.update_one({"_id": ObjectId(event_id)}, {"$set": {"photos": [], "embeddings": []}})
    return {"message": "所有照片已刪除"}

# 測試寫入
@app.get("/test_write")
async def test_write():
    try:
        users.insert_one({"test": "hello", "time": datetime.now().isoformat()})
        return {"message": "寫入成功，請去 Atlas 查看 photo_db.users"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload_logo/{event_id}")
async def upload_logo(
    event_id: str,
    logo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        event = events.find_one({"_id": ObjectId(event_id)})
    except:
        raise HTTPException(400, "無效的活動 ID 格式")
    
    if not event:
        raise HTTPException(404, "活動不存在")
    
    if event["owner"] != current_user["username"]:
        raise HTTPException(403, "無權限上傳 Logo")
    
    # 上傳到 Cloudinary（建議放 event_logos 資料夾）
    content = await logo.read()
    upload_result = cloudinary.uploader.upload(
        content,
        folder="event_logos",  # 可選，方便分類
        resource_type="image"
    )
    logo_url = upload_result["secure_url"]
    
    # 更新 event 文件的 logo_url 欄位
    events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": {"logo_url": logo_url}}
    )
    
    return {
        "message": "Logo 上傳成功",
        "logo_url": logo_url
    }