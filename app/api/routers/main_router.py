from fastapi import APIRouter
from app.api.routers.user import user_router
from app.api.endpoints.ingestion import router as ingestion_router

router = APIRouter()

router.include_router(user_router)
router.include_router(ingestion_router)


