"""
Supabase Storage helpers for CV files.

Uses the singleton Supabase service-role client from db.supabase.
Never initialize a new Supabase client here.
"""

import logging
import uuid
from fastapi import HTTPException
from db.supabase import supabase

logger = logging.getLogger(__name__)
BUCKET_NAME = "cvs"


async def _ensure_bucket() -> None:
    """Create the storage bucket if it doesn't already exist."""
    try:
        await supabase.storage.create_bucket(
            BUCKET_NAME, options={"public": True}
        )
        logger.info("Created storage bucket '%s'", BUCKET_NAME)
    except Exception as exc:
        # "already exists" is fine — anything else we just log and move on
        if "already exists" not in str(exc).lower() and "Duplicate" not in str(exc):
            logger.warning("Bucket ensure check: %s", exc)


async def upload_cv_file(file_bytes: bytes, filename: str, user_id: str, content_type: str) -> str:
    """
    Upload the original CV file to Supabase Storage.

    Returns:
        Public URL string (or signed URL depending on bucket policy).

    Raises:
        HTTPException 500 if upload fails.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
    storage_path = f"{user_id}/{uuid.uuid4()}.{extension}"

    try:
        await supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type or "application/octet-stream"},
        )
        file_url: str = await supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
        return file_url
    except Exception as exc:
        # If bucket doesn't exist, create it and retry once
        if "not found" in str(exc).lower() or "404" in str(exc):
            await _ensure_bucket()
            try:
                await supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=file_bytes,
                    file_options={"content-type": content_type or "application/octet-stream"},
                )
                file_url = await supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
                return file_url
            except Exception as retry_exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"CV storage upload failed after bucket creation: {retry_exc}"
                )
        raise HTTPException(
            status_code=500,
            detail=f"CV storage upload failed: {exc}"
        )

