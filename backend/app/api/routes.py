import logging
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config import settings
from app.schemas.response import MatchResponse, ErrorDetailResponse
from app.services.pipeline import run_pipeline
from app.services.logs import record_execution
from app.core.utils import save_upload_file_tmp, cleanup_files

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/match",
    response_model=MatchResponse,
    responses={
        400: {"model": ErrorDetailResponse, "description": "Invalid file format or missing parameter"},
        413: {"model": ErrorDetailResponse, "description": "File payload exceeds max allowed limit"},
        500: {"model": ErrorDetailResponse, "description": "Server processing error"},
    },
    summary="Register and Align Lunar Surface Images",
    description=(
        "Accepts source and reference images (.png) with corresponding mission metadata "
        "(.xml), returns an image-derived surface blend and clearly labeled synthetic "
        "correspondences and quality metrics. Correspondence metrics are illustrative, "
        "not measured scientific results."
    )
)
async def match_images(
    source_img: UploadFile = File(..., description="Source lunar image (.png)"),
    source_xml: UploadFile = File(..., description="Source mission metadata (.xml)"),
    ref_img: UploadFile = File(..., description="Reference lunar image (.png)"),
    ref_xml: UploadFile = File(..., description="Reference mission metadata (.xml)"),
):
    saved_paths = []
    started_at = time.perf_counter()

    try:
        logger.info(f"Received alignment request: Source='{source_img.filename}', Ref='{ref_img.filename}'")

        # 1. Save uploaded files temporarily to disk with extension validation
        src_img_path = await save_upload_file_tmp(source_img, settings.ALLOWED_IMAGE_EXTENSIONS)
        saved_paths.append(src_img_path)

        src_xml_path = await save_upload_file_tmp(source_xml, settings.ALLOWED_XML_EXTENSIONS)
        saved_paths.append(src_xml_path)

        ref_img_path = await save_upload_file_tmp(ref_img, settings.ALLOWED_IMAGE_EXTENSIONS)
        saved_paths.append(ref_img_path)

        ref_xml_path = await save_upload_file_tmp(ref_xml, settings.ALLOWED_XML_EXTENSIONS)
        saved_paths.append(ref_xml_path)

        # 2. Execute computational pipeline
        result = run_pipeline(
            src_img_path=src_img_path,
            src_xml_path=src_xml_path,
            ref_img_path=ref_img_path,
            ref_xml_path=ref_xml_path,
            use_synthetic_demo=True,
        )

        record_execution(
            status="SUCCESS",
            duration=time.perf_counter() - started_at,
            rmse=(
                result.get("rmse")
                if result["result_mode"] == "measured"
                else None
            ),
        )
        return MatchResponse(**result)

    except HTTPException as http_ex:
        record_execution(
            status="FAILED",
            duration=time.perf_counter() - started_at,
            error=str(http_ex.detail),
        )
        raise http_ex
    except Exception as e:
        record_execution(
            status="FAILED",
            duration=time.perf_counter() - started_at,
            error=str(e),
        )
        logger.error(f"Unhandled error in image matching endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the registration pipeline: {str(e)}"
        )
    finally:
        # 3. Always clean up temporary uploaded files
        cleanup_files(saved_paths)