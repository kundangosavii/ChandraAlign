from typing import Any, Literal

from pydantic import BaseModel, Field

class MatchResponse(BaseModel):
    match_image: str = Field(..., description="Base64 encoded string of feature keypoint match visualization")
    match_image_pages: list[str] = Field(
        default_factory=list,
        description="Paged match visualizations with at most 100 correspondences per image",
    )
    aligned_image: str = Field(..., description="Base64 encoded image-derived source/reference blend")
    rmse: float = Field(..., description="Illustrative synthetic demo RMSE; not measured from image matches")
    inlier_ratio: float = Field(..., description="Illustrative synthetic demo inlier ratio; not measured")
    compute_time: float = Field(..., description="Pipeline processing duration in seconds")
    result_mode: Literal["measured", "synthetic_demo"] = Field(
        default="synthetic_demo",
        description="Whether correspondence metrics are measured or synthetic demo data; the HTTP endpoint returns synthetic demo data",
    )
    aligned_image_status: Literal[
        "registered",
        "preprocessed_source_not_registered",
    ] = Field(
        default="registered",
        description="Whether the returned surface image has actually been registered",
    )
    match_details: dict[str, Any] | None = Field(
        default=None,
        description="Correspondence details; synthetic values in the HTTP endpoint response",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "match_image": "iVBORw0KGgoAAAANSUhEUgAA...",
                "aligned_image": "iVBORw0KGgoAAAANSUhEUgAA...",
                "rmse": 0.1245,
                "inlier_ratio": 0.842,
                "compute_time": 1.432
            }
        }

class ErrorDetailResponse(BaseModel):
    detail: str