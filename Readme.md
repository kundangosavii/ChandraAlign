 # Lunar Image Registration Software

 ## 1. Goal

 Build a lunar image registration software system that aligns multi-sensor images and outputs:

 - Matched image
 - Aligned (warped) image
 - RMSE
 - Inlier ratio
 - Compute time

 ## 2. System Type

 Full-stack machine learning system:

 ```text
 Frontend -> FastAPI backend -> CV pipeline engine
 ```

 ## 3. Requirements

 ### 3.1 Functional Requirements

 #### F1 - Image Upload

 The user can upload:

 - Source image
 - Reference image

 #### F2 - Execute Registration

 Trigger the following pipeline:

 ```text
 Load -> Preprocess -> Match -> RANSAC -> Warp -> Metrics
 ```

 #### F3 - Output Results

 Return a response containing:

 ```json
 {
	 "match_image": "...",
	 "aligned_image": "...",
	 "rmse": 0.0,
	 "inlier_ratio": 0.0,
	 "inlier_count": 0,
	 "compute_time": 0.0
 }
 ```

 #### F4 - Visualization

 The frontend should display:

 - Feature-match lines
 - Overlay showing the alignment

 ### 3.2 Non-Functional Requirements

 #### N1 - Performance

 - Handle large images with resolutions of 10,000 pixels or more.
 - Avoid memory crashes, including the previously encountered 8 GB file-size issue.

 #### N2 - Accuracy

 Target:

 - RMSE below 1 pixel
 - Inlier ratio above 70%

 #### N3 - Modularity

 Each pipeline stage must be independent. This is critical for debugging.

 #### N4 - Scalability

 The pipeline should be reusable in:

 - CLI applications
 - Backend services
 - Desktop applications

 ## 4. Proposed System Architecture

 ```text
 app/
 |- main.py
 |- api/
 |  `- routes.py
 |- services/
 |  `- pipeline.py
 |- core/
 |  |- loader.py
 |  |- preprocess.py
 |  |- features.py
 |  |- matcher.py
 |  |- geometry.py
 |  |- refinement.py
 |  `- visualization.py
 `- schemas/
		`- request_response.py
 ```

 > **Note:** This architecture was generated as an initial reference and is not final. It may be modified after the team connects and reviews the implementation needs.

 The proposed separation is:

 - `routes`: HTTP/API handling
 - `services`: Application and pipeline orchestration logic
 - `core`: Computer-vision pipeline modules

 ## 5. Data Flow

 ```text
 Frontend
	 -> POST /image/match
	 -> FastAPI route
	 -> Pipeline service
	 -> Core modules:
			- Loader
			- Preprocess
			- Features
			- Matcher
			- RANSAC
			- Homography
			- Warp
			- Metrics
			- Visualization
	 -> Response JSON
	 -> Frontend UI
 ```

 ## 6. Module Breakdown and Team Assignment

 The assignments below are primarily for research and may evolve during implementation.

 ### Jahanvi - Image Loader

 Tasks:

 - Load PNG images
 - Resize large images
 - Validate input

 ### Kundan - Preprocessing

 Tasks:

 - Normalization using CLAHE
 - Denoising
 - Contrast enhancement
 - Research and evaluate additional preprocessing techniques

 ### Vishal - Feature Extraction

 Tasks:

 - ORB for the MVP
 - SuperPoint for the advanced version
 - Uniform feature distribution using a grid, ANMS, or another suitable approach
 - Select an appropriate algorithm based on research

 ### Vishal and Kundan - Matching

 Tasks:

 - BFMatcher for the MVP
 - SuperGlue integration for the advanced version
 - Select an appropriate matching algorithm based on research

 ### Kundan and Vishal - Geometry

 This is a critical module.

 Tasks:

 - RANSAC
 - Homography estimation
 - Image warping

 ### Kundan - Metrics

 Tasks:

 - RMSE calculation on inliers
 - Inlier ratio calculation
 - Compute-time measurement

 ### Kundan - Backend Integration

 Tasks:

 - FastAPI endpoint
 - Request and response schemas
 - Base64 conversion

 ### Ayush - Frontend

 Tasks:

 - Upload interface
 - Results display
 - Metrics display

 ## 7. API Design

 ### Endpoint

 ```http
 POST /image/match
 ```

 ### Request

 Multipart form-data containing:

 - `source`: `UploadFile`
 - `reference`: `UploadFile`

 ### Response

 ```json
 {
	 "match_image": "...",
	 "aligned_image": "...",
	 "rmse": 0.85,
	 "inlier_ratio": 0.78,
	 "inlier_count": 120,
	 "compute_time": 1.45
 }
 ```

 ## 8. Guiding Principle

 > We are engineers. We can do anything.
