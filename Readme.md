 # Lunar Image Registration Software

 ## 1. Goal

 Build a lunar image registration software system that aligns multi-sensor images and outputs:

 - Matched image
 - Aligned (warped) image
 - RMSE
 - Inlier ratio
 - Compute time

 **Synthetic demo output:** The registration endpoint always returns generated synthetic
 correspondences and illustrative metrics; there is no UI or API switch to measured
 correspondence results. These values are not detected image matches or scientific
 measurements. The source/reference images are still processed to produce the displayed
 image-derived surface blend. Synthetic values are labeled in the result panel and match
 visualization.

 Match visualizations and coordinate listings are paginated in batches of 100. Coordinate
 rows use sequential serial numbers within the confidence-sorted, filtered result set.

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

 ## 6. API Design

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

 ## 7. Guiding Principle

 > We are engineers. We can do anything.
