3DGS Automation Pipeline
Contents
Overview	11
Pipeline Workflow	50
Folder Structure	77
Installation	91
Running the Pipeline	112
Acknowledgement:	157


Overview
This pipeline automates the generation of a semantically labelled 3D Gaussian Splat (3DGS) model from a set of input photographs. The workflow combines photogrammetric reconstruction, Gaussian Splatting, semantic segmentation and semantic lifting into a single process chain.
The pipeline consists of three main stages:

Stage 1 – COLMAP Reconstruction
The first stage prepares the input images and performs a Structure-from-Motion (SfM) reconstruction using COLMAP.
This stage:
•	Resizes and standardizes the input images
•	Extracts image features
•	Matches features between images
•	Estimates camera poses
•	Generates a sparse 3D reconstruction
The resulting camera model and sparse reconstruction form the basis for the Gaussian Splatting stage.


Stage 2 – Brush Gaussian Splatting
The second stage converts the COLMAP reconstruction into a Brush dataset and trains a 3D Gaussian Splat representation.
This stage:
•	Creates a Brush-compatible dataset structure
•	Uses the COLMAP camera calibration and sparse reconstruction
•	Trains a Gaussian Splat model
•	Exports a viewer-ready PLY file
The resulting PLY file contains the reconstructed Gaussian Splats that represent the scene geometry and appearance.

Stage 3 – Semantic Segmentation and 3D Lifting
The final stage adds semantic information to the Gaussian Splats.
First, a SegFormer model generates semantic masks for each image. These masks are reduced to a compact indoor class scheme consisting of:
•	Floor
•	Wall
•	Ceiling
•	Door / Window
•	Unknown
The semantic labels are then projected back onto the reconstructed Gaussian Splats using the COLMAP camera poses.
Each splat receives votes from all camera views in which it is visible. A majority-vote strategy combined with confidence filtering is used to assign the final semantic class.
The final output is a semantically colored Gaussian Splat model that can be visualized in compatible viewers.




Pipeline Workflow
Images
  │
  ▼
COLMAP
  │
  ▼
Brush (3DGS)
  │
  ├─────────────┐
  │             │
  ▼             ▼
3D Splats   SegFormer
                │
                ▼
        Segmented Images
                │
                ▼
        Semantic Lifting
                │
                ▼
         Majority Voting
                │
                ▼
      Labelled 3DGS Model
      
 
Folder Structure
Project/

Script_S1_Colmap.py
Script_S1_Colmap_GPU.py
Script_S2_Brush.py
Script_S3_Semantics.py

Input/
Images/

Output/


Installation
Step 1 – Install Prerequisites
Install the following software before running the pipeline:
•	Python 3.11
•	NVIDIA Drivers
•	CUDA Toolkit
•	Microsoft Visual C++ Redistributable
•	COLMAP
•	Brush
A detailed dependency list can be found in the accompanying dependency document.

Step 2 – Create a Python Environment
Create a virtual environment:
python -m venv .venv
Activate the environment:
.venv\Scripts\activate
Install PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Install the remaining dependencies:
pip install numpy pillow transformers

Running the Pipeline
Stage 1 – COLMAP Reconstruction
Place all your images inside:
Input/Images/
Run either the CPU or GPU version of the S1 Colmap script.
•	CPU version: python Script_S1_Colmap.py
•	GPU version: python Script_S1_Colmap_GPU.py
Outputs:
•	Resized images
•	COLMAP database
•	Sparse reconstruction
•	Camera calibration

Stage 2 – Gaussian Splatting
Run python Script_S2_Brush.py
This stage:
•	Creates the Brush dataset
•	Trains the Gaussian Splats
•	Exports a PLY file
Output:
Output/viewer_ready/

Stage 3 – Semantic Labelling
Run python Script_S3_Semantics.py
This stage:
•	Generates semantic masks
•	Generates confidence maps
•	Lifts semantic labels to 3D
•	Exports a semantic PLY model
Outputs include:
•	Semantic masks
•	Confidence maps
•	Semantic statistics
•	Semantic Gaussian Splat PLY

Current Limitations
This prototype focuses on structural indoor elements only.
The following classes are currently not included:
•	Signage
•	Landmarks
•	Furniture-specific classes
•	Wayfinding objects
•	Vertical transition elements
The goal of the current implementation is to demonstrate the feasibility of an automated semantic 3DGS workflow before extending the class scheme to more complex wayfinding-related objects.

Acknowledgement: 
Schönberger, J.L. (2026) COLMAP: Structure-from-Motion and Multi-View Stereo [Computer software]. Available at: https://github.com/colmap/colmap 
Brussee, A. (2026) Brush: 3D reconstruction for all [Computer software]. Available at: https://github.com/ArthurBrussee/brush 
Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M. and Luo, P. (2021) SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. arXiv:2105.15203. Available at: https://arxiv.org/abs/2105.15203

