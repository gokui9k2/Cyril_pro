This project builds a pipeline to improve breast cancer detection by combining pixel data with tabular clinical data patient weight, height, age.

## 🎯 Project Goal

Standard analysis often relies solely on images. Our approach hypothesizes that fusing **imaging features** with **clinical metadata** will yield higher predictive accuracy. To achieve this, we need high quality, artifact free 3D volumes.

## Context: DCE-MRI

Dynamic Contrast-Enhanced (DCE) MRI is the **clinical baseline for breast cancer detection**.

It captures the uptake of a contrast agent over time to analyze tissue perfusion. However, this time-series approach faces two major data challenges:

1. **Patient Movement:** Movement between frames creates artifacts that must be corrected (Registration).
2. **Sparse Data:** MRI volumes contain a massive amount of "black background" pixels that carry no information and waste computational power (Cropping).

## Microservices Structure

The project is divided into distinct folders, each containerized (Docker) to run as an independent job on Google Cloud Run:

* **`ingestion/`**
Detects valid DCE sequences from raw sources and assigns a unique, anonymized ID to each patient for tracking.
* **`preprocessing/`**
Cleaning raw metadata and converting complex DICOM series into standard NIfTI 3D volumes.
* **`processing/`**
The core logic. I am currently testing different preprocessing methods to find the best configuration, specifically benchmarking **image reduction** (removing useless black pixels) and **registration algorithms** to fix patient movement without destroying image quality.
* **`analysis/`**
Notebooks and reports used to assess the quality of the registration and cropping strategies.

## Infrastructure

* **Platform:** Google Cloud Platform (GCP)
* **Orchestration:** Google Cloud Workflows