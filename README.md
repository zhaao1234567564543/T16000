Here is the draft README.md file tailored for your MATLAB source code, translated into English. This code is clearly an advanced algorithm designed for digital core construction, reservoir microstructure analysis, or 3D reconstruction of porous media. You can directly copy and paste the following Markdown content into your code repository.

Comprehensive Morphology and Spatial Feature Optimization Reconstruction Model for 3D Porous Media
This project provides an advanced, MATLAB-based 3D porous media (e.g., digital core) reconstruction algorithm. By combining Markov Chain Monte Carlo (MCMC) methods with Simulated Annealing strategies, this program accurately reconstructs 3D models while precisely matching the spatial topological structures and microscopic morphological features of the original data.

📌 Core Features
Dual-Stage Adaptive Optimization: * Morphology-Preserving Phase: Prioritizes aligning the sphericity, elongation, volume distribution, and connectivity of pore clusters.

Spatial Matching Phase: Focuses on optimizing large-scale spatial statistical features such as the Two-Point Correlation (TPC) function, Chord Length Distribution (CLD), and anisotropy.

Multi-Scale Feature Extraction & Anchoring: Supports the calculation of Pore Size Distribution (PSD), Minkowski functionals (including integral mean curvature), and multi-scale spatial spectra, ensuring structural consistency across different resolutions.

Intelligent Cluster Management: Incorporates adaptive watershed splitting, thin-neck repair, isolated matrix cleanup, and density-based porosity fine-tuning to prevent over-fragmentation or abnormal aggregation of pores.

Adaptive Parallel Computing: Automatically detects and enables parpool for multi-threaded batch move evaluation and feature calculation, significantly improving MCMC iteration efficiency.

📂 File Dependencies & I/O
Input Files
DATA1.raw: The original 3D grayscale/binary model data file (format must be 8-bit unsigned integer uint8). The default reading dimension is 150 × 150 × 40, but the program supports custom dimensions at runtime and will automatically perform interpolation scaling.

Output Files
originalDataModel.raw: The original binarized reference model after threshold processing (default pore threshold is 120).

newModel_comprehensive_optimized_X_Y_Z.raw: The final 3D binarized reconstructed model after optimization (X_Y_Z represents the output dimensions).

🚀 Quick Start
Prepare Environment: Ensure MATLAB is installed, along with the Image Processing Toolbox and the Parallel Computing Toolbox.

Place Data: Name your target original core scan data or grayscale model DATA1.raw and place it in the same directory as the script.

Run Main Program: Enter or directly run xing20_3 in the MATLAB command window.

Interactive Configuration: Upon startup, the program will guide you through the following configurations via the command line:

Choose whether to use the original data model size or custom reconstruction dimensions.

Enter the target porosity (a prompt will display the original model's porosity as a reference).

Select the spatial feature target (keeping original consistency is recommended, or use gradient inference).

Choose whether to preserve small pore features.

Set pore cluster statistical targets (e.g., cluster count and size distribution).

⚙️ Algorithm Workflow Overview
Feature Extraction: Reads and binarizes the original model, calculating its comprehensive features (extracting morphological features, spatial features, and computing the porosity gradient model).

Initial Model Construction: Synthesizes a base field via a directional Gaussian random field and embeds representative pore cluster templates based on target porosity, correlation length, and anisotropy.

MCMC Iterative Optimization: * Generates batch local/global candidate moves.

Calculates a comprehensive Energy Function encompassing porosity, morphology, spatial features, connectivity, and structural coherence.

Accepts or rejects moves using the Metropolis criterion and executes dynamic simulated annealing cooling.

Comprehensive Post-Processing: Performs enhanced denoising, morphological reconstruction, internal isolated matrix cleanup, and smart porosity fine-tuning to output the final high-precision digital model.

📊 Performance Monitoring & Visualization
During the MCMC iteration process (default max 1500 iterations), the program will:

Adaptively adjust temperature and weights every 100 iterations.

Print optimization progress, energy status, and matching metrics to the console every 500 iterations.

Generate 3D orthogonal slice views and 3D isosurface rendering plots (Figure 300) every 1000 iterations to visually display the pore network evolution.

Output a detailed statistical comparison report (pore count, size distribution, connectivity, etc.) and plot energy descent and matching evolution curves (Figure 400) once the optimization is complete.
