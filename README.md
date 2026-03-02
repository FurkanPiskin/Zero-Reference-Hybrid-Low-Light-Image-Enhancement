## 🧠 Deep Learning Mechanism: LCE-Net (Curve Estimation)
Instead of directly generating output pixels like traditional Heavy GANs, our lightweight network acts as an intelligent parameter estimator. This makes it an ideal real-time preprocessing step for high-level computer vision tasks (e.g., object detection).

* **Unsupervised & Zero-Reference:** Trained on unpaired data, eliminating the dependency on perfectly paired extreme-low-light/bright datasets.
* **Pixel-Wise Curve Parameters:** The network predicts spatial parameter maps (`t` and `b` matrices) that have the exact same resolution as the input image.
* **Iterative Rational Cubic Curves:** These predicted parameters are applied to a Rational Cubic Curve formula iteratively (across 4 stages). 
* **Local Adaptation:** By using spatial parameter maps rather than global adjustments, the model adapts locally to varying illumination—boosting dark regions while strictly preventing overexposure in already bright areas, all with minimal color distortion.

* ## 🔧 Classic Post-Processing: Adaptive Gamma & Domain Transform

* ### 🔧 Spatial Gamma Refinement & Noise Suppression
While the primary LCE-Net successfully boosts the overall brightness, dark regions inherently contain hidden sensor noise that becomes aggressively visible once illuminated. To suppress this noise without blurring structural details, the pipeline executes a three-step spatial refinement process:

1. **Pixel-Wise Gamma Map Computation:** Instead of relying on a single global $\gamma$ parameter, the system calculates a localized, spatial gamma map. This is achieved by taking the logarithmic ratio between the original extreme-low-light image and the target enhanced estimation.
2. **Domain Transform Recursive Filtering:** Applying smoothing directly to an image destroys high-frequency textures (like hair or edge lines). To bypass this, we apply a highly efficient, edge-preserving recursive filter *strictly to the gamma map itself*. This smooths the overall illumination field while keeping the physical textures of the image completely intact.
3. **Total Variation (TV) Denoising:** Once the optimized gamma map is applied to the original image, a Chambolle Total Variation filter is integrated to clean up any residual low-amplitude chroma noise, achieving a perfect balance between noise suppression and detail preservation.

### 🧮 Autonomous "Key Value" Illumination Mapping
To strictly prevent the overexposure of already bright areas (e.g., streetlights in a dark scene), the pipeline avoids linear brightening. Instead, it calculates a logarithmic **Image Key Value**. Based on this severity score, the system constructs a dynamic **Circular Mapping Arc** strictly on the Luminance (Y) channel. This algorithm mathematically bends dark pixels upward while heavily compressing bright pixels, ensuring a perfectly balanced dynamic range without manual thresholding.

4. ### 🎨 Natural Color Enhancement & Local Contrast (NECI)
Brightening a low-light image in the standard RGB color space inevitably leads to severe color imbalances and unnatural light casts. To counteract this, the pipeline integrates a Natural Enhancement of Color Image (NECI) approach, shifting operations to decouple luminance (brightness) from chrominance (color).

* **Color Space Decoupling (CIELCh / LAB):** Enhancements are strictly applied to the luminance channel first. This prevents the independent RGB channel scaling that typically destroys an image's natural color balance.
* **Global Mapping & Modified Retinex:** The system corrects global light casts using content-dependent mapping, followed by a modified Retinex filter. This specifically boosts local contrast, giving depth to mid-tones and shadows without clipping highlights.
* **Luminance-Weighted Chrominance:** To restore color vibrancy, the algorithm calculates a weighting map based strictly on how much the luminance was enhanced. This weight is then applied to the chrominance channels, ensuring that colors scale proportionally and naturally with the new brightness.
* **High-Frequency Texture Boosting (Cortex Transform):** Following histogram rescaling (normalization), the pipeline isolates high-frequency components. This step physically restores fine structural textures (e.g., wood grain, fur) that might have been softened during the initial AI estimation and denoising stages.
