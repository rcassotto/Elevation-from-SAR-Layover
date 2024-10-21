**Python Script to generate building heights from Sentinel-1 C-Band SAR Layover.** 

It is recommended to execute from within a dedicated python environment, such as conda. 

**<u>Python dependencies</u>**
- rasterio
- numpy
- argarse


File Description:
- SAR_Layover_revXX.py: main python script to perform the calculations
- input_SAR_layover_test.txt: text file containing the input parameters/pathways to relevant files. Users should modify the input statements for their application.  
- inputs.zip: zip file containing the relevant Sentinel-1 RTC images necessary to run the script. 

Inputs:
- Building outlines with Bing-dervived building outlines
- Range Terrain Corrected (RTC) Sigma0 images; the samples were generated using the Sentinel-1 Toolbox (SNAP)
- SNAP generated incidence angle files corresponding to the RTC Sigma0 images.
- SNAP generated '*dim' text file containing critical metadata.

Outputs:
- Geotiff for each Sigma0 image in the input directory.  The geotiffs contain the buildling outlines set at the calcualted buidling height. One Geotiff will be created with all the building outlines and elevations that are included in the building outlines directory. 


 To execute:
 1) Create a dedicated conda environment with the necessary dependencies, then activate that environment.
 2) Unzip the inputs.zip file
 3) Update the input_SAR_layover_test.txt file to relect the paths of the Sigma0 files ("sigma_file_loc"), the output directory ("out_dir"), and building outline directory ("building_footprint_directory").
 4) Execute the following command:  **Python3 SAR_Layver_revXX.py inout_SAR_layover_test.txt**
