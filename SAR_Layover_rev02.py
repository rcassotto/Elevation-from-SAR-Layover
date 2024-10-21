#### Author: Ryan Cassotto, CIRES

import numpy as np
import rasterio
#import matplotlib.pyplot as plt
import glob
import os
import argparse,ast

sigma0_building_buffer_m = 400 # buffer in meters, to add on all sides of building. 
intensity_threshold = 0.3
max_building_height=400; # assumed starting building height in meters
increment_step=1#0.2; # increment size for shifting pixels in footprint / sigma0 alignment phase
layover_threshold_prct=50;



# making the output directory
def mk_outdirectory(outdirectory):
    if not os.path.exists(outdirectory):
        print("Making output directory: ", outdirectory)
        os.makedirs(outdirectory)
    return

## Get azimuth and orbit information
def get_azimuth_and_orbit_info(meta_file_and_path):    
        ## Search meta file for Azimuth angle (i.e. platformHeading)
        searchfile = open(meta_file_and_path, "r")
        for line in searchfile:
            if "platformHeading" in line: 
                tmpStr = line.split(">")
                tmpStr1 = tmpStr[1]
                tmpStr2 = tmpStr1.split("<")
                azimuth = float(tmpStr2[0]) 
        searchfile.close()

        ## Search meta file for orbit direction (ie. Ascending or Descending)
        searchfile = open(meta_file_and_path, "r")
        for line in searchfile:
            if "ASCENDING or DESCENDING" in line: 
                tmpStr = line.split(">")
                tmpStr1 = tmpStr[1]
                tmpStr2 = tmpStr1.split("<")
                orbit_direction = tmpStr2[0] 
        searchfile.close() 
        
        if orbit_direction == "DESCENDING":
             shift_direction = 'ESE'
        elif orbit_direction == "ASCENDING":
            shift_direction = 'WSW'
        return azimuth, orbit_direction, shift_direction

## Get image georeference information (e.g. lat, lon, pixel size, etc)
def get_image_geo_info(in_sigma_file):
        image_data = rasterio.open(in_sigma_file)
        lon = (image_data.bounds.left, image_data.bounds.right)
        lat = (image_data.bounds.top, image_data.bounds.bottom)
        minLat = np.min(lat)
        maxLat = np.max(lat)
        minLon = np.min(lon)
        maxLon = np.max(lon)
        medianLat = np.median(lat)
 
        gt = image_data.transform
        pixel_size_lon_deg = gt[0]
        pixel_size_lat_deg = gt[4]
        pixel_size_lat_m = gt[4]*111e3
        pixel_size_lon_m = gt[0]*111e3*np.cos(np.radians(medianLat))
        lon_arr = np.arange(minLon+0.5*pixel_size_lon_deg, maxLon, pixel_size_lon_deg, dtype=float)# lon array with 0.5 chip offset
        lat_arr = np.arange(maxLat+0.5*pixel_size_lat_deg, minLat, pixel_size_lat_deg, dtype=float)
        
        return lat_arr,lon_arr,pixel_size_lat_m,pixel_size_lon_m,pixel_size_lat_deg,pixel_size_lon_deg,image_data
        
    
  ## Function to subset Sigma0 and Incidence Angle images  
def subset_sigma0_and_incidence_images(VV_image, incidence_image, lon_arr, lat_arr, pixel_size_lon_deg, pixel_size_lat_deg, bldg_img_wBuff, b_lat_arr_wBuff, b_lon_arr_wBuff):
        bldg_min_lon = np.min(b_lon_arr_wBuff)
        bldg_max_lon = np.max(b_lon_arr_wBuff)
        bldg_min_lat = np.min(b_lat_arr_wBuff)  ## bounds is minx miny maxx maxy - original
        bldg_max_lat = np.max(b_lat_arr_wBuff)  # original
                
    
    ### Subset Sigma0 for current buildling outline - this could be modularized
        ix_sigma_start=np.argmin(np.abs(lon_arr - bldg_min_lon)) # index into original image 
        ix_sigma_stop=np.argmin(np.abs(lon_arr - bldg_max_lon)) 
        iy_sigma_start=np.argmin(np.abs(lat_arr - bldg_min_lat)) 
        iy_sigma_stop=np.argmin(np.abs(lat_arr - bldg_max_lat)) 
        
        if iy_sigma_start > iy_sigma_stop:
            tmpVal = iy_sigma_start
            iy_sigma_start = iy_sigma_stop
            iy_sigma_stop = tmpVal
        
        ## Sigm0_sub, _lon_sub, and _lat_sub are numpy arrays with the subset of Sigma0/lat/lon around the building, plus a uniform buffer
        sigma0_sub = VV_image[iy_sigma_start : iy_sigma_stop, ix_sigma_start:ix_sigma_stop]        
        sigma0_lon_sub = lon_arr[ix_sigma_start : ix_sigma_stop]
        sigma0_lat_sub = lat_arr[iy_sigma_start : iy_sigma_stop]
            
        ## Subset incidence angle using the same values above
        in_incidence_sub = incidence_image[iy_sigma_start : iy_sigma_stop, ix_sigma_start:ix_sigma_stop]        
        
        ## Account for mismatch < 1 pixel (i.e. within spatial resolution error of a single pixel)
        lat_offset = b_lat_arr_wBuff[0] - sigma0_lat_sub[0]
        lon_offset = b_lon_arr_wBuff[0] - sigma0_lon_sub[0]

        if np.abs(lon_offset) < np.abs(pixel_size_lon_deg):
            sigma0_lon_sub = sigma0_lon_sub + lon_offset
                
        if np.abs(lat_offset) < np.abs(pixel_size_lat_deg):
            sigma0_lat_sub = sigma0_lat_sub + lat_offset   # original

        col_mismatch = bldg_img_wBuff.shape[0] - sigma0_sub.shape[0]
        row_mismatch = bldg_img_wBuff.shape[1] - sigma0_sub.shape[1] 
        
        if (col_mismatch == 1):
            B = np.pad(sigma0_sub,((0, col_mismatch),(0, 0)),'constant', constant_values=0);  # pad end of columns with a zero
            sigma0_sub = B
            C = np.pad(in_incidence_sub,((0, col_mismatch),(0, 0)),'constant', constant_values=0);  # pad end of columns with a zero
            in_incidence_sub = C
            del B, C
        
        if (row_mismatch == 1):
            B = np.pad(sigma0_sub,((0, 0),(0, row_mismatch)),'constant', constant_values=0);  # pad end of rows with a zero
            sigma0_sub = B;
            C = np.pad(in_incidence_sub,((0, 0),(0, row_mismatch)),'constant', constant_values=0);  # pad end of rows with a zero
            in_incidence_sub = C
            del B, C
#        print("sigma0 sub size: ", sigma0_sub.shape)
        return sigma0_sub, in_incidence_sub, sigma0_lon_sub, sigma0_lat_sub, ix_sigma_start, ix_sigma_stop, iy_sigma_start, iy_sigma_stop


## Function to add buffer around building outline
def add_buffer_around_building(sigma0_building_buffer_m, b_pixel_size_lon_m, bldg_img, b_lon_arr, b_lat_arr, b_pixel_size_lon_deg, b_pixel_size_lat_deg):
        npix_buffer = int(np.ceil(sigma0_building_buffer_m / b_pixel_size_lon_m))  # calculate buffer size, in pixels
        bldg_img_wBuff = np.zeros((npix_buffer*2+bldg_img.shape[0], npix_buffer*2+bldg_img.shape[1])) # initialize bldg raster with buffer
#        print("Buffer loop: bldg_img_wBuff size  ", bldg_img_wBuff.shape)
        bldg_img_wBuff[npix_buffer : npix_buffer + bldg_img.shape[0], npix_buffer : npix_buffer + bldg_img.shape[1]]=bldg_img # put bldg values in bldg raster with buffer
#        print("Buffer loop: bldg_img_wBuff size  ", bldg_img_wBuff.shape)

        
        new_lon_start = np.min(b_lon_arr) - (npix_buffer * b_pixel_size_lon_deg)
        new_lon_stop = np.max(b_lon_arr) + (npix_buffer * b_pixel_size_lon_deg)
        new_lat_start = np.max(b_lat_arr) - (npix_buffer * b_pixel_size_lat_deg)  # original
        new_lat_stop = np.min(b_lat_arr) + (npix_buffer * b_pixel_size_lat_deg)   # original

        b_lon_arr_wBuff = np.arange(new_lon_start, new_lon_stop, b_pixel_size_lon_deg, dtype=float)
        b_lat_arr_wBuff = np.arange(new_lat_start, new_lat_stop, b_pixel_size_lat_deg, dtype=float)
        
        lat_mismatch = bldg_img_wBuff.shape[0] - len(b_lat_arr_wBuff)
        lon_mismatch = bldg_img_wBuff.shape[1] - len(b_lon_arr_wBuff)

        if lat_mismatch == 1:
            b_lat_arr_wBuff = np.append(b_lat_arr_wBuff, new_lat_stop + b_pixel_size_lat_deg)
        if lon_mismatch == 1:
            b_lon_arr_wBuff = b_lon_arr_wBuff.append(new_lon_stop + b_pixel_size_lon_deg)
                     
        return bldg_img_wBuff, b_lon_arr_wBuff, b_lat_arr_wBuff
  
## Function to calculate indices for buildling boundaries within Sigma0 and Incidence Angle input images
def calc_building_indices_for_full_sigma0_image(sigma0_lon_sub, sigma0_lat_sub, b_lon_arr, b_lat_arr):
        ### Find indices into Sigma0 sub image for building outline
        ix_building_start =  np.argmin(np.abs(sigma0_lon_sub - np.min(b_lon_arr)));
        ix_building_stop =  np.argmin(np.abs(sigma0_lon_sub - np.max(b_lon_arr)));
        iy_building_start =  np.argmin(np.abs(sigma0_lat_sub - np.min(b_lat_arr)));
        iy_building_stop =  np.argmin(np.abs(sigma0_lat_sub - np.max(b_lat_arr)));
       
        if iy_building_start > iy_building_stop:
            tmpVal = iy_building_start
            iy_building_start = iy_building_stop
            iy_building_stop = tmpVal
            del tmpVal
        iy_building_stop = iy_building_stop + 1
        ix_building_stop = ix_building_stop + 1    
        return iy_building_start, iy_building_stop, ix_building_start, ix_building_stop
    
def write_out_geotiff(infilename, infile_path, image_data, sar_layover_output_product):           
        layover_outname = infilename.replace('_Sigma0_VV_AST_B02.tif','_AST_B07.tif') # layover outfilename
        sar_product_outfilename_and_path = os.path.join(outdirectory, layover_outname)
        print("Writing SAR Layover Results to: ", sar_product_outfilename_and_path)
        profile = image_data.profile.copy()  # copy geotiff meta data from input file   
        with rasterio.open(sar_product_outfilename_and_path, 'w', **profile) as dst:
           dst.write(sar_layover_output_product,1)
    
## (DIAG only) Write metadata file for current building and footprint- diag   
def write_metafile(infile_path, infilename, nBuilding, azimuth, orbit_direction, shift_direction, height_output, cube_RMS_Height_sigma,imax_overlap):
        in_building_fname = nBuilding.split('/')[-1] # granule is a string with the original Sigma0 filename (e.g. S1A_IW_GRDH_1SDV_20150921T232856_20150921T232918_007820_00AE3B_E36F_Sigma0_VV)
        b_head_tail = os.path.split(nBuilding)
        in_building_path = b_head_tail[0]
       
        Line1 = ( "Input Sigma0 fname: " + infilename + "\n")
        Line2 = ( "Input Sigma0 Path: " + infile_path + "\n")
        Line3 = ( "Input Building fname: " + in_building_fname + "\n")
        Line4 = ( "Input Building Path: " + in_building_path + "\n")
        Line5 = ( "Azimuth Heading: " + str(azimuth) + "\n")
        Line6 = ( "Orbit Direction: " + orbit_direction + "\n")
        Line7 = ( "Layover shift direction: " + shift_direction + "\n")
        Line8 = ( "RMS Mean Height (m): " +  str(np.nanmean(height_output)) + "\n")
        Line9 = ( "Std Dev Height (m): " + str(cube_RMS_Height_sigma[imax_overlap]) + "\n")

        ##metafile_outname = ( outpath_full + '/' + in_dir_name + '_meta.txt' )
        FID = (in_building_fname.split("FID_")[1]).split("_")[0]
        sigma0_basename = infilename.replace('_Sigma0_VV_AST_B02.tif','') # infile basename
        metafile_outname = ( sigma0_basename + '_FID_' + FID + '.txt')
        metafile_outname_full = os.path.join( outdirectory, metafile_outname)  
        print('Writing metadata file: ', metafile_outname_full)
        meta_file = open (metafile_outname_full, "w")
        meta_file.write(Line1)
        meta_file.write(Line2)
        meta_file.write(Line3)
        meta_file.write(Line4)
        meta_file.write(Line5)
        meta_file.write(Line6)
        meta_file.write(Line7)
        meta_file.write(Line8)
        meta_file.write(Line9)
        meta_file.close()
        


#def Run_SAR_layover_algorithm(Sigma_files,footprint_dir, footprint_files):
def Run_SAR_layover_algorithm(Sigma_files,footprint_dir, footprint_files, outdirectory):

    for in_sigma_file in Sigma_files:  # Loop over each Sigma0 image
        
        mk_outdirectory(outdirectory)
        ## Parse infilename, basename, and path from current input file
        head_tail=os.path.split(in_sigma_file)   # partition input path from filename  
        infile_path=head_tail[0] # full infile path
        infilename = in_sigma_file.split('/')[-1] # granule is a string with the original Sigma0 filename (e.g. S1A_IW_GRDH_1SDV_20150921T232856_20150921T232918_007820_00AE3B_E36F_Sigma0_VV)
        infile_basename = infilename.replace('_Sigma0_VV_AST_B02.tif','') # infile basename
        
        ## Read in Sigma0 image
        print("Sigma0_VV file: ", in_sigma_file)
        VV_in = rasterio.open(in_sigma_file) # opens geotiff with rasterio; saves it as numpy array with M x N shape
        VV_image = VV_in.read(1) # Read band 1
        
         ## Get image corner coordinates - this could be cleaned up
        (lat_arr,lon_arr,pixel_size_lat_m,pixel_size_lon_m,pixel_size_lat_deg,pixel_size_lon_deg,image_data) = get_image_geo_info(in_sigma_file)
      
        ## Read incidence Angle File
        incidence_angle_fname = (infile_basename + '_incidenceAngleFromEllipsoid.tif')
        incidence_angle_path_and_filename = os.path.join(infile_path,incidence_angle_fname)
        incidence_image = rasterio.open(incidence_angle_path_and_filename) # opens geotiff with rasterio; saves it as numpy array with M x N shape
        incidence_image = incidence_image.read(1) # Read band 1

        ## Get scene metadata (orbit direction, platformHeading)
        try:
            metadata_filename = (infile_basename + '_OB_GBN_CAL_SP_TC.dim')  ## updated 10/5/22
            meta_file_and_path = os.path.join(infile_path,metadata_filename)
            (azimuth, orbit_direction, shift_direction) = get_azimuth_and_orbit_info(meta_file_and_path)

        except:
            metadata_filename = (infile_basename + '_OB_GBN_CAL.dim')  ## updated 10/5/22
            meta_file_and_path = os.path.join(infile_path,metadata_filename)
            (azimuth, orbit_direction, shift_direction) = get_azimuth_and_orbit_info(meta_file_and_path)

        print(".dim filename: ", metadata_filename)
        
       ## Initialize numpy array for final product, create output product name
        sar_layover_output_product = np.zeros((VV_image.shape[0], VV_image.shape[1]))  # initialize 2D output array
     
        for nBuilding in footprint_files:
            print("Building filename: ", nBuilding)

            ## Read in building outline geotiff
            bldg_in = rasterio.open(nBuilding) # opens geotiff with rasterio; saves it as numpy array with M x N shape
            bldg_img = bldg_in.read(1) # Read band 1

            ## Get building outline image parameters 
            (b_lat_arr,b_lon_arr,b_pixel_size_lat_m,b_pixel_size_lon_m,b_pixel_size_lat_deg,b_pixel_size_lon_deg, b_image_data) = get_image_geo_info(nBuilding)
            
            ## Add buffer around building outline
            (bldg_img_wBuff, b_lon_arr_wBuff, b_lat_arr_wBuff) = add_buffer_around_building(sigma0_building_buffer_m, b_pixel_size_lon_m, bldg_img, b_lon_arr, b_lat_arr, b_pixel_size_lon_deg, b_pixel_size_lat_deg)

             ## Subset Sigma0 and Incidence Angle images
            (sigma0_sub, in_incidence_sub, sigma0_lon_sub, sigma0_lat_sub, ix_sigma_start, ix_sigma_stop, iy_sigma_start, iy_sigma_stop) = subset_sigma0_and_incidence_images(VV_image, incidence_image, lon_arr, lat_arr, pixel_size_lon_deg, pixel_size_lat_deg, bldg_img_wBuff, b_lat_arr_wBuff, b_lon_arr_wBuff)

            ### Convert Sigma0 into a binary image        
            in_sigma_binary = sigma0_sub > intensity_threshold
        
            ### Calculate total and component (N &E) layover for a maximum building height increments; Establishes stopping point for algorithm
            in_incidence_sub[in_incidence_sub == 0] = np.nan # convert zeros along edge (in buffer space) to nans; else div by 0 --> Inf
            if np.isnan(in_incidence_sub).all():
                print("Incidence file contains only nans. Moving on...")
                pass
            else:
            
                Layover_total = max_building_height / np.tan(np.deg2rad(in_incidence_sub))  # total layover assuming max building height
                layover_northing = Layover_total * np.sin(np.deg2rad(azimuth))  # max northing component of layover; starting piont
                layover_easting = Layover_total * np.cos(np.deg2rad(azimuth)) # max easting component of layover; starting point
                layover_northing_mean_m = np.nanmean(layover_northing)
                layover_easting_mean_m = np.nanmean(layover_easting)
                layover_northing_mean_pixels = layover_northing_mean_m / pixel_size_lat_m
                layover_easting_mean_pixels = layover_easting_mean_m / pixel_size_lon_m
                pixel_ratio_e_to_n = np.floor(np.abs(layover_easting_mean_pixels / layover_northing_mean_pixels)) # for every n pixels in east, shift scene 1/n north

                ### set-up increment size; test for smaller increments
                northing_shift = int(increment_step); # 1/5 pixel increments
                easting_shift = int(pixel_ratio_e_to_n * northing_shift); # original
                maxIterations = int(np.min([np.floor(np.abs(layover_easting_mean_pixels)), np.floor(np.abs(layover_northing_mean_pixels) )])*1/increment_step);

                ### Initizlize cube_ arrays 
                cube_overlap = np.zeros((maxIterations, sigma0_sub.shape[0], sigma0_sub.shape[1]), dtype=float) 
                cube_binary_image_shift = np.zeros((maxIterations, sigma0_sub.shape[0], sigma0_sub.shape[1]), dtype=float)   
                cube_Height_magnitude = np.zeros((maxIterations, sigma0_sub.shape[0], sigma0_sub.shape[1]), dtype=float)     
                cube_RMS_Height = np.zeros((maxIterations, sigma0_sub.shape[0], sigma0_sub.shape[1]), dtype=float) 

                cube_percent_overlap = np.zeros(maxIterations, dtype=np.double); 
                cube_nIterations = np.zeros(maxIterations, dtype=np.double);
                cube_northing_offset_pixels = np.zeros(maxIterations, dtype=np.double); cube_easting_offset_pixels = np.zeros(maxIterations, dtype=np.double); cube_northing_offset_m = np.zeros(maxIterations, dtype=np.double);  cube_easting_offset_m = np.zeros(maxIterations, dtype=np.double);
                cube_Height_magnitude_mean = np.zeros(maxIterations, dtype=np.double);  cube_Height_magnitude_sigma = np.zeros(maxIterations, dtype=np.double);  cube_RMS_Height_mean = np.zeros(maxIterations, dtype=np.double);   cube_RMS_Height_sigma = np.zeros(maxIterations, dtype=np.double);
                
                ## Loop - increment steps to determine offsets at each iteration
                for nIterations in range(maxIterations):
                    if nIterations % 10 == 0:     # update iteration value every 10th loop
                        print('Calculating Height: iteration ', nIterations, ' of ', maxIterations)
                    
                    binary_image_shift = np.zeros((in_sigma_binary.shape[0], in_sigma_binary.shape[1]))
                    n_pixel_shift = northing_shift * nIterations + 1 #number of row pixels to add/substract
                    e_pixel_shift = easting_shift * nIterations  + 1  # number of col pixels to add/subtract
                
                    if orbit_direction == "ASCENDING":
                        ## Ascending - WSW
                        tmp_subarray = in_sigma_binary[ n_pixel_shift : -1 , 0 : -1 - e_pixel_shift] # ENE, for   ascending orbits
                        if tmp_subarray.shape[0]==0 or tmp_subarray.shape[1]==0:
                            pass  # Stops nIterations loop if iterations exceed size of subarray
                        else:
                                binary_image_shift[0 : -1 - n_pixel_shift ,  e_pixel_shift : -1] = tmp_subarray;  # This shifts the template ENE, For ascending orbits
            
                    elif orbit_direction == "DESCENDING":
                        tmp_subarray = in_sigma_binary[ n_pixel_shift : -1 , e_pixel_shift : -1 ] # WNW, for   descending orbits
                        if tmp_subarray.shape[0]==0 or tmp_subarray.shape[1]==0:
                            pass  # Stops nIterations loop if iterations exceed size of subarray
                        else:
                                binary_image_shift[ 0 :-1 - n_pixel_shift ,  0 : -1 - e_pixel_shift ] = tmp_subarray;  # This shifts the template WNW, For descending orbits                       
                                   
                    ## Pad beginning of array         
                    nRow_diff = bldg_img_wBuff.shape[0] - binary_image_shift.shape[0]
                    nCol_diff = bldg_img_wBuff.shape[1] - binary_image_shift.shape[1]
                    if (nRow_diff != 0) or (nCol_diff !=0):
                        B = np.pad(binary_image_shift,((nRow_diff,0),(nCol_diff,0)),'constant', constant_values=0);
                        binary_image_shift = B;
                        del B          

                   
                    ## Sum the template (binary image) and building out to check for overlap
                    bldg_img_wBuff_scaled = bldg_img_wBuff / bldg_img_wBuff # scales the footprint from 0 -->1 (incase of 8bit (e.g. 255) images)
                    alignment_cube_sum = np.add(bldg_img_wBuff_scaled, binary_image_shift)
                    overlapping_pixel_count = np.count_nonzero(alignment_cube_sum > 1) # Finds total number of elements in alignment_cube_sum > 1
                    nSumFootprint=np.count_nonzero(bldg_img_wBuff) # calculate length of the footprint (i.e. number of non-zero pixels)
                    percent_overlapping = overlapping_pixel_count / nSumFootprint *100;  # calculate the percent overlapping.

                    cube_overlap[nIterations,:,:] = alignment_cube_sum
                    cube_binary_image_shift[nIterations,:,:] = binary_image_shift;
                    cube_percent_overlap[nIterations] = percent_overlapping
                    cube_nIterations[nIterations] = nIterations
                    cube_northing_offset_pixels[nIterations] = northing_shift*nIterations
                    cube_easting_offset_pixels[nIterations] = easting_shift*nIterations
                    cube_northing_offset_m[nIterations] = northing_shift*nIterations * pixel_size_lat_m
                    cube_easting_offset_m[nIterations] = easting_shift*nIterations * pixel_size_lon_m

                    ## Calculate current height estimate                
                    Height_easting = cube_easting_offset_m[nIterations] * np.tan(np.deg2rad(in_incidence_sub)) / np.cos(np.deg2rad(azimuth))
                    Height_easting[ Height_easting < -500 ] = np.nan;     Height_easting[ Height_easting > 1000 ] = np.nan
                    Height_northing = cube_northing_offset_m[nIterations] * np.tan(np.deg2rad(in_incidence_sub)) / np.sin(np.deg2rad(azimuth))
                    Height_northing[ Height_easting < -500 ] = np.nan;     Height_northing[ Height_easting > 1000 ] = np.nan
                    Height_magnitude = np.sqrt(np.square(Height_northing) + np.square(Height_easting));
                    RMS_Height = np.sqrt(0.5*(np.square(Height_northing) + np.square(Height_easting)));

                    ## Save data into data cube variables
                    cube_Height_magnitude[nIterations,:,:]=Height_magnitude
                    cube_Height_magnitude_mean[nIterations]=np.nanmean(Height_magnitude)
                    cube_Height_magnitude_sigma[nIterations]=np.nanstd(Height_magnitude)
                    cube_RMS_Height[nIterations,:,:]=RMS_Height;
                    cube_RMS_Height_mean[nIterations]=np.nanmean(RMS_Height)
                    cube_RMS_Height_sigma[nIterations]=np.nanstd(RMS_Height)

                    del percent_overlapping, alignment_cube_sum
             
                
                ## Find maximum overlap
                imax_overlap=np.nanargmax(cube_percent_overlap)
                max_overlap=np.nanmax(cube_percent_overlap)
                print("Max overlap: Index, Value ", imax_overlap, ",", max_overlap)
                print(" ")
            
                 ## Calculate indices of building start/end points within Sigma0 and Incidence Angle input geotiffs
                (iy_building_start, iy_building_stop, ix_building_start, ix_building_stop) = calc_building_indices_for_full_sigma0_image(sigma0_lon_sub, sigma0_lat_sub, b_lon_arr, b_lat_arr)
            
            
                ## Test for threshold limit, save building footprint as -9999 if not enough overlap is present
                if max_overlap < layover_threshold_prct:
                    ## save output as -9999 
                    tmp_output = -9999 * np.ones((bldg_img.shape[0], bldg_img.shape[1]))  # save as -9999 if not enough overlap
                    height_output = tmp_output * bldg_img # mask non-building pixels 
                    height_output[ height_output == 0 ] = np.nan
                    height_output[ height_output == -9999 ] = np.nan; 


                else:
                    ## Reduce output array and mask pixels outside of building
                    tmp_output = cube_RMS_Height[imax_overlap, iy_building_start:iy_building_stop, ix_building_start:ix_building_stop]  # sample height to the extents of building outline subarray
                    print("mean height (m): ", np.mean(tmp_output))
                    print("iMax_overlap: ", imax_overlap)
                    height_output = tmp_output * bldg_img # mask non-building pixels 
                    height_output[ height_output == 0 ] = np.nan; 
                    print('Height output shape: ', height_output.shape)

                ## Add building height estimate to 2D numpy output array
                ## Note, there may be a slight offset due to differences between building and sigma0 reference frames. These offsets are within spatial resolution of a single pixel. E.g. This occurs when the upper left coordinates are not exact, but are within the precision of the input pixel (ambiguity in spatial resolution within 1 pixel)
                sar_layover_output_product[iy_sigma_start + iy_building_start : iy_sigma_start  + iy_building_stop, ix_sigma_start + ix_building_start : ix_sigma_start  + ix_building_stop ] = height_output;
        
        
                ## Write metadata file for current building and footprint
                write_metafile(infile_path, infilename, nBuilding, azimuth, orbit_direction, shift_direction, height_output, cube_RMS_Height_sigma, imax_overlap)
         
        ## Write out geotiff, after all buildings have been evaluated for current input Sigma0 image
        sar_layover_output_product [sar_layover_output_product == 0] =np.nan  # convert no data values to NaNs
        write_out_geotiff(infilename, infile_path, image_data, sar_layover_output_product)    
     


if __name__ ==  "__main__":

#### Implement Arg parse to pull relevant input parameters from input.txt file
      #Use Argparge to get information from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=argparse.FileType('r'))
    p = parser.parse_args()
     
    with p.filename as file:
        contents = file.read()
        args = ast.literal_eval(contents)
    
    outdirectory = args['out_dir']
    
    Sigma_files = glob.glob(os.path.join(args['sigma_file_loc'], '*Sigma0_VV_AST_B02.tif')) # Get list of input files
    footprint_dir = args['building_footprint_directory'] # Get building footprint from input argument text file
    #footprint_files = glob.glob(os.path.join(args['building_footprint_directory'], '*BuildingOutline*10m.tif')) # Get list of input files
    footprint_files = glob.glob(os.path.join(args['building_footprint_directory'], '*BuildingOutline*2m.tif')) # Get list of input files
    
    
#    Run_SAR_layover_algorithm(Sigma_files,footprint_dir, footprint_files) # execute main script
    Run_SAR_layover_algorithm(Sigma_files,footprint_dir, footprint_files, outdirectory) # execute main script


