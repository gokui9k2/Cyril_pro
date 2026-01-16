import pandas as pd
from core_function import clean_string, cleaning_numeric_dicom, extract_candidate_clusters, filter_by_image_ratio,safe_parse_orientation
import numpy as np 
import ast 

cleaning_num_configs = {"image_position" : {"size" : 3,
                        "cols" : ["patient_x","patient_y","patient_z"]},

    "Pixel_pacing" : {"size" : 2,
                      "cols" : ["pixel_spacing_x","pixel_spacing_y"]},

    "image_orientation" : {"size" : 6,
                           "cols" : ["V_row_x","V_row_y", "V_row_z", "V_col_x", "V_col_y" ,"V_col_z"]}}

AGG_BASIC = {
    "nb_series_in_cluster": ("series_instance_uid", "nunique"),
    "series_instance_uid":  ("series_instance_uid", "unique"),
    "patient_id":           ("cn", "first")}

AGG_WITH_IMG = {
    **AGG_BASIC,
    "all_images": ("series_image_count", "unique"),
    "n_images":   ("series_image_count", "nunique")}

DCE_KEYWORDS = [
    "dyn", "ph", "posts", "measure", "vibe", "thrive", "fspgr",
    "inj", "gd", "gad", r"\+c", "multiphase"]

pattern = "|".join(DCE_KEYWORDS)

VALID_RATIO=[2,3,4,5,6,7]

OFFSET_DCE_ID_1,OFFSET_DCE_ID_2,CASE_3_OFFSET  = 500,2000,5000

TAG_MAPPING =  {
    '00100020': "patient_id",
    '00080018': "SOP_ID",
    '0020000D': "study_instance_uid",
    '0020000E': "series_instance_uid",
    '00200011': "series_number",
    '00080031': "series_time",
    '00080060': "modality",
    '0008103E': "series_description",
    '00080008': "image_type",
    '00080020': "study_date",
    '00200013': "instance_number",
    '00080032': "acquisition_time",
    '00180080': "repetition_time",
    '00180081': "echo_time",
    '00180010': "contrast_bolus_agent",
    '00200105': "NumberOfTemporalPositions",
    '00200100': "TemporalPositionIdentifier",
    '00080070': "Manufacturer",
    '00251007': "Nb_image_in_one_serie_GE",
    '00081090': "manufacturer_model",
    '00201002': "DCE_tot_GE",
    '00181250': "Breast_R_L",
    '00200032': "image_position",
    '00200037': "image_orientation",
    '00181310': "Acquisition_Matrix",
    "00280010" : "Rows" ,
    "00280011" : "Columns" ,
    "00280030": "Pixel_pacing",
    "00180050" : "slice_thickness",
    "00180022" : "scan_option",
    "00181316" : "SAR"}

def detect_dce(df):
    
    # Normalization of the laterality data
    # Case 1: The laterality is provided but has several format
    df = df.assign(cn = df["patient_id"].str.split("-" ,expand = True)[4])
    
    breast_conditon_1 = ( (df["Breast_R_L"].isin(["R_BREAST","RBREAST"]) ),
                        (df["Breast_R_L"].isin(["L_BREAST","LBREAST"]) ),
                        (df["Breast_R_L"].isin(["2_BREAST"])))
    
    breast_value = ["R","L","B"]

    df["Laterality"] = np.select(breast_conditon_1, breast_value , default = None)
    
    # Case 2: The laterality information can be extracted through the description
    
    mask_na = df["Laterality"].isna()
    df["series_description"] = df["series_description"].fillna("NAN")
    series_lower = df.loc[mask_na, "series_description"].str.lower()
    
    left_pattern = (series_lower.str.match(r'(?=.*left)(?!.*right)' , na=False) |
                    series_lower.str.match(r'(?=.*\blt\b)(?!.*\brt\b)' , na=False) |
                    series_lower.str.match(r'(?=.*\bl\b)(?!.*\br\b)' , na=False))
    
    right_pattern = (series_lower.str.match(r'(?=.*right)(?!.*left)' , na=False) |
                     series_lower.str.match(r'(?=.*\brt\b)(?!.*\bl\b)' , na=False) |
                     series_lower.str.match(r'(?=.*\br\b)(?!.*\blt\b)' , na=False))

    both_pattern = (series_lower.str.match(r'(?=.*bilat)|(?=.*BIL)|(?=.*\BOTH\b)' , na=False))
    
    breast_conditon_2 = ((mask_na & right_pattern),
                          (mask_na & left_pattern),
                          (mask_na & both_pattern))

    df["Laterality"] = np.select(breast_conditon_2, breast_value, default = df["Laterality"])
    
    # Cleaning of numerical data parsing and numerical conversion
    
    clean_pattern = r'[\[\]\s\'"]'
    sep = ","
    
    for key, value in cleaning_num_configs.items():
    
      df[value["cols"]] = cleaning_numeric_dicom(df[key],clean_pattern,sep,value["size"])

    df['z_clean'] = df['patient_z'].astype(float).round(2)

    df[["Columns" ,"Rows","slice_thickness"]] = df[["Columns" ,"Rows","slice_thickness"]].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the physical width and height vectors projected onto the X-axis
    
    df['vec_width_x'] = (df['Columns'] - 1) * df['pixel_spacing_x'] * df['V_row_x']
    df['vec_height_x'] = (df['Rows'] - 1) * df['pixel_spacing_y'] * df['V_col_x']
    
    # Calculate X-coordinates for all four corners of the image slice
    
    P_tl_x = df['patient_x'] # Top-Left
    P_tr_x = df['patient_x'] + df['vec_width_x'] # Top-Right
    P_bl_x = df['patient_x'] + df['vec_height_x'] # Bottom-Left
    P_br_x = df['patient_x'] + df['vec_width_x'] + df['vec_height_x'] # Bottom-Right
    
    x_stack = np.vstack([P_tl_x,P_tr_x,P_bl_x,P_br_x])
    
    df['x_min_slice'] = np.min(x_stack,axis =0 )
    df['x_max_slice'] =  np.max(x_stack,axis =0 )
    
    # Group by Series to calculate spatial variance to find the scanning axis
    
    series_stats = df.groupby('series_instance_uid').agg(x_var=('patient_x', 'var'),y_var=('patient_y', 'var'),z_var=('patient_z', 'var'),
        x_min_series=('x_min_slice', 'min'),x_max_series=('x_max_slice', 'max'),V_row_x_first=('V_row_x', 'first')).fillna(0).reset_index()
    
    series_stats['x_center'] = (series_stats['x_min_series'] + series_stats['x_max_series']) / 2
    plane_condition = ((series_stats["x_var"] > np.max(series_stats[["y_var", "z_var"]] , axis = 1)),
                       (series_stats["y_var"] > np.max(series_stats[["x_var", "z_var"]] , axis = 1)),
                       (series_stats["z_var"] > np.max(series_stats[["x_var", "y_var"]] , axis = 1)))
    plane_value = ["Sagital", "Coronal", "Axial"]
    
    series_stats["Plane"] = np.select(plane_condition, plane_value, default="Oblique")

    # Case 3: For the remaining missing laterality values we can base our strategy on the RAS coordinate system
    
    laterality_condition = ((series_stats["Plane"].isin(['Axial', 'Coronal']) & (series_stats["x_center"] > 10)),
        (series_stats["Plane"].isin(['Axial', 'Coronal']) & (series_stats["x_center"] < -10)),
        (series_stats["Plane"] == 'Sagittal') & (series_stats["x_center"] > 10),
        (series_stats["Plane"] == 'Sagittal') & (series_stats["x_center"] < -10))
    
    laterality_value = ["L", "R", "L", "R"]
    
    series_stats["Laterality_2"] = np.select(laterality_condition, laterality_value, default="B")
    result_df = series_stats[['series_instance_uid', 'Plane', 'x_center', 'Laterality_2']]
    
    df = df.merge(result_df, on='series_instance_uid', how='left')
    
    df["Laterality"] = df["Laterality"].fillna(df['Laterality_2'])
    df = df.drop(columns = ["Laterality_2","x_center"])
    
    df["series_image_count"] = df.groupby("series_instance_uid")["series_description"].transform("count")
    

    mask = df["image_orientation"].notna()

    df.loc[mask, 'image_orientation_norm'] = df.loc[mask, 'image_orientation'].apply(safe_parse_orientation)
    df = df.dropna(subset=['image_orientation_norm'])
    df.loc[mask, 'image_orientation_norm'] = df.loc[mask, 'image_orientation_norm'].apply(tuple)
    # Date time normalization
    
    df["acquisition_time"] = pd.to_numeric(df["acquisition_time"], errors='coerce').fillna(0)
    df["series_time"] = pd.to_numeric(df["series_time"], errors='coerce').fillna(0)
    df["study_date"] =  pd.to_numeric(df["study_date"], errors='coerce').fillna(0)
    
    # To retrieve the DCE data more easily we remove all unnecessary series such as SCOUT, T2, DERIVATIVE, etc.....
    
    forbidden_types = r'SUB|MIP|DERIVED|SECONDARY|REFORMAT|SCREEN SAVE|PROJECTION'
    forbidden_desc = r't2|stir|flair|dwi|adc|scout|localizer|loc|topo'
    
    mask_clean_type = ~df['image_type'].astype(str).str.contains(forbidden_types, case=False, na=False)
    mask_clean_desc = ~df['series_description'].str.contains(forbidden_desc, case=False, na=False)
    
    df_cleaned = df[mask_clean_type & mask_clean_desc].copy()

    df_cleaned = df_cleaned.assign(cn = df_cleaned["cn"].astype(float),series_number = df_cleaned["series_number"].astype(float),
        instance_number = df_cleaned["instance_number"].astype(float))

    # Filter the DataFrame to keep only data from the 'Philips Medical Systems' manufacturer
    df_philips = df[df["Manufacturer"] == "Philips Medical Systems"].copy()
    df_philips['NumberOfTemporalPositions'] = df_philips['NumberOfTemporalPositions'].astype(float).fillna(0).astype(int)
    
    # Categorize breast laterality and detect DCE based on NumberOfTemporalPositions
    condition =  (df_philips['NumberOfTemporalPositions'] > 2) & (df_philips['NumberOfTemporalPositions'] <= 7)  & (df_philips["series_image_count"] > 70)
    dce_series_uids = df_philips.loc[condition , "series_instance_uid"]
   
    df_philips_dce = df_philips[df_philips['series_instance_uid'].isin(dce_series_uids)].copy()

    df_philips_dce['DCE_ID'] = ("DCE_" +df_philips_dce.groupby('series_instance_uid').ngroup().astype(str).str.zfill(4))
    
    # Filter the DataFrame to remove data from the 'Philips Medical Systems' manufacturer
    # This method is based on the image geometry to retrieve the DCE
    # Case 1: In this case we focus on DCE series that are split across different series
    
    COLS_GROUP_1 = ["study_instance_uid","Rows","Columns", "image_orientation_norm","slice_thickness","Laterality","Plane","series_image_count","scan_option"]
    
    df_cleaned_GE_SIE = df_cleaned[df_cleaned["Manufacturer"] != "Philips Medical Systems"].copy()
    dce_pass_1 = extract_candidate_clusters(df=df_cleaned_GE_SIE,group_cols=COLS_GROUP_1,
                                            agg_config=AGG_BASIC,query_filter="nb_series_in_cluster >= 3 and nb_series_in_cluster <= 7",id_offset = OFFSET_DCE_ID_1)
    mapping_case_1 = dce_pass_1.set_index("series_instance_uid")["DCE_ID"]
    mapping_case_1 = mapping_case_1[~mapping_case_1.index.duplicated(keep='first')]
    
    df_cleaned_GE_SIE["DCE_ID"] = df_cleaned_GE_SIE["series_instance_uid"].map(mapping_case_1)

    df_final_case_1 = df_cleaned_GE_SIE[df_cleaned_GE_SIE["DCE_ID"].notna()].copy()
    remaining_df = df_cleaned_GE_SIE[df_cleaned_GE_SIE["DCE_ID"].isna()].copy()
        
    # Case 2: Here,we focus on DCE series divided into one serie pre contrast and one serie post contrast
    
    COLS_GROUP_2 = ["study_instance_uid","Rows","Columns", "image_orientation_norm","slice_thickness","Laterality","Plane","scan_option"]
    
    dce_2 = extract_candidate_clusters(df=remaining_df,group_cols=COLS_GROUP_2,
                                            agg_config=AGG_WITH_IMG,query_filter="nb_series_in_cluster == 2 & n_images == 2 ",id_offset = OFFSET_DCE_ID_2)
    
    dce_pass_2_validated = filter_by_image_ratio(dce_2, VALID_RATIO)
    mapping_case_2 = dce_pass_2_validated.set_index("series_instance_uid")["DCE_ID"]
    mapping_case_2 = mapping_case_2[~mapping_case_2.index.duplicated(keep='first')]

    remaining_df["DCE_ID"] = remaining_df["DCE_ID"].fillna(remaining_df["series_instance_uid"].map(mapping_case_2))
    df_final_case_2 = remaining_df[remaining_df["DCE_ID"].notna()].copy()

    remaining_df_2 = remaining_df[remaining_df["DCE_ID"].isna()].copy()
    
    # Case 3: In this case, the DCE series is contained in a single large series of images
    
    CASE_3_PREFIX = "DCE_"

    idx_max_images = remaining_df_2.groupby("study_instance_uid")["series_image_count"].idxmax()
    candidates = remaining_df_2.loc[idx_max_images].copy()

    mask_single_dce = (
        (remaining_df_2["series_image_count"] > 120) &
        (remaining_df_2["series_description"].str.contains(pattern, case=False, na=False))
    )

    valid_candidates = candidates[mask_single_dce].copy()

    valid_candidates["numeric_id"] = range(len(valid_candidates))
    valid_candidates["numeric_id"] = valid_candidates["numeric_id"] + CASE_3_OFFSET
    valid_candidates["DCE_ID"] = CASE_3_PREFIX + valid_candidates["numeric_id"].astype(str).str.zfill(5)

    mapping_case_3 = valid_candidates.set_index("series_instance_uid")["DCE_ID"]

    remaining_df_2["DCE_ID"] = remaining_df_2["DCE_ID"].fillna(remaining_df_2["series_instance_uid"].map(mapping_case_3))
    df_final_case_3 = remaining_df_2[remaining_df_2["DCE_ID"].notna()].copy()
    
    df_final_dce = pd.concat([df_final_case_3 , df_final_case_2, df_final_case_1,df_philips_dce ], axis = 0)

    return df_final_dce