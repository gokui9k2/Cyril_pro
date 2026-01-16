import pandas as pd  
import numpy as np
import ast 

def safe_parse_orientation(x):

    if pd.isna(x) or x == "" or x is None:
        return None

    if isinstance(x, str) and x.lower().strip() == "nan":
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return tuple(round(float(v), 3) if not pd.isna(v) else 0.0 for v in x)
        except:
            return None

    if isinstance(x, str):
        try:

            clean_str = x.replace("nan", "None").replace("NaN", "None")
            
            parsed = ast.literal_eval(clean_str)
            
            return tuple(
                round(float(v), 3) if v is not None else 0.0 for v in parsed)
        except Exception:
            return None

    return None

def clean_string(df, pattern, sep):
  """
  Cleaning fonction for the string columns DICOM data.
  """
  df   = df.str.replace(pattern, '', regex=True)
  data = df.str.split(sep, expand = True)

  return data

def cleaning_numeric_dicom(series,pattern,sep,size_cols ):
  """
  Cleaning fonction for the numeric columns DICOM data.
  """
  df_string = clean_string(series, pattern,sep)

  if df_string.shape[1] != size_cols:
    df_string = df_string.reindex(columns =list(range(size_cols)))

  return df_string.apply(pd.to_numeric, errors='coerce')

def extract_candidate_clusters(df, group_cols, agg_config, query_filter, id_prefix="DCE_", id_offset=0):
    """
    Groups candidates, filters them, assigns IDs, and returns ALL columns (needed for ratios).
    """

    candidates = df.groupby(group_cols).agg(**agg_config)

    candidates = candidates.query(query_filter).copy()

    candidates["numeric_id"] = range(len(candidates))
    candidates["numeric_id"] = candidates["numeric_id"] + id_offset
    candidates["DCE_ID"] = id_prefix + candidates["numeric_id"].astype(str).str.zfill(5)

    candidates = candidates.reset_index().explode("series_instance_uid")

    return candidates


def filter_by_image_ratio(candidates_df, valid_ratios):
    """
    This function aims to verify whether a candidate DCE series meets a certain number of temporal position
    """
    imgs = pd.DataFrame(candidates_df["all_images"].tolist(), index=candidates_df.index)
    ratio = imgs.max(axis=1) / imgs.min(axis=1)

    return candidates_df[ratio.isin(valid_ratios)].copy()

def contraposal_cancer(df_pa,df_pe,df_dce):
  """
  This function collects information about patients who have cancer from the clinical report
  """
  cancer_pae_ids = df_pa.loc[df_pa['pae10'].isin([3.0, 4.0]), 'cn'].unique()
  cancer_pee_ids = df_pe.loc[df_pe['pee4'].isin([3.0, 4.0]), 'cn'].unique()
  merged_ids = np.union1d(cancer_pae_ids, cancer_pee_ids)

  df_cancer = pd.DataFrame({'cn': merged_ids})
  df_cancer['has_cancer'] = True

  final_df = df_dce.merge(df_cancer, on= "cn" , how ="left")

  return final_df

def true_dce(df_me,df_dce):
  """
  This function collects information about patients who have kinetics data in their clinical report
  """
  dce_ids_me_one = df_me[df_me["m3e14"].isin([1.0, 2.0, 3.0])]["cn"].unique()
  dce_ids_me_two = df_me[df_me["m3e66"].isin([1.0, 2.0, 3.0])]["cn"].unique()

  merged_ids = np.union1d(dce_ids_me_one, dce_ids_me_two)

  df_true_dce = pd.DataFrame({'cn': merged_ids})
  df_true_dce['confirmed_dce'] = True
  final_df = df_dce.merge(df_true_dce, on= "cn" , how ="left")

  return final_df

def initial_hill_breast(df_ii,df_dce):
  """
  This function collects information about the initial breast lesion
  """
  final_df = df_dce.merge(df_ii[["cn", "i1e34"]], on= "cn" , how ="left")

  return final_df


    
