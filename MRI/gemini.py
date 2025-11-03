def extract_function_calls(response):
    function_calls = []
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_calls.append({
                    'name': part.function_call.name,
                    'args': dict(part.function_call.args)
                })
    return function_calls

def get_mri_classification_tool() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="submit_mri_classifications",
        description="Submits the DCE classifications for all studies across all patients. Each study gets its own classification object.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "patients": types.Schema(
                    type="ARRAY",
                    description="List of patient classifications",
                    items=types.Schema(
                        type="OBJECT",
                    properties={
                        "patient_index": types.Schema(
                            type="INTEGER",
                            description="The patient number in the batch"),
                        "studies": types.Schema(
                        type="ARRAY",
                        description="List of study classifications for this patient",
                        items=types.Schema(
                            type="OBJECT",
                    properties={
                        "study_index": types.Schema(
                            type="INTEGER",
                            description="The study number for this patient"),
                        "dce_series_indices": types.Schema(
                            type="ARRAY",
                            description="List of SERIES INDICES (not series numbers!) that are DCE sequences. Use the index shown in brackets [index].",
                            items=types.Schema(type="INTEGER"))},
                    required=["study_index", "dce_series_indices"]))},
                    required=["patient_index", "studies"]))},
                        required=["patients"]))

def call_gemini_with_genai_sdk(prompt,system_prompt,api_key=None,model="gemini-2.5-pro",
    use_tools=True,
    enable_mri_classifier=False,
    temperature: float = 0.0):
    api_key = api_key or API_GEMINI
    if not api_key:
        raise ValueError("API key not provided")
    
    client = genai.Client(api_key=api_key)
    config_params = {"system_instruction": system_prompt,"temperature": temperature}
    
    if use_tools:
        function_declarations = []
        if enable_mri_classifier:
            function_declarations.append(get_mri_classification_tool())
        
        if function_declarations:
            tools = types.Tool(function_declarations=function_declarations)
            config_params["tools"] = [tools]
            config_params["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
    
    from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold
    
    safety_settings = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,hreshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,threshold=HarmBlockThreshold.BLOCK_NONE)]
    
    config_params["safety_settings"] = safety_settings
    config = types.GenerateContentConfig(**config_params)
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        return response
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise

def prepare_study_data(df, study_id):
    study_data = df[df['study_instance_uid'] == study_id].copy()
    
    agg_operations = {
        'series_number': 'first',
        'series_description': 'first', 
        'contrast_bolus_agent': 'first',
        'series_instance_uid': 'first',"Laterality" : "first"}
    
    if 'series_time' in study_data.columns:
        agg_operations['series_time'] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
    
    if 'series_instance_uid' in study_data.columns:
        series_data = study_data.groupby('series_instance_uid', as_index=False).agg(agg_operations)
        
        image_counts = study_data.groupby('series_instance_uid').size().reset_index(name='series_image_count')
        series_data = series_data.merge(image_counts, on='series_instance_uid', how='left')
    else:
        series_data = study_data
        series_data['series_image_count'] = 1
    
    if 'series_number' in series_data.columns:
        series_data = series_data.sort_values('series_number').reset_index(drop=True)
    elif 'series_time' in series_data.columns:
        series_data = series_data.sort_values('series_time', na_position='last').reset_index(drop=True)
    
    series_data['series_index'] = range(1, len(series_data) + 1)
    
    output_cols = ['series_index', 'series_number', 'series_description', 
                   'contrast_bolus_agent', 'series_image_count', 'series_time', 'series_instance_uid']
    existing_cols = [col for col in output_cols if col in series_data.columns]
    
    result = series_data[existing_cols].to_dict('records')

    import pandas as pd
    for record in result:
        for key, value in record.items():
            if pd.isna(value) or value is None:
                record[key] = 'N/A'
    
    return result

def prepare_patient_data(df, patient_id):
    patient_studies = df[df['patient_id'] == patient_id]['study_instance_uid'].unique()
    
    studies_data = []
    for study_id in patient_studies:
        study_data = prepare_study_data(df, study_id)
        studies_data.append({
            'study_instance_uid': study_id,
            'series_data': study_data
        })
    
    return studies_data

def format_batch_prompt(patients_data_list):
    
    prompt = """Analyze these patient MRI studies and identify all DCE series."""
    
    for patient_idx, patient_data in enumerate(patients_data_list, 1):
        prompt += f"{'='*60}\n"
        prompt += f"PATIENT {patient_idx}\n"
        prompt += f"{'='*60}\n\n"
        
        for study_idx, study_info in enumerate(patient_data['studies'], 1):
            prompt += f"{'='*25} STUDY {study_idx} {'='*25}\n"
            
            for series in study_info['series_data']:
                series_idx = series.get('series_index', '?')
                series_num = series.get('series_number', 'N/A')
                prompt += f"\n[{series_idx}] Series {series_num}:\n"
                prompt += f"  Description: {series.get('series_description', 'N/A')}\n"
                prompt += f"  Contrast: {series.get('contrast_bolus_agent', 'N/A')}\n"
                prompt += f"  Images: {series.get('series_image_count', 'N/A')}\n"
                prompt += f"  Acquisition time: {series.get('series_time', 'N/A')}\n"
                prompt += f"  Laterality: {series.get('Laterality', 'N/A')}\n"
            
            prompt += "\n"
        
        prompt += f"{'='*60}\n\n"
    
    return prompt

def process_dataset_in_batches(df, batch_size=5, api_key=None, max_retries=3, retry_delay=5, system_prompt=""):

    unique_patients = df['patient_id'].unique()
    all_dce_series = []
    
    for i in tqdm(range(0, len(unique_patients), batch_size), desc="Processing patient batches"):
        batch_patients = unique_patients[i:i + batch_size]
        
        patients_data_list = []
        patient_mapping = {}
        series_index_mapping = {}
        
        for patient_idx, patient_id in enumerate(batch_patients, 1):
            studies_data = prepare_patient_data(df, patient_id)
            
            patient_mapping[patient_idx] = patient_id
            
            for study_idx, study_info in enumerate(studies_data, 1):
                study_uid = study_info['study_instance_uid']
                
                for series in study_info['series_data']:
                    series_idx = series.get('series_index')
                    series_uid = series.get('series_instance_uid')
                    if series_idx and series_uid:
                        key = (patient_idx, study_idx, series_idx)
                        series_index_mapping[key] = {
                            'series_uid': series_uid,
                            'study_uid': study_uid}
            
            patients_data_list.append({'patient_id': patient_id,'studies': studies_data})
        prompt = format_batch_prompt(patients_data_list)
        success = False
        for attempt in range(max_retries):
            try:
                response = call_gemini_with_genai_sdk(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    enable_mri_classifier=True,
                    temperature=0.0)
                
                function_calls = extract_function_calls(response)
                
                if function_calls:
                    patients = function_calls[0]['args'].get('patients', [])
                    
                    for patient_classification in patients:
                        patient_idx = patient_classification.get('patient_index')
                        studies = patient_classification.get('studies', [])
                        
                        if patient_idx in patient_mapping:
                            for study_classification in studies:
                                study_idx = study_classification.get('study_index')
                                dce_series_indices = study_classification.get('dce_series_indices', [])
                                
                                for series_idx in dce_series_indices:
                                    key = (patient_idx, study_idx, series_idx)
                                    if key in series_index_mapping:
                                        mapping_info = series_index_mapping[key]
                                        all_dce_series.append({
                                            'patient_id': patient_mapping[patient_idx],
                                            'study_instance_uid': mapping_info['study_uid'],
                                            'series_instance_uid': mapping_info['series_uid'],
                                            'series_index': series_idx,
                                            'dce_category': 'DCE'})
                    
                    success = True
                    break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if i + batch_size < len(unique_patients):
            time.sleep(2)

    if all_dce_series:
        dce_df = pd.DataFrame(all_dce_series)
        dce_series_uids = dce_df['series_instance_uid'].unique()
        
        result_df = df[df['series_instance_uid'].isin(dce_series_uids)].copy()
        result_df['dce_category'] = 'DCE'
        
        return result_df
    else:
        return pd.DataFrame()
    

SYSTEM_PROMPT = """
You are an expert radiology assistant. Your task is to identify Dynamic Contrast-Enhanced (DCE) sequences from MRI series metadata.

## Task

You will receive different studies, each containing series with the following information:
    1.  **Series Description**: A free text written by a doctor.
    2.  **Contrast**: The contrast agent used for the series.
    3.  **Image Count**: The number of images contained in the series.
    4.  **Acquisition Time**: This is the acquisition time of the series.
    5.  **Laterality**: This represents the breast location. There are 4 values: R for right, L for left, B for Both, and O for Oblique.

You will have to analyze these five pieces of information to detect the DCE sequences in a study. A DCE can be contained in one series or split across several.

If the DCE is split across several series, their acquisition times should be close together, and they should generally have the same number of images. Be aware that a single series can have a very small number of images.

## Critical Information

* Keep in mind that a DCE has only one **Pre-Contrast** phase (which means an acquisition without the contrast agent) hower dont assume that it the case for every study 
this data is really heterogerone 
* Finally, keep in mind that one study can have two DCEs only if there is one for the right breast and another one for the left.

## OUTPUT
You MUST call the `submit_mri_classifications` function with your analysis.

Example for two studies:
- Study 1: Series 2, 3, 4 are DCE (1 pre-contrast + 2 post-contrast phases)
- Study 2: No DCE sequences

Call: `submit_mri_classifications`({
  "studies": [
    {"study_index": 1, "dce_series": [2, 3, 4]},
    {"study_index": 2, "dce_series": []}
  ]
})
"""