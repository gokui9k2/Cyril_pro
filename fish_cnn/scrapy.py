import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.request
import base64
import time
import os 

driver_path = '' 

fish_list = {"Chromis_cyanea": ["Chromis+cyanea", "Blue+Chromis", "Blue+Reef+Chromis"],
    "Holacanthus_bermudensis": ["Holacanthus+bermudensis", "Blue+Angelfish", "Bermuda+Blue+Angelfish"],
    "Paracanthurus_hepatus": ["Paracanthurus+hepatus", "Blue+Tang", "Regal+Tang"],
    "Scarus_coeruleus": ["Scarus+coeruleus", "Blue+Parrotfish", "Blue+Parrot+Fish"], 
    "Chromis_viridis": ["Chromis+viridis", "Green+Chromis", "Blue+Green+Chromis"],
    "Odonus_niger": ["Odonus+niger", "Redtoothed+Triggerfish", "Niger+Trigger"],
    "Rhinomuraena_quaesita": ["Rhinomuraena+quaesita", "Ribbon+Eel", "Blue+Ribbon+Eel"],
    "Thalassoma_bifasciatum": ["Thalassoma+bifasciatum", "Bluehead+Wrasse", "Blue+Head+Wrasse"]}

root_project_path = ""
counter = 0
options = webdriver.ChromeOptions()
driver = uc.Chrome(options=options)

for key, fish_query in fish_list.items():
    print(f'Start : {key}')
    
    for fish in fish_query:
        url = f"https://www.google.com/search?q={fish}&tbm=isch"
        driver.get(url)
        
        # --- Handle Google Cookie Popup ---
        try:
            wait = WebDriverWait(driver, 5)
            # 'L2AGLb' is the ID for the "Accept all" button on Google
            accept_btn = wait.until(EC.element_to_be_clickable((By.ID, "L2AGLb")))
            accept_btn.click()
            print("Popup closed")
            time.sleep(2) 
        except Exception as e:
            print("No popup detected")
            
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        # --- Infinite Scroll Logic ---
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            try:
                more_button = driver.find_element(By.XPATH, "//input[@value='Afficher plus de résultats']")
                driver.execute_script("arguments[0].click();", more_button)
                print("'Show more' button clicked")
                time.sleep(3)
            except:
                pass
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  
            last_height = new_height
            
        time.sleep(2)
        
        # --- Folder Creation ---
        folder_path = root_project_path + key      
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Successfully created folder: {folder_path}")
        
        # --- Image Extraction & Download ---
        # 'YQ4gaf' can change over time
        img_elements = driver.find_elements(By.CLASS_NAME, "YQ4gaf")

        for idx, img in enumerate(img_elements):
            try:
                img_src = img.get_attribute('src')
                
                if not img_src:
                    continue
                
                file_path = f"{folder_path}/{key}_{counter + 1}.jpg"
                img_bytes = None
                
                # Handle Base64 images
                if img_src.startswith('data:image'):
                    img_data = img_src.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                
                # Handle HTTP images
                elif img_src.startswith('http'):
                    req = urllib.request.Request(img_src, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        img_bytes = response.read()
                else:
                    continue

                if img_bytes and len(img_bytes) >= 1000:
                    with open(file_path, 'wb') as f:
                        f.write(img_bytes)
                    counter += 1
                else:
                    continue
                    
            except Exception as e:
                continue

driver.quit()