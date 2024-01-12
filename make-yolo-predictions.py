# Import necessary libraries
import pandas as pd
from PIL import Image
import os
from utils import *

# On which Aquagar will this run?
AA_serials = ['AA-202310-002']

for AA_serial in AA_serials:

    # Define path of an Aquagar machine
    AA_machine_folder_path = f'/Users/guillemcobos/Library/CloudStorage/GoogleDrive-guillem.cobos@koabiotech.com/.shortcut-targets-by-id/1UQ_YnLRmoAqhCue-qXYNP9QhYorsiAxS/KOA Biotech/08. PRODUCTS/AQUAGAR/{AA_serial}'

    # Get information about Muestras (read muestras excel)
    AA_muestras_excel_path = os.path.join(AA_machine_folder_path, [item for item in os.listdir(AA_machine_folder_path) if 'Muestras.xlsx' in item][0])
    muestras_df = pd.read_excel(AA_muestras_excel_path, sheet_name = 'Muestras', dtype=str)

    # Define path of Raw Data folders
    AA_RawData_folder_path = os.path.join(AA_machine_folder_path, 'Raw Data')
    experiment_folders_paths = [os.path.join(AA_RawData_folder_path, item) for item in os.listdir(AA_RawData_folder_path) if 'DS_Store' not in item]

    for experiment_folder_path in experiment_folders_paths:

        # Create 'outputs' folder
        folder_name = os.path.join(experiment_folder_path, 'outputs')
        
        if not os.path.exists(folder_name):
        # Create the folder
            os.makedirs(folder_name)

            images_paths = sorted([os.path.join(experiment_folder_path, path) for path in os.listdir(experiment_folder_path) if path.endswith('.jpg')])

            if len(images_paths)>0:
                # Read image (the latest for now)
                input_image = Image.open(images_paths[-1])

                # Get plate_id from the picture!
                plate_id = get_plateid_from_image(input_image, expected_digits=4)

                # Get row_sample_names
                row_sample_names = get_row_sample_info(plate_id, muestras_df, experiment_folder_path)

                # Get dictionary of the six agar predictions
                six_agar_predictions, pred_images = get_six_agar_predictions(input_image)

                # Merge into a prediction for each row
                upperRowPred, lowerRowPred = three_agars_to_one_prediction(six_agar_predictions["upperRowTCBS"], six_agar_predictions["upperRowMSA"], six_agar_predictions["upperRowBA"]), three_agars_to_one_prediction(six_agar_predictions["lowerRowTCBS"], six_agar_predictions["lowerRowMSA"], six_agar_predictions["lowerRowBA"])
                
                # Transform these predictions from species-level to family-level
                upperRowFamiliesPred, lowerRowFamiliesPred = species2families(upperRowPred), species2families(lowerRowPred)
                
                # Display as icons representing categories
                upperRowFamiliesIconsPred, lowerRowFamiliesIconsPred = families2icons(upperRowFamiliesPred), families2icons(lowerRowFamiliesPred)

                # Get results_df
                results_df = get_results_df(upperRowFamiliesPred, lowerRowFamiliesPred, row_sample_names).drop(columns="FechaPreparacion")
                
                if not os.path.exists(os.path.join(experiment_folder_path, 'outputs', 'results.xlsx')):
                    results_df.to_excel(os.path.join(experiment_folder_path, 'outputs', 'results.xlsx'), index=False)
                if not os.path.exists(os.path.join(experiment_folder_path, 'outputs', 'yolo_predictions.jpg')):
                    get_and_save_collage(pred_images, os.path.join(experiment_folder_path, 'outputs', 'yolo_predictions.jpg'))