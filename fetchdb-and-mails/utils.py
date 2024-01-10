def get_plateid_from_image(img, expected_digits):
    from ultralytics import YOLO
    import cv2
    import easyocr

    model_path = 'models/detect-plateid.pt'
    model = YOLO(model_path)

    results = model(img, conf = 0.014, iou = 0.7)

    if results[0].boxes:
        # Assuming len(results)==1
        print(results)
        box = results[0].boxes.xyxy.numpy().astype(int)[0]
        crop = img.crop((box[0], box[1], box[2], box[3]))
        # img[box[1]:box[3], box[0]:box[2]]
        rotatedCrop = np.rot90(crop, k=-1)

        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Perform OCR (only look for NUMERICAL digits)
        ocr_results = reader.readtext(rotatedCrop)
        numerical_characters = ''.join([entry[1] for entry in ocr_results if entry[1].isdigit()])

        # The desired output should have 
        if len(numerical_characters)==expected_digits:
            return numerical_characters
        else:
            return 'Non-readable'
            console.log('Unable to extract plate_id')
    else: 
        return 'Non-detected'
    
def get_positions(circles, num_rows, num_cols):
    import numpy as np
    from sklearn.cluster import KMeans

    # Location algorithm for desired pocillo    
    x_kmeans = KMeans(n_clusters=num_cols, n_init=num_cols-1).fit(circles[:,0].reshape(-1, 1))
    x_labels, x_centers = x_kmeans.labels_, x_kmeans.cluster_centers_.squeeze()
    
    x_new_labels = np.zeros_like(x_labels)
    for i in range(len(x_centers)):
        x_new_labels[x_labels == i] = np.where(np.argsort(x_centers) == i)[0][0] 

    y_kmeans = KMeans(n_clusters=num_rows, n_init=num_rows-1).fit(circles[:,1].reshape(-1, 1))
    y_labels, y_centers = y_kmeans.labels_, y_kmeans.cluster_centers_.squeeze()
    
    y_new_labels = np.zeros_like(y_labels)
    for i in range(len(y_centers)):
        y_new_labels[y_labels == i] = np.where(np.argsort(y_centers) == i)[0][0]
        
    # Output is a dataframe where each row corresponds to a different circle.
    # We give coordinates of center, value of radius, row number and column number (in the grid)
    
    return np.column_stack((circles, y_new_labels+1,  x_new_labels+1))

def get_path_dict(results):
    # Get dictiionary with count of pathogens on a YOLO result

    names = results[0].names
    number_colonies = np.zeros((len(names)), dtype = int)
    
    for i in range(0, len(results[0])):
        box = results[0].boxes[i]
        class_id = int(box.cls[0].item())
        number_colonies[class_id] = number_colonies[class_id] + 1

    counting = {}
    for i in range(0, len(names)):
        bacteria_name = names.get(i)
        counting[bacteria_name] = int(number_colonies[i])

    return counting
    
def get_row_sample_names(plate_id, muestras_df):
    try:
        if any(muestras_df['Plate_id'] == plate_id):
            return_dict = {'UpperRow': list(muestras_df[(muestras_df['Plate_id'] == plate_id) & (muestras_df['Fila']=='1')].values.flatten())[::-1][1:],
                                    
                       'LowerRow': list(muestras_df[(muestras_df['Plate_id'] == plate_id) & (muestras_df['Fila']=='2')].values.flatten())[::-1][1:]}
        else: 
            return_dict = {'UpperRow': ['FilaSuperior', '', ''], 'LowerRow': ['FilaInferior', '', '']}
        return return_dict
    except (ValueError, IndexError):
        return_dict = {'UpperRow': ['FilaSuperior', '', ''], 'LowerRow': ['FilaInferior', '', '']}
    
    return return_dict

def get_six_agar_predictions(input_image):

    # Locate each agar (to crop later and perform pathogen prediction on each crop!)
    modelAgarsWells = YOLO('models/model_agars_wells.pt')

    resultsAgars = modelAgarsWells(input_image)[0]
    allBoxes = resultsAgars.boxes.xyxy.numpy().astype(int)

    agarsPositions = get_positions(allBoxes[:, 0:2], 2, 3)
    # Sort the array first by the last column (index 3), then by the one before column (index 2)
    sorted_indices = np.lexsort((agarsPositions[:, 3], agarsPositions[:, 2]))
    # Here sorting is top-down, and left-right, so like:
    # 1 2 3
    # 4 5 6 
    allBoxesSorted = allBoxes[sorted_indices]

    # Perform pathogen prediction on all the agars
    modelColonies = YOLO('models/micro_colony_counting.pt')
    pred_on_all_agars = []
    pred_images = []
    for box in allBoxesSorted:
        agarCrop = input_image.crop((box[0], box[1], box[2], box[3]))
        #agarCrop = input_image[box[1]:box[3], box[0]:box[2]] 
        results = modelColonies.predict(agarCrop, conf = .5)
            # Colonies prediction on agarCrop
        path_dict = get_path_dict(results)        # Get count of each pathogen type
        pred_on_all_agars.append(path_dict)       # Append global list
        pred_images.append(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))

    if len(pred_on_all_agars)==6:
        # The official column order for agars is: TCBS - MSA - BA
        upperRowTCBS, upperRowMSA, upperRowBA, lowerRowTCBS, lowerRowMSA, lowerRowBA= pred_on_all_agars[0], pred_on_all_agars[1], pred_on_all_agars[2], pred_on_all_agars[3], pred_on_all_agars[4], pred_on_all_agars[5]
    else:
        upperRowTCBS, upperRowMSA, upperRowBA, lowerRowTCBS, lowerRowMSA, lowerRowBA= {}, {}, {}, {}, {}, {}

    return ({"upperRowTCBS": upperRowTCBS, 
            "upperRowMSA": upperRowMSA, 
            "upperRowBA": upperRowBA, 
            "lowerRowTCBS": lowerRowTCBS, 
            "lowerRowMSA": lowerRowMSA, 
            "lowerRowBA": lowerRowBA}, pred_images)

def three_agars_to_one_prediction(TCBS, MSA, BA):
    return BA

def species2families(input_dict):
    # Define category mappings
    category_mapping = {
        'Vibrio': ['vharveyi', 'valgino', 'vangil'],
        'Aeromonas': ['assalmonicida'],
        'Photobacterium': ['pddamselae', 'pdpiscicida'],
        'Staphyloccocus': ['staphylo-']
    }

    # Aggregate values based on categories
    output_dict = {category: sum(input_dict.get(item, 0) for item in items) for category, items in category_mapping.items()}

    return output_dict

def families2icons(input_dict):
    categorized_dict = {}

    for key, value in input_dict.items():
        if value <= 1:
            categorized_dict[key] = '⚪️'
        elif value <=10:
            categorized_dict[key] = '🟢'
        elif value <= 20:
            categorized_dict[key] = '🟠'
        else:
            categorized_dict[key] = '🔴'

    return categorized_dict    

def families2microdiversity(pred_family):
    import numpy as np
    data = np.array(list(pred_family.values()))
    # Calculate mean and standard deviation
    mean_value = np.mean(data)
    std_dev = np.std(data)

    
    if std_dev!=0:
        # Calculate z-scores
        z_scores = (data - mean_value) / std_dev

        # Define a threshold for identifying outliers (e.g., z-score > 3 or < -3)
        threshold = 1.5

        # Identify outliers
        outliers = np.where(np.abs(z_scores) > threshold)[0]
    else: outliers=[]

    if len(outliers) in [1,2]:
        return '➖'
    elif np.all(data <5):
        return '🟰'
    else:
        return '➕'
    
def get_results_df(upperRowFamiliesPred, lowerRowFamiliesPred, upperRowFamiliesIconsPred, lowerRowFamiliesIconsPred, row_sample_names):
    microdiversity_indicator_df = pd.DataFrame([{'Biodiversidad':families2microdiversity(upperRowFamiliesPred)}, 
                                                {'Biodiversidad': families2microdiversity(lowerRowFamiliesPred)}]).transpose()
    families_icons_df = pd.DataFrame([upperRowFamiliesIconsPred, lowerRowFamiliesIconsPred]).transpose()
    results_df = pd.concat([families_icons_df, microdiversity_indicator_df])
    results_df.columns = [' '.join(row_sample_names['UpperRow']), ' '.join(row_sample_names['LowerRow'])]
    return results_df

def get_and_save_collage(pred_images, output_image_path):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Create a figure with 2 rows and 3 columns
    fig, axarr = plt.subplots(2, 3)

    # Display each image in its corresponding subplot
    axarr[0, 0].imshow(pred_images[0])
    axarr[0, 1].imshow(pred_images[1])
    axarr[0, 2].imshow(pred_images[2])
    axarr[1, 0].imshow(pred_images[3])
    axarr[1, 1].imshow(pred_images[4])
    axarr[1, 2].imshow(pred_images[5])


    # Remove axis ticks and labels for better visualization
    for ax in axarr.flat:
        ax.axis('off')

    # Adjust layout to prevent clipping of the subplot titles
    plt.tight_layout()

    plt.savefig(output_image_path)



###### Sending emails #####
serial_numbers_to_email_addresses = {'AA-202310-994': {'email': 'guillem.cobos@koabiotech.com', 'name': 'Guishermo'},
                                     'AA-202310-001': {'email': 'guillem.cobos@koabiotech.com', 'name': 'Guishermo'},
                                     'AA-202310-002': {'email': 'guillem.cobos@koabiotech.com', 'name': 'Guishermo'},
                                     'AA-202310-003': {'email': 'guillem.cobos@koabiotech.com', 'name': 'Guishermo'},
}

def generate_html_tables(row1_dict, row2_dict, diversity_indicator_row1, diversity_indicator_row2):
    # Generate HTML dynamically
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        }

        .container {
        display: flex;
        }

        .left-column {
        flex: 70%;
        padding: 20px;
        }

        .right-column {
        flex: 30%;
        padding: 20px;
        }

        table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        }

        th, td {
        border: 1px solid #dddddd;
        text-align: center;  /* Center align text */
        padding: 8px;
        }

        th {
        background-color: #f2f2f2;
        }
        
        /* Style for the blank row */
        .blank-row td {
            border: none;
        }

        /* Other styles remain unchanged */

    </style>
    <title>Dynamic HTML with Python</title>
    </head>
    <body>

    <div class="container">
        <div class="left-column">
        <table class="tg tg-left">
            <thead>
            <tr>
                <th class="tg-rlus">Bacteria</th>
                <th class="tg-rlus">Tanque 1</th>
                <th class="tg-rlus">Tanque 2</th>
            </tr>
            </thead>
            <tbody>
    """

    for bacteria, value1 in row1_dict.items():
        value2 = row2_dict.get(bacteria, '')  # Get corresponding value from tanque2_dict
        html_code += f"""
            <tr>
                <td class="tg-jjt0">{bacteria}</td>
                <td class="tg-j0tj">{value1}</td>
                <td class="tg-zsvf">{value2}</td>
            </tr>
    """

    html_code += f"""
            <tr class="blank-row">
                <td colspan="3"></td>
            </tr>
            <tr>
                <th class="tg-rlus"> <b> Biodiversidad </b> </th>
                <td class="tg-j0tj"> {diversity_indicator_row1} </td>
                <td class="tg-zsvf"> {diversity_indicator_row2} </td>
            </tr>
        </tbody>
        </table>
        </div>

        <div class="right-column">
        <table class="tg tg-right">
            <thead>
            <tr>
                <th class="tg-baqh" colspan="2"><span style="font-weight:bold">Indicadores</span></th>
            </tr>
            </thead>
            <tbody> 
            <tr>
                <td class="tg-j0tj"><span style="font-weight:400;font-style:normal">⚪️</span></td>
                <td class="tg-hmp3"><span style="font-weight:400;font-style:normal">Presencia nula</span></td>
            </tr>
            <tr>
                <td class="tg-j0tj"><span style="font-weight:400;font-style:normal">🟢</span></td>
                <td class="tg-hmp3"><span style="font-weight:400;font-style:normal">Presencia baja</span></td>
            </tr>
            <tr>
                <td class="tg-baqh"><span style="font-weight:400;font-style:normal">🟠</span></td>
                <td class="tg-0lax"><span style="font-weight:400;font-style:normal">Presencia media</span></td>
            </tr>
            <tr>
                <td class="tg-j0tj"><span style="font-weight:400;font-style:normal">🔴</span></td>
                <td class="tg-hmp3">Presencia alta</td>
            </tr>
            <tr>
                <td class="tg-baqh">➕</td>
                <td class="tg-0lax"><span style="font-weight:400;font-style:normal">Microbiota diversa</span></td>
            </tr>
            <tr>
                <td class="tg-j0tj">➖</td>
                <td class="tg-hmp3"><span style="font-weight:400;font-style:normal">Microbiota monopolizada</span></td>
            </tr>
            <tr>
                <td class="tg-baqh">🟰</td>
                <td class="tg-0lax">No aplicable</td>
            </tr>
            </tbody>
        </table>
        </div>
    </div>

    </body>
    </html>
    """

# Print or save the generated HTML code
    return html_code


def send_report_email(email_receivers, name_receiver, timestamp, plate_id, serial_num, html_tables):
    from email.message import EmailMessage
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    import ssl
    import smtplib

    email_sender = 'guillem.cobos@koabiotech.com'
    email_password = 'yqeq lxsp vdxl nkni'

    subject = f'Resultados del Análisis Microbiológico - Informe de Laboratorio🧬🔬' 
    body = f""" 
    <p>Estimado/a {name_receiver},</p>
    <p>Los siguientes resultados ofrecen una visión detallada de los parámetros biológicos evaluados en la muestra proporcionada.</p>
    <p><strong>Detalles del Análisis:</strong><br />- Fecha del Análisis: {timestamp} <br />- Tipo de Muestra: Cultivo microbiológico sobre agar<br />- Identificación de la Muestra: {plate_id}</p>
    """

    after_report = """ 
    <p><strong>Observaciones</strong>:<br />[Incluir cualquier observaci&oacute;n o nota relevante]</p>
    <p><strong>Recomendaciones</strong>:<br />[En caso de ser necesario, proporcionar recomendaciones basadas en los resultados]</p>
    <p>Si tiene alguna pregunta o necesita m&aacute;s informaci&oacute;n, no dude en ponerse en contacto con nuestro equipo de expertos. Agradecemos su confianza en nuestros servicios y esperamos poder atenderle en futuras ocasiones.</p>
    """
    
    email_signature = """
    <p>Atentamente,</p>
    <p>Equipo de an&aacute;lisis y laboratorio.</p>
    <p>&copy; 2023 KOA Biotech.</p>
    <p><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTqEEqzU6gsIyaAflx43K90fwr6f04EJZtLrGz9fC8_pOY4b19ayUC0SFt0muHUsf-9jQ&amp;usqp=CAU    " alt="" width="150" height="46" /></p>
    """
    em = MIMEMultipart()
    em['From'] = email_sender
    em['To'] = ', '.join(email_receivers)
    em['Subject'] = subject

    # Attach the body of the email
    em.attach(MIMEText(body, 'html'))
    # Attach the HTML content to the email
    em.attach(MIMEText(html_tables, "html"))
    em.attach(MIMEText(after_report, "html"))
    em.attach(MIMEText(email_signature, "html"))

    '''# Attach an image
    attachment_path = img_report_path
    with open(attachment_path, 'rb') as attachment:
        image = MIMEImage(attachment.read(), name='pathogen_predictions.png')
    em.attach(image)'''

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context = context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receivers, em.as_string())