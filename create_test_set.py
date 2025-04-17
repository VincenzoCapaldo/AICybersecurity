import os
import csv
import shutil
import random


def main(csv_file, dataset_directory_origin, dataset_directory_destination, number_img):
    
    # Lettura del file CSV contenente gli ID delle persone
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  # i campi sono separati da virgole
        for row in reader:
            if len(row) < 1:
                continue  # salta righe vuote

            person_id = row[0].strip()  # es: "n000016"
            person_name = row[1].strip(' "')

            # Costruzione dei percorsi di origine e destinazione per ogni persona
            origin_path = os.path.join(dataset_directory_origin, person_id)
            destination_path = os.path.join(dataset_directory_destination, person_id)

            if not os.path.exists(origin_path):
                print(f"Cartella non trovata: {origin_path}")
                continue

            # Lista dei file immagine nella cartella della persona
            images = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
            if not images:
                print(f"Nessuna immagine in: {origin_path}")
                continue

            # Seleziona un numero massimo di immagini in modo casuale
            random.seed(2025)
            selected_images = random.sample(images, min(number_img, len(images)))

            # Crea la cartella di destinazione se non esiste
            os.makedirs(destination_path, exist_ok=True)

            # Copia e rinomina le immagini nella cartella di destinazione
            for idx, image in enumerate(selected_images):
                src = os.path.join(origin_path, image)
                ext = os.path.splitext(image)[1]  # estensione del file (es. .jpg)
                dst_filename = f"{person_id}_{idx + 1:02d}{ext}"  # es: n000016_01.jpg
                dst = os.path.join(destination_path, dst_filename)
                shutil.copy2(src, dst)

            print(f"Copiate {len(selected_images)} immagini per {person_name}")

if __name__ == "__main__":
    # Parametri
    csv_file = '.\\dataset\\test_set.csv' # Percorso al file CSV contenente gli ID delle persone da inserire nel dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID
    dataset_directory_origin = '.\\dataset\\vggface2_train\\train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
    dataset_directory_destination = '.\\dataset\\test_set' # Directory in cui salvare le immagini selezionate
    number_img = 10 # Numero massimo di immagini da copiare per ciascun ID
    
    main(csv_file, dataset_directory_origin, dataset_directory_destination, number_img)