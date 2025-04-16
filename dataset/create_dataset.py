import os
import csv
import random
import shutil



def create_dataset(csv_file = 'test_set.csv', dataset_directory_origin = './vggface2_train/train',
    dataset_directory_destination = './test_set', number_img = 10):
    """
        Crea un sottoinsieme del dataset VGGFace2 copiando un numero specifico di immagini
        da ciascun ID persona specificato in un file CSV, e salvandole in una directory di destinazione.

        Args:
            csv_file (str): Percorso al file CSV contenente gli ID delle persone da inserire nel
            dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID.
            dataset_directory_origin (str): Directory di origine contenente le sottocartelle
                                            per ogni ID persona con le immagini.
            dataset_directory_destination (str): Directory in cui salvare le immagini selezionate.
            number_img (int): Numero massimo di immagini da copiare per ciascun ID.

        Returns:
            None. Le immagini vengono copiate e rinominate nella cartella di destinazione.
    """
    # Lettura del CSV
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  # i campi sono separati da virgole
        for row in reader:
            if len(row) < 1:
                continue

            person_id = row[0].strip()  # es: n000016

            origin_path = os.path.join(dataset_directory_origin, person_id)
            destination_path = os.path.join(dataset_directory_destination, person_id)

            if not os.path.exists(origin_path):
                print(f"Cartella non trovata: {origin_path}")
                continue

            images = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
            if not images:
                print(f"Nessuna immagine in: {origin_path}")
                continue

            selected_images = random.sample(images, min(number_img, len(images)))

            os.makedirs(destination_path, exist_ok=True)

            for idx, image in enumerate(selected_images):
                src = os.path.join(origin_path, image)
                ext = os.path.splitext(image)[1]
                dst_filename = f"{person_id}_{idx + 1:02d}{ext}"  # es: n000016_01.jpg
                dst = os.path.join(destination_path, dst_filename)
                shutil.copy2(src, dst)
            print(f"Copiate {len(selected_images)} immagini per {person_id}")


if __name__ == "__main__":
    #create_dataset()
    print_true_labels()