import cv2

def detecter_boule(image):
    # Convertir l'image en niveaux de gris
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un seuillage pour détecter la boule
    _, seuil = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours dans l'image seuillée
    contours, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trouver le contour le plus grand (supposé être la boule)
    plus_grand_contour = max(contours, key=cv2.contourArea)
    
    # Trouver le centre de la boule
    M = cv2.moments(plus_grand_contour)
    centre_x = int(M["m10"] / M["m00"])
    centre_y = int(M["m01"] / M["m00"])
    
    return centre_x, centre_y

# Chemin local de la vidéo sur votre appareil Android
chemin_video = "/Stockage interne/Movies/XRecorder0/XRecorder_Edited_03 062024_122101.mp4"

# Charger la vidéo localement
video = cv2.VideoCapture(chemin_video)

# Boucle pour lire les images de la vidéo
while True:
    # Lire une frame de la vidéo
    ret, frame = video.read()
    
    # Vérifier si la lecture de la vidéo est terminée
    if not ret:
        break
    
    # Détecter la boule dans l'image
    centre_x, centre_y = detecter_boule(frame)
    
    # Afficher le résultat
    print("Coordonnées de la boule : ({}, {})".format(centre_x, centre_y))

# Libérer la vidéo
video.release()
