import cv2
import mediapipe as mp
import time
import os

# Suppress unnecessary logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class HandDetector:
    def __init__(self, mode=False, maxhands=2, detectioncon=0.7, trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,  
            max_num_hands=self.maxhands,  
            min_detection_confidence=self.detectioncon,  
            min_tracking_confidence=self.trackcon  
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            h, w, _ = image.shape  # Get image dimensions

            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    lm.x *= w  # Convert normalized x to pixel coordinates
                    lm.y *= h

        return image
    def find_position(self, img, hand_no=0):
        lm_list = []
        h, w, _ = img.shape  # Get image size
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):  # Ensure the hand index is valid
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * 1), int(lm.y * 1)  # Convert to pixel coordinates
                    lm_list.append([id, cx, cy])

        return lm_list

def main():
    pTime = 0
    video = cv2.VideoCapture(1)  # Use 0 for default webcam
    detector = HandDetector()

    while True:
        success, image = video.read()
        if not success:
            break
        
    
        image = detector.find_hands(image)
        lmlist= detector.find_position(image)
        print(lmlist)
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
