import cv2
from flask import Flask
from roboflow import Roboflow
import numpy as np
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

rf = Roboflow(api_key="neFuPyAEA71ZraMWqvRH")
project = rf.workspace().project("playing-cards-ow27d")
model = project.version(4).model

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255) 
thickness = 2


def capture(filename: str):
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow("Capturing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite(filename, frame)
            break

def infer(filename: str, confidence: int = 20):
    card_predict = model.predict(filename, confidence=confidence, overlap=30).json()
    cards = set([card["class"] for card in card_predict["predictions"]])

    # this is where the 10s are handled
    processed_cards = set()
    for card in cards:
        if card[0] == '1' and card[1] == '0':
            processed_cards.add(card[2] + 'T')
        else:
            processed_cards.add(card[1]+card[0])

    return processed_cards
    #return list(item[::-1] for item in cards)


def calc_prob(my_cards , community_cards):
    my_cards_obj = gen_cards(my_cards)
    community_cards_obj = gen_cards(community_cards)

    # Set the number of opponents
    num_opponents = 3

    # Calculate the win rate of your current hand
    win_rate = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=num_opponents+1, hole_card=my_cards_obj, community_card=community_cards_obj)

    # Output the result
    return f"Win rate: {win_rate * 100:.2f}%"

def live_cap(filename1, filename2):
    cap = cv2.VideoCapture(0)  
    prob_text = "Probability: None"

    write1, write2 = False, False
    while True:
        _, frame = cap.read()

        cv2.putText(frame, prob_text, (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Capturing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite(filename1, frame)
            write1 = True
            print("firstimagetaken")
            print(infer(filename1))

        if cv2.waitKey(1) & 0xFF == ord("e"):
            cv2.imwrite(filename2, frame)
            write2 = True
            print("secondimagetaken")
            print(infer(filename2))

        if write1 and write2 and (cv2.waitKey(1) & 0xFF == ord("f")):
            my_cards, community_cards = list(infer(filename1)), list(infer(filename2))
            for a_card in my_cards:
                if a_card in community_cards:
                    community_cards.remove(a_card)

            prob_text = "Probability: " + str(calc_prob(my_cards, community_cards))
            #prob_text = "Probability: 50%"
            print(prob_text)
            prob_img = np.zeros((50, 300, 3), np.uint8)
            cv2.putText(prob_img, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Capturing", frame)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            break
        


if __name__ == "__main__":
    # capture("my_cards.jpg")
    live_cap("1.jpg", "2.jpg")