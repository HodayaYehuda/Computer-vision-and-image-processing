"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    img = cv2.imread(img_path)

    # gray format
    if rep == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gamma window
    title = 'Gamma Display'
    controller = 'Gamma'
    cv2.namedWindow(title)
    cv2.createTrackbar(controller, title, 0, 100, temp)

    while True:
        # specific brightness that the user stand on in the scale
        gamma_pos = cv2.getTrackbarPos(controller, title)
        gamma_pos = gamma_pos / 100 * (2 - 0.01)

        if gamma_pos == 0:
            gamma_pos = 0.01

        invGamma = 1.0 / gamma_pos
        # create new image and display it
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        newImg = cv2.LUT(img, table)
        cv2.imshow(title, newImg)

        # How smooth will be the transition between the images
        cv2.waitKey(1)

        # breakepoint
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) == 0:
            break

    cv2.destroyAllWindows()


def temp(i):
    pass


def main():
    gammaDisplay('test.txt', 2)


if __name__ == '__main__':
    main()
