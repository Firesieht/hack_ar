import cv2
import numpy as np
from math import sqrt

size = (7, 7)
kernel = np.ones((3, 3),np.uint8)
board = None

def extend(a, b, n):
    return a + (b - a) / sqrt(6) * sqrt(n)

def interp(edges, u, v):
    a, b, c, d = edges
    # v = 0.5
   #  print(d-a)
    return (a+(b-a)*u)*v+(d+(c-d)*u)*(1-v)

def main():
    global board
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        img = cv2.medianBlur(img, 3)
        # img = cv2.resize(img, None, fx=0.8, fy=0.8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # cv2.drawChessboardCorners(img, size, corners2, ret)
            # cv2.circle(img, tuple(int(x) for x in corners2[6, 0]), 0, (255, 0, 0), 10)

            edges = corners2[0, 0], corners2[6, 0], corners2[48, 0], corners2[42, 0]
            edges = np.array([extend(edges[2], edges[0], 8), extend(edges[3], edges[1], 8),
                              extend(edges[0], edges[2], 8), extend(edges[1], edges[3], 8)])

            l0, l1, r0, r1 = sorted(edges, key=lambda x: x[0])
            lu, ld, ru, rd = sorted((l0, l1), key=lambda x: x[1])+ sorted((r0, r1), key=lambda x:x[1])
            edges = np.array([lu, ru, rd, ld])

            # cv2.circle(img, tuple(int(x) for x in edges[0]), 0, (255, 0, 0), 10)
            # cv2.circle(img, tuple(int(x) for x in edges[1]), 0, (0, 255, 0), 10)
            # cv2.circle(img, tuple(int(x) for x in edges[2]), 0, (0, 0, 255), 10)
            board_mask = np.zeros_like(gray)
            cv2.fillConvexPoly(board_mask, edges.reshape((-1, 1, 2)).astype(np.int32), (255, 255, 255))
            # cv2.fillConvexPoly(board_mask, corners2.astype(np.int32), (255, 255, 255))
            board = board_mask, edges

        if board is not None:
            board_mask, edges = board
            cv2.circle(img, tuple(int(x) for x in edges[0]), 0, (255, 0, 0), 10)
            cv2.circle(img, tuple(int(x) for x in edges[1]), 0, (0, 255, 0), 10)
            cv2.circle(img, tuple(int(x) for x in edges[2]), 0, (0, 0, 255), 10)
            cv2.circle(img, tuple(int(x) for x in edges[3]), 0, (255, 0, 255), 10)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 50, 0), (50, 255, 255))
            mask *= board_mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            contours, im = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours is not None and len(contours) > 0:
                cnt = sorted(contours, key=cv2.contourArea, reverse=True)
                M = cv2.moments(cnt[0])
                try:
                    x = M['m10'] / M['m00']
                    y = M['m01'] / M['m00']
                    cv2.circle(img, (int(x), int(y)), 0, (0, 0, 255), 10)
                    better = []
                    for a in range(8):
                        for b in range(8):
                            point = interp(edges, (a+.5)/8, (b+.5)/8)
                            cv2.circle(img, (int(point[0]), int(point[1])), 0, (0, 255, 255), 10)
                            dist = point - np.array([x, y])
                            better.append((np.sum(dist **2), (a, b)))
                    better.sort(key=lambda x: x[0])
                    point = better[0][1]
                    point = 8-point[0], 1+point[1]
                    print(point)
                except ZeroDivisionError:
                    pass
            cv2.imshow("mask", mask)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()