import cv2

print("quit with q")
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
print("cap.isOpened()", cap.isOpened())
print("cap2.isOpened()", cap2.isOpened())
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # type: ignore
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # type: ignore
print("set CAP_PROP_FOURCC", cap.get(cv2.CAP_PROP_FOURCC))
while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    print(ret, frame.shape if ret else None)
    print(ret2, frame2.shape if ret2 else None)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
