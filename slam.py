#!/usr/bin/env python3
import cv2
import numpy as np

#extract features on a smaller patch, not the entire video

H = 3840
W = 2160

class FeatureExtractor(object):
  GX = 16
  GY = 16
  
  def __init__(self):
    self.orb = cv2.ORB_create(1000)

  def extract(self, frame):
    '''
    akp = []
    sy = frame.shape[0] // self.GY
    sx = frame.shape[1] // self.GX
    for ry in range(0, frame.shape[0], sy):
      for rx in range(0, frame.shape[1], sx):
        img_chunk = frame[ry:ry+sy,rx:rx+sx] 
        kp = self.orb.detect(img_chunk, None)
        for p in kp:
          p.pt = (p.pt[0] + rx, p.pt[1] + ry)
          akp.append(p)
    #orb = cv2.ORB_create()
    #kp = orb.detect(frame, None)
    #kp, des = orb.compute(frame, kp)
    return akp
    '''
    return cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 6000, qualityLevel=0.01, minDistance=3)

fe = FeatureExtractor()

def process_frame(frame):
  img = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
  kp = fe.extract(img)
  for f in kp:
    u, v = map(lambda x: int(round(x)), f[0])
    cv2.circle(img, (u, v), color=(0, 255, 0), radius = 3)
  '''
  for p in kp:
    u, v = map(lambda x: int(round(x)), p.pt)
    cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)
  return img
  '''
  return img

if __name__ == '__main__':
  path = "video2.mp4"
  video = cv2.VideoCapture(path)
  if video.isOpened() == False:
    print("Video could not be opened")
  while 1:
    ret, frame = video.read() 
    if ret == False:
      break
    frame = process_frame(frame)    
    #frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags = 0)
    cv2.imshow('SLAM', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  video.release()
  cv2.destroyAllWindows()

