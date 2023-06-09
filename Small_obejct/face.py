import face_recognition
import numpy as np
# from train import Generate_high_resolution
def face_reid(unknowface,suspect_image):

#   unknowface = Generate_high_resolution(unknowface)
  rgb = np.ascontiguousarray(unknowface[:, :, ::-1])
  # Initialize some variables
  face_encodings = []

  try:
      suspect_image_encoding = face_recognition.face_encodings(suspect_image)[0]
      face_encodings = face_recognition.face_encodings(rgb)[0]

  except IndexError:
      print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
      return False
  
  known_faces = [
      suspect_image_encoding
  ]

  # results is an array of True/False telling if the unknown face matched anyone in the known_faces array

  results = face_recognition.compare_faces(known_faces, face_encodings)

  print("Is the unknown face a picture of target? {}".format(results[0]))
  print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))


  return results[0]

