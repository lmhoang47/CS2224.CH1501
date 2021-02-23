import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from flask_ngrok import run_with_ngrok
import os
import requests
from flask import Flask, render_template, request, jsonify
import match_images
import feature_extract_single_web
import random

#import param_config
from shutil import copyfile


app = Flask(__name__)
run_with_ngrok(app)

#load image & feature location from Oxford library
feature_image_names = []
for filename in os.listdir("/content/gdrive/MyDrive/Classroom/VIR/DELF/static/img"):
  if filename.endswith("jpg"): 
    feature_image_names.append(filename[0:filename.find('.')])

feature_query_paths = []
feature_index_paths = []
delg_local_dataset = []
feature_image_paths = []


def get_filepaths(directory):
  file_paths = []  # List which will store all of the full filepaths.
  # Walk the tree.
  for root, directories, files in os.walk(directory):
    #root = root.replace(' ', '\ ')
    for filename in files:
      if filename.find('delg_global') == -1:
        # Join the two strings in order to form the full filepath.        
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)  # Add it to the list.
  return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
feature_query_paths = get_filepaths("/content/gdrive/MyDrive/Classroom/VIR/DELF/static/feature")
#feature_index_paths = get_filepaths('/content/gdrive/MyDrive/Classroom/VIR/DELF/data/oxford5k_features/r50delg_gld/index')
delg_local_dataset = feature_query_paths + feature_index_paths
feature_image_paths = get_filepaths("/content/gdrive/MyDrive/Classroom/VIR/DELF/static/img")



#Web server:
##Allow web client can send photo for query
##Calculate feature of query photo
##Calculate inlier between query photo with Oxford photo
##Select top matched photo with highest inlier values

matched_images = []
target_matched_images = []
scores = []


@app.route('/', methods=['GET', 'POST'])
def index():
    matched_images = []
    matched_inlier_images = []
    target_matched_images = []
    target_matched_inliers = []
    scores = []
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        #uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_plot_" + file.filename
        uploaded_img_path = "static/uploaded/" + file.filename
        uploaded_img_path_saved = "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/uploaded/" + file.filename
        img.save(uploaded_img_path_saved)
        uploaded_img_path_saved = "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/uploaded/" + file.filename

        beonheo = file.filename[0:file.filename.find('.')]

        feature_extract_single_web.main('/content/gdrive/MyDrive/Classroom/VIR/DELF/parameters/r50delg_gld_config.pbtxt', beonheo, "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/uploaded", "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/uploaded/feature")

        uploaded_feature_path = "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/uploaded/feature/" + file.filename[0:file.filename.find('.')] + ".delg_local"

        cnt = 0
        for feature_image_name in feature_image_names:
          #print('feature_image_name: '+ feature_image_name)
          if cnt>19:
            break

          random.shuffle(feature_image_paths)
          for feature_image_path in feature_image_paths:
            #print('feature_image_path: '+ feature_image_path)
            if feature_image_path.find(feature_image_name) != -1:
              for delg_local_feature in delg_local_dataset:
                #print('delg_local_feature: '+ delg_local_feature)
                if delg_local_feature.find(feature_image_name) != -1:
                  output_name = datetime.now().isoformat().replace(":", ".") + '_matched_' + feature_image_name + '.jpg'
                  output_image = "/content/gdrive/MyDrive/Classroom/VIR/DELF/static/output/" + output_name

                  print('uploaded_img_path_saved: '+ uploaded_img_path_saved)
                  print('feature_image_path: '+ feature_image_path)
                  print('uploaded_feature_path: '+ uploaded_feature_path)
                  print('delg_local_feature: '+ delg_local_feature)

                  inlier = match_images.calculate_inlier(uploaded_img_path_saved, feature_image_path, uploaded_feature_path, delg_local_feature, output_image)

                  matched_images.append({"path": Path("static/img")/(feature_image_name+'.jpg'), "name": output_name, "inlier": inlier, "matched": Path("static/output")/(output_name)})
                  matched_inlier_images.append(inlier)

                  print('Count: ' + str(cnt))
                  cnt +=1

        for temp in sorted(matched_images, key = lambda line : line['inlier'], reverse = True)[0:10]:
          col = []
          for j in range(0,3):
            if j == 0:
              temp_path = temp['path']
              #print('temp_path: ' + str(temp_path))
              col.append(temp_path)
            elif j == 1:
              temp_inlier = temp['inlier']
              col.append(temp_inlier)
            else:
              temp_outputmatched = temp['matched']
              col.append(temp_outputmatched)
          target_matched_images.append(col)

        for target_matched_image in target_matched_images:
          print(target_matched_image)
        #print('len target_matched_images: ' + str(len(target_matched_images)))

        i = 0
        for target_matched_image in target_matched_images:
          col = []
          for j in range(0,4):
            if j == 0:
              col.append(i)
            elif j == 1:
              a = str(target_matched_image[0])
              col.append(a)
            elif j == 2:
              col.append(target_matched_image[1])
            else:
              col.append(target_matched_image[2])
          scores.append(col)
          i += 1
          print(scores)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run()

