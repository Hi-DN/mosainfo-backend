import cv2
import numpy as np
import torch
import torchvision.models as models
from time import time, sleep
import os
import signal
from flask import Flask, jsonify
from flask_ngrok import run_with_ngrok
from torch.utils.data import DataLoader

import ffmpeg
import subprocess

base_url = "rtmp://3.34.97.138/"

waybill_weight = "/content/waybill_v1.2_r45.pt"
plate_weight="/content/plate.pt"
combined_weight="/content/all_combined_v2.3.2_flip.pt"

app = Flask(__name__)
run_with_ngrok(app) 

DataLoader.num_workers=0

CONN_LIMIT=10   
possible_list = [True] * CONN_LIMIT
q = [0,1,2,3,4,5,6,7,8,9]

# 현재 작동중인 프로세스 리스트 조회
@app.route("/processes/prev")
def getProcessList_prev():
    process_data_list = []

    for i in range(CONN_LIMIT):
        if not possible_list[i]:
            process_data = {
                'result' : 'true',
                'id': i
            }
            process_data_list.append(process_data)

    json_data = {
        'list': process_data_list
    }
    return jsonify(json_data)


@app.route("/processes")
def getProcessList():
    process_data_list = []

    for i in q:
      process_data = {
            'result' : 'true',
            'id': i
      }
      process_data_list.append(process_data)

    json_data = {
        'list': process_data_list
    }
    return jsonify(json_data)


# 촬영 시작 -> 지정 번호 get
@app.route("/get-id/prev")
def getId_prev():
    id=findPossibleId()

    if id!=-1 : 
        json_data={
            'result' : 'true',
            'id':id
        }
    else : json_data={
        'result':'false',
        'message': 'connection limit...'
        }

    return jsonify(json_data)


@app.route("/get-id")
def getId():

    if len(q)==0 : 
        json_data={
        'result':'false',
        'message': 'connection limit...'
        }
    else : 
          json_data={
            'result' : 'true',
            'id': q.pop(0)
        }

    return jsonify(json_data)


def findPossibleId():
    for i in range(0,CONN_LIMIT):
        if(possible_list[i]) : 
            possible_list[i]=False
            return i
    return -1

# 촬영 종료 -> 지정 번호 release
@app.route("/release/<id>/prev")
def releaseNumber_prev(id):
    possible_list[int(id)]=True
    print('release id : ',id)

    return jsonify({'result':'true'})

@app.route("/release/<id>")
def releaseNumber(id):
    q.append(int(id))
    print('release id : ',id)

    return jsonify({'result':'true'})


# 모자이크 시작
@app.route("/mosaic/<id>")
def mosaic(id):
    pid=os.fork()
    if pid == 0:
        my_pid = os.getpid()
        print('fork(), pid=',my_pid)

        work(id)

        print('mosaic finish, pid=',my_pid)
        os.kill(my_pid,signal.SIGKILL)

        return jsonify({'result':'true'})

    else: 
        return jsonify({'result':'true'})


def work(id):
    
    rtmp_in_url = base_url + "live/"+str(id)
    rtmp_out_url = base_url + "live-out/"+str(id)

    cap=cv2.VideoCapture(rtmp_in_url)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp_out_url]
  
    p=subprocess.Popen(command,stdin=subprocess.PIPE)

    mosaicObject = MosaicObject()
    mosaicObject.__init__

    sleep(1)

    while cap.isOpened():
      start_time = time()

      status, frame = cap.read()
      if not status : 
        print('can not read!')
        break

      results=mosaicObject.score_frame(frame)
      frame=mosaicObject.mosaic_frame(results,frame)

      # rtmp 서버로 push
      p.stdin.write(frame.tobytes())
      
      end_time=time()
      fps=1/np.round(end_time-start_time,3)
      print(f"FPS = {fps}")


    cap.release()
    print('cap release! id=',str(id))
    cv2.destroyAllWindows()

class MosaicObject:

    # 초기화
    def __init__(self):
        self.model=self.load_model()
        self.classes=self.model.names

    # 모델 설정
    def load_model(self):
        # model = torch.hub.load('ultralytics/yolov5','yolov5s')
        model=torch.hub.load('ultralytics/yolov5', 'custom', path=combined_weight)
        return model

    # 프레임별로 inference 진행 -> 인식한 label 및 정보 반환
    def score_frame(self, frame):
        frame=[frame]
        results=self.model(frame)
        labels,cord = results.xyxyn[0][:,-1].cpu().numpy(), results.xyxyn[0][:,:-1].cpu().numpy()
        return labels,cord

    # 클래스 값에 해당하는 label명 (string) 반환
    def class_to_label(self,x):
        return self.classes[int(x)]

    # 모자이크
    def mosaic_frame(self,results,frame):
        labels,cord=results
        n=len(labels)

        x_shape,y_shape=frame.shape[1],frame.shape[0]   # x_shape = 높이, y_shape = 너비

        # 검출된 객체 수만큼 돌며, 모자이크 처리
        for i in range(n):
            row=cord[i]     # row : xmin, ymin, xmax, ymax, confidence, class, name 
            if row[4] >= 0.6 :      # confidence 하한값 설정
                left=int(row[0]*x_shape)
                top=int(row[1]*y_shape)
                right=int(row[2]*x_shape)
                bottom=int(row[3]*y_shape)

                mosaic_part=frame[top:bottom,left:right]    # 모자이크 범위 설정
                mosaic_part=cv2.blur(mosaic_part,(50,50))   

                frame[top:bottom,left:right]=mosaic_part    # 해당 범위 모자이크처리한걸 원본에 덮어쓰기
        
        return frame


# 서버 start
if __name__ == "__main__":
    app.run()