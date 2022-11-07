from crypt import methods
from flask import Flask,jsonify,request
import json
import os
import signal
from datetime import datetime

app = Flask(__name__)

CONN_LIMIT=10   
possible_id_queue = [0,1,2,3,4,5,6,7,8,9]
working_streaming_queue=[]

class Streaming:
    def __init__(self, streaming_id, streaming_title, streaming_category,streaming_start_time):
        self.streaming_id=streaming_id # 스트림 id
        self.streaming_title=streaming_title # 스트림 제목
        self.streaming_category=streaming_category # 스트림 카테고리
        self.streaming_start_time = streaming_start_time # 스트림 시작시간

def deleteStreamingById(streaming_id):
    for i in range(len(working_streaming_queue)):
        if(working_streaming_queue[i].streaming_id == int(streaming_id)):
            working_streaming_queue.pop(i)
            break


# 현재 작동중인 프로세스 리스트 조회
@app.route("/streamings")
def getProcessList():
    process_data_list = []
    category = request.args.get('category')
    
    if category==None:
        for streaming in working_streaming_queue:
            process_data_list.append(streaming.__dict__)
    else:
        for streaming in working_streaming_queue:
            if (streaming.streaming_category==category):
                process_data_list.append(streaming.__dict__)

    json_data = {
        'list': process_data_list
    }

    return jsonify(json_data)

# 촬영 시작 -> 지정 번호 get
@app.route("/streaming",methods=['POST'])
def createStreaming():
    assigned_id = -1

    if len(possible_id_queue)==0 : 
        json_data={
        'result':'false',
        'message': 'connection limit...'
        }

    else : 
        assigned_id = possible_id_queue.pop(0)
        title = request.args.get('title')
        category = request.args.get('category')
        now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        new_streaming = Streaming(assigned_id,title,category,now)
        working_streaming_queue.append(new_streaming)

        json_data={
            'result' : 'true',
            'streaming': new_streaming.__dict__
        }

    return jsonify(json_data)



# 촬영 종료 -> 지정 번호 release
@app.route("/release/<id>")
def releaseNumber(id):
    # 진행 프로세스 큐에서 삭제
    deleteStreamingById(id)

    # 할당 id 큐 맨끝으로 반환
    possible_id_queue.append(int(id))

    print('release id : ',id)
    return jsonify({
        'result':'true',
        'id': id
    })


# 특정 id 로 스트리밍 실행중인가 조회
@app.route("/streaming/<id>")
def isStreamingExists(id):
    for streaming in working_streaming_queue:
        if streaming.streaming_id==int(id):
            return jsonify({'result':'true'})
     
    return jsonify({'result':'false'})


# 모자이크 시작
@app.route("/mosaic/<id>")
def mosaic(id):
    pid=os.fork()
    if pid == 0:
        my_pid = os.getpid()
        work(id)
        print('finish! ',my_pid)
        os.kill(my_pid,signal.SIGKILL)

    else: return jsonify({'result':'true'})


def work(id):
    print('work! id=',id)


# 서버 start
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8282', debug=True)