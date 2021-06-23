import asyncio, json, os, cv2, platform, sys, time
#from time import sleep
from aiohttp import web
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
from aiohttp_basicauth import BasicAuthMiddleware
import numpy as np 

CAMERA_DEVICE=os.getenv('CAMERA_STREAM_URL',0)

class CameraDevice():
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_DEVICE)
        
        writer = None
        (W, H) = (None, None)
        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print("[INFO] {} total frames in video".format(total))
        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1
    
        ret, frame = self.cap.read()
        if not ret:
            print('Failed to open default camera. Exiting...')
            sys.exit()
        #self.cap.set(3, 640)
        #self.cap.set(4, 480)
        
        # if the frame dimensions are empty, grab them
        if self.W is None or H is None:
            (self.H, self.W) = frame.shape[:2]
        
        
        # load the COCO class labels our YOLO model was trained on
        self.LABELS = open("coco.names").read().strip().split("\n")
        
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
        
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)  
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        

    def rotate(self, frame):
        if flip:
            (h, w) = frame.shape[:2]
            center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
        return frame

    async def get_latest_frame(self):
        ret, frame = self.cap.read()
        
        # if the frame dimensions are empty, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
            
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        await asyncio.sleep(0)
        return self.rotate(frame)

    async def get_jpeg_frame(self):
        encode_param = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        frame = await self.get_latest_frame()
        frame, encimg = cv2.imencode('.jpg', frame, encode_param)
        return encimg.tostring()

class PeerConnectionFactory():
    def __init__(self):
        self.config = {'sdpSemantics': 'unified-plan'}
        self.STUN_SERVER = None
        self.TURN_SERVER = None
        self.TURN_USERNAME = None
        self.TURN_PASSWORD = None
        if all(k in os.environ for k in ('STUN_SERVER', 'TURN_SERVER', 'TURN_USERNAME', 'TURN_PASSWORD')):
            print('WebRTC connections will use your custom ICE Servers (STUN / TURN).')
            self.STUN_SERVER = os.environ['STUN_SERVER']
            self.TURN_SERVER = os.environ['TURN_SERVER']
            self.TURN_USERNAME = os.environ['TURN_USERNAME']
            self.TURN_PASSWORD = os.environ['TURN_PASSWORD']
            iceServers = [
                {
                    'urls': self.STUN_SERVER
                },
                {
                    'urls': self.TURN_SERVER,
                    'credential': self.TURN_PASSWORD,
                    'username': self.TURN_USERNAME
                }
            ]
            self.config['iceServers'] = iceServers

    def create_peer_connection(self):
        if self.TURN_SERVER is not None:
            iceServers = []
            iceServers.append(RTCIceServer(self.STUN_SERVER))
            iceServers.append(RTCIceServer(self.TURN_SERVER, username=self.TURN_USERNAME, credential=self.TURN_PASSWORD))
            return RTCPeerConnection(RTCConfiguration(iceServers))
        return RTCPeerConnection()

    def get_ice_config(self):
        return json.dumps(self.config)


class RTCVideoStream(VideoStreamTrack):
    def __init__(self, camera_device):
        super().__init__()
        self.camera_device = camera_device
        self.data_bgr = None

    async def recv(self):
        self.data_bgr = await self.camera_device.get_latest_frame()
        frame = VideoFrame.from_ndarray(self.data_bgr, format='bgr24')
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

async def index(request):
    content = open(os.path.join(ROOT, 'client/index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)

async def stylesheet(request):
    content = open(os.path.join(ROOT, 'client/style.css'), 'r').read()
    return web.Response(content_type='text/css', text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, 'client/client.js'), 'r').read()
    return web.Response(content_type='application/javascript', text=content)

async def balena(request):
    content = open(os.path.join(ROOT, 'client/balena-cam.svg'), 'r').read()
    return web.Response(content_type='image/svg+xml', text=content)

async def balena_logo(request):
    content = open(os.path.join(ROOT, 'client/balena-logo.svg'), 'r').read()
    return web.Response(content_type='image/svg+xml', text=content)

async def favicon(request):
    return web.FileResponse(os.path.join(ROOT, 'client/favicon.png'))

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])
    pc = pc_factory.create_peer_connection()
    pcs.add(pc)
    # Add local media
    local_video = RTCVideoStream(camera_device)
    pc.addTrack(local_video)
    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))

async def mjpeg_handler(request):
    boundary = "frame"
    response = web.StreamResponse(status=200, reason='OK', headers={
        'Content-Type': 'multipart/x-mixed-replace; '
                        'boundary=%s' % boundary,
    })
    await response.prepare(request)
    while True:
        data = await camera_device.get_jpeg_frame()
        await asyncio.sleep(0.2) # this means that the maximum FPS is 5
        await response.write(
            '--{}\r\n'.format(boundary).encode('utf-8'))
        await response.write(b'Content-Type: image/jpeg\r\n')
        await response.write('Content-Length: {}\r\n'.format(
                len(data)).encode('utf-8'))
        await response.write(b"\r\n")
        await response.write(data)
        await response.write(b"\r\n")
    return response

async def config(request):
    return web.Response(
        content_type='application/json',
        text=pc_factory.get_ice_config()
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

def checkDeviceReadiness():
    if not os.path.exists('/dev/video0') and platform.system() == 'Linux':
        print('Video device is not ready')
        print('Trying to load bcm2835-v4l2 driver...')
        os.system('bash -c "modprobe bcm2835-v4l2"')
        sleep(1)
        sys.exit()
    else:
        print('Video device is ready')

if __name__ == '__main__':
    checkDeviceReadiness()

    ROOT = os.path.dirname(__file__)
    pcs = set()
    camera_device = CameraDevice()

    flip = False
    try:
        if os.environ['rotation'] == '1':
            flip = True
    except:
        pass

    auth = []
    if 'username' in os.environ and 'password' in os.environ:
        print('\n#############################################################')
        print('Authorization is enabled.')
        print('Your balenaCam is password protected.')
        print('#############################################################\n')
        auth.append(BasicAuthMiddleware(username = os.environ['username'], password = os.environ['password']))
    else:
        print('\n#############################################################')
        print('Authorization is disabled.')
        print('Anyone can access your balenaCam, using the device\'s URL!')
        print('Set the username and password environment variables \nto enable authorization.')
        print('For more info visit: \nhttps://github.com/balena-io-playground/balena-cam')
        print('#############################################################\n')
    
    # Factory to create peerConnections depending on the iceServers set by user
    pc_factory = PeerConnectionFactory()

    app = web.Application(middlewares=auth)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/favicon.png', favicon)
    app.router.add_get('/balena-logo.svg', balena_logo)
    app.router.add_get('/balena-cam.svg', balena)
    app.router.add_get('/client.js', javascript)
    app.router.add_get('/style.css', stylesheet)
    app.router.add_post('/offer', offer)
    app.router.add_get('/mjpeg', mjpeg_handler)
    app.router.add_get('/ice-config', config)
    web.run_app(app, port=80)
