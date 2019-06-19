function initDetection() {
  const captureHeight = 480;
  const captureWidth = 640;

  const outputElement = document.getElementById('output');
  const cameraElement = document.createElement('video');


  let frameBuffer = new cv.Mat(captureHeight, captureWidth, cv.CV_8UC4);


  downloadModels()
    .then(createProcessingStep)
    .then(
      processingStep => ({
        cameraStream: setupCameraStream(cameraElement, captureHeight, captureWidth),
        processingStep: processingStep
      })
    ).then(
      objs => captureFrame(objs.cameraStream, frameBuffer, outputElement, objs.processingStep)
    ).then(
      () => console.log('capturing started')
    );

}


function downloadModels() {  
  console.log('download models started');

  return Promise.all([
    downloadFaceDetectionNet(),
    downloadAgeGenderNet()
  ])
    .then(nets => ({
      faceDetection: nets[0],
      ageGender: nets[1]
    })
    );
}


function downloadFaceDetectionNet() {
  console.log('Downloading face detection network');

  const faceDetectionPaths = {
    proto: "opencv_face_detector.prototxt",
    caffe: "opencv_face_detector.caffemodel"
  };

  const utils = new Utils('');

  return Promise.all([
    new Promise( (resolve, reject) =>
      utils.createFileFromUrl(faceDetectionPaths.proto, faceDetectionPaths.proto, resolve)
    ),
    new Promise( (resolve, reject) =>
      utils.createFileFromUrl(faceDetectionPaths.caffe, faceDetectionPaths.caffe, resolve)
    )
  ])
    .then(
      () => cv.readNet(faceDetectionPaths.proto, faceDetectionPaths.caffe)
    );
}

function downloadAgeGenderNet() {
  console.log('Downloading age/gender network');

  const ageGenderPaths = {
    xml: 'age-gender-recognition-retail-0013.bin',
    bin: 'age-gender-recognition-retail-0013.xml'
  };

  const utils = new Utils('');


  const binDownloadPromise = 
    new Promise( (resolve, reject) => {
      console.log('bin download');
      try {
        utils.createFileFromUrl(ageGenderPaths.bin, ageGenderPaths.bin, resolve);
      } catch(err) {
        console.log(err);
        reject(err);
      }
    });

  const xmlDownloadPromise = 
    new Promise( (resolve, reject) => {
      console.log('xml download');
      try {
        utils.createFileFromUrl(ageGenderPaths.xml, ageGenderPaths.xml, resolve)
      } catch(err) {
        console.log(err);
        reject(err);
      }
    });

  return Promise.all([
    binDownloadPromise,
    xmlDownloadPromise
  ])
    .then(
      () => {
        let net; 
        console.log('reading net');
        try {
          net = cv.readNet(ageGenderPaths.bin, ageGenderPaths.xml);
          console.log('got age gender net');
          console.dir(net);
        } catch(err) {
          console.log(err);
        }

        return net;
      }
    );
}


function captureFrame(cameraStream, frameBuffer, outputElement, processingStep) {
  console.log('Capturing frame');

  cameraStream.read(frameBuffer);

  const processedFrame = processingStep(frameBuffer);
  cv.imshow(outputElement, processedFrame); 
  processedFrame.delete();

  const frameTimeout = 1000;
  setTimeout(
    () => captureFrame(cameraStream, frameBuffer, outputElement, processingStep),
    frameTimeout
  );
};


function setupCameraStream(cameraElement, height, width) {
  console.log('setting up camera stream');

  cameraElement.setAttribute('width', width);
  cameraElement.setAttribute('height', height);

  // Get a permission from user to use a cameraElement.
  navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then((stream) => {
      cameraElement.srcObject = stream;
      cameraElement.onloadedmetadata = e => cameraElement.play();
    });

  // return open cameraElement stream
  return new cv.VideoCapture(cameraElement);
}


function createProcessingStep(models) {
  console.log('Creating processing steps');
  //console.log('Creating processing step');
  return function(frame) {
    //console.log('Running processing step');

    const processedFrame = frame.clone();
    cv.cvtColor(frame, processedFrame, cv.COLOR_RGBA2BGR);

    const confidenceThreshold = 0.5;
    const faceBounds = getFacesBoundingBoxes(processedFrame, models.faceDetection, confidenceThreshold);

    if(faceBounds.length > 0) {
      const faceBound = faceBounds[0];
      const net = models.ageGender;
      console.dir(net);
      const blob = cv.blobFromImage(processedFrame, 1, {width: 62, height: 62});
      net.setInput(blob);
      const out = net.forward();
      console.log(out);

    }


    drawBoundingBoxes(processedFrame, faceBounds, [0, 255, 0, 255]);


    cv.cvtColor(processedFrame, processedFrame, cv.COLOR_BGR2RGBA);
    return processedFrame;
  };
}


function drawBoundingBoxes(frame, boundingBoxes, color) {
  boundingBoxes.forEach( box => 
    cv.rectangle(
      frame,
      {x: box.left, y: box.top},
      {x: box.right, y: box.bottom}, 
      color
    )
  );
}


function getFacesBoundingBoxes(frame, net, thresh) {
  console.log('Getting faces bounding boxes');

  const faceBounds = [];

  const blob = cv.blobFromImage(frame, 1, {width: 192, height: 144}, [104, 117, 123, 0]);

  net.setInput(blob);
  const out = net.forward();

  for (let i = 0, n = out.data32F.length; i < n; i += 7) {
    const confidence = out.data32F[i + 2];
    if (confidence < thresh) { continue; }
    let left = out.data32F[i + 3] * frame.cols;
    let top = out.data32F[i + 4] * frame.rows;
    let right = out.data32F[i + 5] * frame.cols;
    let bottom = out.data32F[i + 6] * frame.rows;
    const width = right - left;
    const height = bottom - top;

    const scale = 0.25;
    left -= scale*width;
    right += scale*width;
    top -= scale*height;
    bottom += scale*height;

    faceBounds.push({
      left: (left > 0) ? left : 0,
      right: (right < frame.cols) ? right : frame.cols,
      top: (top > 0) ? top : 0,
      bottom: (bottom < frame.rows) ? bottom : frame.rows
    });

  }
  blob.delete();
  out.delete();

  return faceBounds;
}
