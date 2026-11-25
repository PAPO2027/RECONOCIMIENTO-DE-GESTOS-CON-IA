import { FilesetResolver, GestureRecognizer, FaceLandmarker, ObjectDetector } from "@mediapipe/tasks-vision";
import "./style.css";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const cameraStage = document.querySelector(".camera-stage");
const ctx = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const placeholder = document.querySelector("#placeholder");
const message = document.querySelector("#message");
const appStatus = document.querySelector("#appStatus");
const cameraStatus = document.querySelector("#cameraStatus");

const gestureValue = document.querySelector("#gestureValue");
const handDetail = document.querySelector("#handDetail");
const faceValue = document.querySelector("#faceValue");
const faceDetail = document.querySelector("#faceDetail");
const headValue = document.querySelector("#headValue");
const headDetail = document.querySelector("#headDetail");
const objectValue = document.querySelector("#objectValue");
const objectDetail = document.querySelector("#objectDetail");
const faceActionValue = document.querySelector("#faceActionValue");
const faceActionDetail = document.querySelector("#faceActionDetail");
const headModel = document.querySelector("#headModel");
const customModelInput = document.querySelector("#customModelInput");
const customModelButton = document.querySelector("#customModelButton");

const MODELS = {
  gesture: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  objects: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
};

let stream = null;
let gestureRecognizer = null;
let faceLandmarker = null;\nlet objectDetector = null;
let animationId = null;
let lastVideoTime = -1;
let lastFaceDetectionTime = 0;
let latestHandResult = { gestures: [], handednesses: [], landmarks: [] };
let latestFaceResult = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
let latestObjectResult = { detections: [] };
let lastObjectDetectionTime = 0;
const OBJECT_INTERVAL_MS = 180;
let smoothedHeadAngles = { yaw: 0, pitch: 0, roll: 0 };
const FACE_INTERVAL_MS = 66;
let isRunning = false;
const previousHandCenters = new Map();
let previousFrameTime = 0;
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

const gestureNames = {
  Closed_Fist: "Puño cerrado",
  Open_Palm: "Mano abierta",
  Pointing_Up: "Dedo índice arriba",
  Thumb_Down: "Pulgar abajo",
  Thumb_Up: "Pulgar arriba",
  Victory: "Señal de victoria",
  ILoveYou: "Señal de amor y paz"
};

function setMessage(text, isError = false) {
  message.textContent = text;
  message.style.color = isError ? "#ff8fa3" : "";
}

async function loadModels() {
  setMessage("Preparando inteligencia artificial y aceleración gráfica...");
  appStatus.textContent = "CARGANDO MODELOS";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const createModels = async (delegate) => {
    const [gesture, face] = await Promise.all([
      GestureRecognizer.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODELS.gesture, delegate },
        runningMode: "VIDEO",
        numHands: 2
      }),
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODELS.face, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true
      })
    ]);
    const objects = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.objects, delegate },
      runningMode: "VIDEO",
      scoreThreshold: 0.45,
      maxResults: 8
    });
    return [gesture, face, objects];
  };

  // GPU evita bloquear tanto el hilo principal. Si no está disponible, usa CPU.
  try {
    [gestureRecognizer, faceLandmarker, objectDetector] = await createModels("GPU");
    console.info("MediaPipe ejecutándose con aceleración GPU.");
  } catch (gpuError) {
    console.warn("GPU no disponible; se utilizará CPU.", gpuError);
    [gestureRecognizer, faceLandmarker, objectDetector] = await createModels("CPU");
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("Tu navegador no permite acceder a la cámara. Abre la página mediante HTTPS o desde localhost.", true);
    return;
  }
  startButton.disabled = true;
  try {
    if (!gestureRecognizer || !faceLandmarker || !objectDetector) await loadModels();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    cameraStage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    isRunning = true;
    lastVideoTime = -1;
    lastFaceDetectionTime = 0;
    latestHandResult = { gestures: [], handednesses: [], landmarks: [] };
    latestFaceResult = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
    latestObjectResult = { detections: [] };
    lastObjectDetectionTime = 0;
    smoothedHeadAngles = { yaw: 0, pitch: 0, roll: 0 };
    placeholder.hidden = true;
    cameraStage.classList.add("scanning");
    stopButton.disabled = false;
    cameraStatus.innerHTML = "<span></span> CÁMARA ACTIVA";
    cameraStatus.classList.add("active");
    appStatus.textContent = "SISTEMA ANALIZANDO";
    setMessage("Escáner activo. Coloca tus manos y tu rostro frente a la cámara.");
    renderLoop();
  } catch (error) {
    console.error(error);
    const reason = error.name === "NotAllowedError"
      ? "No diste permiso para usar la cámara. Autoriza el acceso desde los ajustes del navegador."
      : error.name === "NotFoundError"
        ? "No se encontró una cámara conectada a este dispositivo."
        : error.name === "NotReadableError"
          ? "La cámara está siendo utilizada por otra aplicación. Ciérrala e inténtalo de nuevo."
          : "No se pudo iniciar el escáner. Revisa tu conexión a internet e inténtalo otra vez.";
    setMessage(reason, true);
    appStatus.textContent = "ERROR DE CONEXIÓN";
    startButton.disabled = false;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
}

function renderLoop() {
  if (!isRunning) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const timestamp = performance.now();

    // Las manos se actualizan en cada ciclo; el rostro se analiza a 15 FPS
    // para reducir carga sin dejar de mostrar una malla fluida.
    latestHandResult = gestureRecognizer.recognizeForVideo(video, timestamp);
    if (timestamp - lastFaceDetectionTime >= FACE_INTERVAL_MS) {
      latestFaceResult = faceLandmarker.detectForVideo(video, timestamp);
      lastFaceDetectionTime = timestamp;
      showFace(latestFaceResult);
    }

    showHands(latestHandResult);
    drawPoints(latestHandResult, latestFaceResult);
  }

  animationId = requestAnimationFrame(renderLoop);
}

function showHands(result) {
  const gestures = result.gestures ?? [];
  if (!gestures.length) {
    gestureValue.textContent = "Sin manos detectadas";
    handDetail.textContent = "Muestra una o ambas manos frente a la cámara.";
    return;
  }
  const labels = gestures.map((hand, index) => {
    const best = hand[0];
    const label = gestureNames[best.categoryName] || "Gesto no identificado";
    const side = result.handednesses?.[index]?.[0]?.displayName;
    const sideSpanish = side === "Left" ? "izquierda" : side === "Right" ? "derecha" : "";
    return sideSpanish ? `${label} (${sideSpanish})` : label;
  });
  gestureValue.textContent = labels.join(" · ");
  const confidence = Math.round((gestures[0][0].score || 0) * 100);
  const movement = getHandMovement(result.landmarks || [], result.handednesses || []);
  handDetail.textContent = `${gestures.length} mano(s) reconocida(s) · Confianza principal: ${confidence}% · ${movement}`;
}

function showFace(result) {
  if (!result.faceLandmarks?.length) {
    faceValue.textContent = "Rostro no detectado";
    faceDetail.textContent = "Coloca tu rostro dentro del área de la cámara.";
    faceActionValue.textContent = "Sin actividad";
    faceActionDetail.textContent = "Se detectarán guiños y movimientos visibles de ojos y boca.";
    headValue.textContent = "Sin datos";
    headDetail.textContent = "Se necesita detectar un rostro.";
    headModel.style.transform = "rotateX(0deg) rotateY(0deg) rotateZ(0deg)";
    return;
  }

  const categories = result.faceBlendshapes?.[0]?.categories || [];
  const scores = Object.fromEntries(categories.map(item => [item.categoryName, item.score]));
  const smile = ((scores.mouthSmileLeft || 0) + (scores.mouthSmileRight || 0)) / 2;
  const mouthOpen = scores.jawOpen || 0;
  const blinkLeft = scores.eyeBlinkLeft || 0;
  const blinkRight = scores.eyeBlinkRight || 0;

  let visualLabel = "Rostro detectado";
  if (smile > 0.45) visualLabel = "Sonrisa detectada";
  else if (mouthOpen > 0.5) visualLabel = "Boca abierta";
  faceValue.textContent = visualLabel;
  faceDetail.textContent = "Análisis de rasgos visibles; no interpreta emociones ni identidad.";

  let faceAction = "Ojos abiertos";
  if (blinkLeft > 0.62 && blinkRight < 0.38) faceAction = "Guiño del ojo izquierdo";
  else if (blinkRight > 0.62 && blinkLeft < 0.38) faceAction = "Guiño del ojo derecho";
  else if (blinkLeft > 0.62 && blinkRight > 0.62) faceAction = "Parpadeo";
  else if (mouthOpen > 0.55) faceAction = "Boca abierta";
  faceActionValue.textContent = faceAction;
  faceActionDetail.textContent = `Ojo izquierdo: ${Math.round(blinkLeft*100)}% · Ojo derecho: ${Math.round(blinkRight*100)}% · Apertura de boca: ${Math.round(mouthOpen*100)}%`;

  const matrix = result.facialTransformationMatrixes?.[0];
  const raw = matrix?.data ? Array.from(matrix.data) : [];
  if (raw.length >= 16) {
    // La matriz de MediaPipe describe la rotación 3D de la cara.
    const r00 = raw[0], r10 = raw[1], r20 = raw[2];
    const r21 = raw[6], r22 = raw[10];
    const pitch = Math.asin(Math.max(-1, Math.min(1, -r20))) * 180 / Math.PI;
    const yaw = Math.atan2(r10, r00) * 180 / Math.PI;
    const roll = Math.atan2(r21, r22) * 180 / Math.PI;
    const alpha = 0.28;
    smoothedHeadAngles.yaw += (yaw - smoothedHeadAngles.yaw) * alpha;
    smoothedHeadAngles.pitch += (pitch - smoothedHeadAngles.pitch) * alpha;
    smoothedHeadAngles.roll += (roll - smoothedHeadAngles.roll) * alpha;
    const y = smoothedHeadAngles.yaw;
    const p = smoothedHeadAngles.pitch;
    const r = smoothedHeadAngles.roll;
    const direction = Math.abs(y) > 12 ? (y > 0 ? "Giro a la izquierda" : "Giro a la derecha")
      : Math.abs(p) > 12 ? (p > 0 ? "Mirando hacia abajo" : "Mirando hacia arriba")
      : Math.abs(r) > 12 ? (r > 0 ? "Inclinación a la izquierda" : "Inclinación a la derecha")
      : "Mirando al frente";
    headValue.textContent = direction;
    headDetail.textContent = `Yaw ${y.toFixed(0)}° · Pitch ${p.toFixed(0)}° · Roll ${r.toFixed(0)}°`;
    headModel.style.transform = `rotateX(${-p}deg) rotateY(${y}deg) rotateZ(${-r}deg)`;
  } else {
    headValue.textContent = "Calculando orientación";
    headDetail.textContent = "Esperando la matriz facial 3D.";
  }
}

function identifyRaisedFingers(hand) {
  if (!hand || hand.length < 21) return [];
  const distance = (a,b) => Math.hypot(hand[a].x-hand[b].x, hand[a].y-hand[b].y, (hand[a].z||0)-(hand[b].z||0));
  const names = [];
  if (distance(4,5) > distance(3,5) + 0.025) names.push("pulgar");
  if (distance(8,0) > distance(6,0) + 0.035) names.push("índice");
  if (distance(12,0) > distance(10,0) + 0.035) names.push("medio");
  if (distance(16,0) > distance(14,0) + 0.035) names.push("anular");
  if (distance(20,0) > distance(18,0) + 0.035) names.push("meñique");
  return names;
}

function showObjects(result) {
  const detections = result.detections || [];
  if (!detections.length) {
    objectValue.textContent = "Sin objetos detectados";
    objectDetail.textContent = "El detector general busca objetos cotidianos del conjunto COCO.";
    return;
  }
  const labels = detections.slice(0,5).map(item => {
    const category = item.categories?.[0];
    const name = category?.categoryName || "Objeto";
    const confidence = Math.round((category?.score || 0) * 100);
    return `${translateObject(name)} (${confidence}%)`;
  });
  objectValue.textContent = labels.join(" · ");
  objectDetail.textContent = `${detections.length} detección(es). Puedes cargar un modelo TFLite compatible entrenado con tus propias categorías.`;
}

function translateObject(label) {
  const names = { person:"persona", bicycle:"bicicleta", car:"auto", motorcycle:"motocicleta", airplane:"avión", bus:"autobús", train:"tren", truck:"camión", boat:"barco", bottle:"botella", cup:"taza", chair:"silla", couch:"sofá", laptop:"computadora", cell_phone:"celular", book:"libro", backpack:"mochila", tv:"televisor", keyboard:"teclado", mouse:"ratón", apple:"manzana", banana:"plátano", orange:"naranja", dog:"perro", cat:"gato" };
  return names[label] || label.replaceAll("_"," ");
}

async function loadCustomObjectModel(file) {
  try {
    setMessage("Cargando tu modelo personalizado...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const buffer = new Uint8Array(await file.arrayBuffer());
    const customDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: buffer, delegate: "GPU" },
      runningMode: "VIDEO",
      scoreThreshold: 0.4,
      maxResults: 10
    });
    objectDetector = customDetector;
    objectDetail.textContent = `Modelo personalizado cargado: ${file.name}`;
    setMessage("Modelo personalizado activo. Las detecciones se procesan localmente.");
  } catch (error) {
    console.error(error);
    setMessage("No se pudo cargar. Usa un modelo .tflite compatible con MediaPipe Object Detector y con metadatos.", true);
  }
}
