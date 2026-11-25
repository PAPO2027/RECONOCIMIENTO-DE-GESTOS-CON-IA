import { FilesetResolver, GestureRecognizer, FaceLandmarker, ObjectDetector } from "@mediapipe/tasks-vision";
import "./style.css";

const $ = (selector) => document.querySelector(selector);
const video = $("#camera");
const canvas = $("#overlay");
const cameraStage = $(".camera-stage");
const ctx = canvas.getContext("2d");
const startButton = $("#startButton");
const stopButton = $("#stopButton");
const placeholder = $("#placeholder");
const message = $("#message");
const appStatus = $("#appStatus");
const cameraStatus = $("#cameraStatus");
const gestureValue = $("#gestureValue");
const handDetail = $("#handDetail");
const faceValue = $("#faceValue");
const faceDetail = $("#faceDetail");
const headValue = $("#headValue");
const headDetail = $("#headDetail");
const objectValue = $("#objectValue");
const objectDetail = $("#objectDetail");
const faceActionValue = $("#faceActionValue");
const faceActionDetail = $("#faceActionDetail");
const headModel = $("#headModel");
const customModelInput = $("#customModelInput");
const customModelButton = $("#customModelButton");
const customGestureInput = $("#customGestureInput");
const customGestureButton = $("#customGestureButton");
const featureInputs = [...document.querySelectorAll("[data-feature]")];
const featureCards = [...document.querySelectorAll("[data-feature-card]")];
const selectionSummary = $("#selectionSummary");
const selectedFeatures = new Set(featureInputs.filter(input => input.checked).map(input => input.dataset.feature));

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODELS = {
  gesture: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  objects: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
};

let stream = null;
let gestureRecognizer = null;
let faceLandmarker = null;
let objectDetector = null;
let zeroShotObjectDetector = null;
let zeroShotLoading = false;
let zeroShotFailed = false;
let zeroShotBusy = false;
let lastZeroShotDetectionTime = 0;
let latestBaseObjectDetections = [];
let latestZeroShotDetections = [];
let latestFaceAccessoryDetections = [];
let latestGeneralZeroShotDetections = [];
let lastGeneralZeroShotTime = 0;
let smoothedObjectBoxes = [];
const objectInputCanvas = document.createElement("canvas");
objectInputCanvas.width = 640;
objectInputCanvas.height = 480;
const objectInputContext = objectInputCanvas.getContext("2d", { willReadFrequently: true });
const faceObjectCanvas = document.createElement("canvas");
faceObjectCanvas.width = 512;
faceObjectCanvas.height = 512;
const faceObjectContext = faceObjectCanvas.getContext("2d", { willReadFrequently: true });
let animationId = null;
let isRunning = false;
let lastVideoTime = -1;
let lastFaceDetectionTime = 0;
let lastObjectDetectionTime = 0;
let latestHandResult = { gestures: [], handednesses: [], landmarks: [] };
let latestFaceResult = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
let latestObjectResult = { detections: [] };
let smoothedHeadAngles = { yaw: 0, pitch: 0, roll: 0 };
const FACE_INTERVAL_MS = 66;
const OBJECT_FACE_INTERVAL_MS = 300;
const OBJECT_INTERVAL_MS = 450;
const ZERO_SHOT_INTERVAL_MS = 2400;
const GENERAL_ZERO_SHOT_INTERVAL_MS = 12000;
const ZERO_SHOT_LABELS = [
  // Electrónica y accesorios
  "TV remote control", "computer mouse", "computer keyboard", "computer monitor",
  "laptop computer", "desktop computer", "tablet", "smartphone", "phone case",
  "wireless earbuds", "earbuds charging case", "over-ear headphones", "headphone case",
  "webcam", "digital camera", "camera lens", "microphone", "tripod",
  "portable speaker", "Bluetooth speaker", "video game controller", "game console",
  "Wi-Fi router", "USB flash drive", "SD memory card", "USB cable",
  "phone charger", "power adapter", "power bank", "extension cord",
  "smartwatch", "digital wristwatch", "calculator",
  // Escritorio, oficina y útiles escolares
  "ballpoint pen", "pencil", "marker", "highlighter", "eraser",
  "pencil sharpener", "ruler", "geometry compass", "protractor",
  "notebook", "textbook", "paper folder", "ring binder", "pencil case",
  "stapler", "paper clips", "tape dispenser", "glue stick", "scissors",
  "sticky notes", "envelope", "printer", "desk organizer", "backpack",
  // Cocina y recipientes
  "water bottle", "thermos", "coffee mug", "drinking glass", "ceramic plate",
  "bowl", "spoon", "fork", "electric kettle", "frying pan", "cooking pot",
  "toaster", "blender", "food storage container", "lunch box",
  // Hogar y objetos personales
  "wall clock", "alarm clock", "table lamp", "light bulb", "flashlight",
  "candle", "tissue box", "soap dispenser", "plant pot", "vase",
  "picture frame", "door key", "keychain", "wallet", "purse",
  "suitcase", "umbrella", "shoe", "sneaker", "baseball cap", "hat",
  "belt", "sunglasses", "eyeglasses", "glasses case", "perfume bottle",
  "deodorant stick", "toothbrush", "toothpaste tube", "hairbrush", "comb",
  "hair dryer", "shampoo bottle", "lotion bottle", "makeup brush", "lipstick",
  // Herramientas y accesorios
  "screwdriver", "hammer", "wrench", "pliers", "measuring tape",
  // Deportes, juegos y juguetes
  "soccer ball", "basketball", "tennis racket", "tennis ball",
  "skateboard", "toy car", "teddy bear", "building blocks",
  "playing cards", "board game box", "dice",
  // Otros objetos comunes
  "book", "handbag", "camera", "remote control", "headphones",
  "watch", "water bottle", "coffee mug", "power bank", "flash drive"
];
const FACE_ACCESSORY_LABELS = [
  "eyeglasses", "glasses", "sunglasses"
];
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const previousHandCenters = new Map();

const gestureNames = {
  Closed_Fist: "Puño cerrado",
  Open_Palm: "Mano abierta",
  Pointing_Up: "Índice levantado",
  Thumb_Down: "Pulgar abajo",
  Thumb_Up: "Pulgar arriba",
  Victory: "Señal de victoria",
  ILoveYou: "Señal te quiero"
};

function isFeatureSelected(name) { return selectedFeatures.has(name); }

function updateFeatureSelection() {
  selectedFeatures.clear();
  featureInputs.forEach(input => {
    if (input.checked) selectedFeatures.add(input.dataset.feature);
  });
  featureCards.forEach(card => {
    card.hidden = !isFeatureSelected(card.dataset.featureCard);
  });
  selectionSummary.textContent = selectedFeatures.size + (selectedFeatures.size === 1 ? " función seleccionada" : " funciones seleccionadas");
  if (!selectedFeatures.size) setMessage("Selecciona al menos una función para iniciar el escáner.", true);
  else if (!isRunning) setMessage("Configuración lista. Puedes activar una o varias funciones.");
  if (isRunning && isFeatureSelected("objects")) loadZeroShotObjectModel();
}

featureInputs.forEach(input => input.addEventListener("change", updateFeatureSelection));
$("#selectAllFeatures").addEventListener("click", () => {
  featureInputs.forEach(input => { input.checked = true; });
  updateFeatureSelection();
});
$("#clearFeatures").addEventListener("click", () => {
  featureInputs.forEach(input => { input.checked = false; });
  updateFeatureSelection();
});
updateFeatureSelection();

function setMessage(text, isError = false) {
  message.textContent = text;
  message.style.color = isError ? "#ff8fa3" : "";
}

async function loadModels() {
  setMessage("Preparando modelos de manos, rostro y objetos...");
  appStatus.textContent = "CARGANDO MODELOS";
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  async function createAll(delegate) {
    const gesture = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.gesture, delegate },
      runningMode: "VIDEO",
      numHands: 2
    });
    const face = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.face, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    });
    const objects = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.objects, delegate },
      runningMode: "VIDEO",
      scoreThreshold: 0.45,
      maxResults: 20
    });
    return { gesture, face, objects };
  }

  try {
    const models = await createAll("GPU");
    gestureRecognizer = models.gesture;
    faceLandmarker = models.face;
    objectDetector = models.objects;
    console.info("Modelos MediaPipe cargados con GPU.");
  } catch (gpuError) {
    console.warn("GPU no disponible; intentando CPU.", gpuError);
    const models = await createAll("CPU");
    gestureRecognizer = models.gesture;
    faceLandmarker = models.face;
    objectDetector = models.objects;
    console.info("Modelos MediaPipe cargados con CPU.");
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("Abre esta página con HTTPS y permite el acceso a la cámara.", true);
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
    // Mantiene la imagen de entrada del segundo modelo a resolución nativa:
    // los lentes ocupan pocos píxeles y se pierden al reducir la cámara a 320x240.
    objectInputCanvas.width = video.videoWidth || 640;
    objectInputCanvas.height = video.videoHeight || 480;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    cameraStage.style.aspectRatio = video.videoWidth + " / " + video.videoHeight;
    isRunning = true;
    lastVideoTime = -1;
    lastFaceDetectionTime = 0;
    lastObjectDetectionTime = 0;
    lastZeroShotDetectionTime = 0;
    latestBaseObjectDetections = [];
    latestZeroShotDetections = [];
    smoothedObjectBoxes = [];
    latestHandResult = { gestures: [], handednesses: [], landmarks: [] };
    latestFaceResult = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
    latestObjectResult = { detections: [] };
    smoothedHeadAngles = { yaw: 0, pitch: 0, roll: 0 };
    previousHandCenters.clear();
    placeholder.hidden = true;
    cameraStage.classList.add("scanning");
    stopButton.disabled = false;
    cameraStatus.innerHTML = "<span></span> CÁMARA ACTIVA";
    cameraStatus.classList.add("active");
    appStatus.textContent = "SISTEMA ANALIZANDO";
    setMessage("Escáner activo. Muestra tus manos, rostro y objetos.");
    renderLoop();
    if (isFeatureSelected("objects")) loadZeroShotObjectModel();
  } catch (error) {
    console.error(error);
    const reason = error.name === "NotAllowedError"
      ? "No diste permiso para usar la cámara. Autoriza el acceso en el navegador."
      : error.name === "NotFoundError"
        ? "No se encontró una cámara conectada."
        : error.name === "NotReadableError"
          ? "La cámara está ocupada por otra aplicación."
          : "No se pudo iniciar el escáner. Revisa la conexión y los modelos.";
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
    if (isFeatureSelected("hands")) {
      latestHandResult = gestureRecognizer.recognizeForVideo(video, timestamp);
    } else {
      latestHandResult = { gestures: [], handednesses: [], landmarks: [] };
    }

    // También localizamos el rostro cuando se buscan objetos para ampliar la zona
    // de análisis y detectar mejor accesorios pequeños como los lentes.
    const needsDedicatedFace = isFeatureSelected("face") || isFeatureSelected("head") ||
      isFeatureSelected("faceActions");
    const needsFace = needsDedicatedFace || isFeatureSelected("objects");
    const faceInterval = needsDedicatedFace ? FACE_INTERVAL_MS : OBJECT_FACE_INTERVAL_MS;
    if (needsFace && timestamp - lastFaceDetectionTime >= faceInterval) {
      latestFaceResult = faceLandmarker.detectForVideo(video, timestamp);
      lastFaceDetectionTime = timestamp;
      showFace(latestFaceResult);
    } else if (!needsFace) {
      latestFaceResult = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
    }

    if (isFeatureSelected("objects")) {
      if (timestamp - lastObjectDetectionTime >= OBJECT_INTERVAL_MS) {
        const baseResult = objectDetector.detectForVideo(video, timestamp);
        latestBaseObjectDetections = (baseResult.detections || []).filter(item =>
          (item.categories?.[0]?.categoryName || "").toLowerCase() !== "person"
        );
        lastObjectDetectionTime = timestamp;
      }
      if (zeroShotObjectDetector && !zeroShotBusy && timestamp - lastZeroShotDetectionTime >= ZERO_SHOT_INTERVAL_MS) {
        runZeroShotObjectDetection(timestamp);
      }
      latestObjectResult = { detections: smoothObjectDetections([...latestBaseObjectDetections, ...latestZeroShotDetections]) };
      showObjects(latestObjectResult);
    } else {
      latestBaseObjectDetections = [];
      latestZeroShotDetections = [];
      latestObjectResult = { detections: [] };
    }
    showHands(latestHandResult);
    drawPoints(latestHandResult, latestFaceResult, latestObjectResult);
  }
  animationId = requestAnimationFrame(renderLoop);
}

function showHands(result) {
  const gestures = result.gestures || [];
  const hands = result.landmarks || [];
  if (!hands.length) {
    gestureValue.textContent = "Sin manos detectadas";
    handDetail.textContent = "Muestra una o ambas manos frente a la cámara.";
    return;
  }
  const descriptions = hands.map((hand, index) => {
    const raised = identifyRaisedFingers(hand);
    const side = result.handednesses?.[index]?.[0]?.displayName;
    const sideLabel = side === "Left" ? "izquierda" : side === "Right" ? "derecha" : "mano";
    const modelGesture = gestures[index]?.[0];
    const known = modelGesture && modelGesture.score > 0.55 ? gestureNames[modelGesture.categoryName] : "";
    const fingerText = raised.length ? raised.join(", ") : "ningún dedo extendido";
    return sideLabel + ": " + (known ? known + " · " : "") + fingerText;
  });
  gestureValue.textContent = descriptions.join(" | ");
  const confidence = gestures[0]?.[0]?.score ? Math.round(gestures[0][0].score * 100) : 0;
  handDetail.textContent = hands.length + " mano(s) · Confianza del gesto: " + confidence + "% · " + getHandMovement(hands, result.handednesses || []);
}

function identifyRaisedFingers(hand) {
  if (!hand || hand.length < 21) return [];
  const dist = (a, b) => Math.hypot(hand[a].x-hand[b].x, hand[a].y-hand[b].y, (hand[a].z || 0)-(hand[b].z || 0));
  const raised = [];
  if (dist(4, 5) > dist(3, 5) + 0.025) raised.push("pulgar");
  if (dist(8, 0) > dist(6, 0) + 0.035) raised.push("índice");
  if (dist(12, 0) > dist(10, 0) + 0.035) raised.push("medio");
  if (dist(16, 0) > dist(14, 0) + 0.035) raised.push("anular");
  if (dist(20, 0) > dist(18, 0) + 0.035) raised.push("meñique");
  return raised;
}

function getHandMovement(hands, handednesses) {
  let label = "Movimiento estable";
  const current = new Set();
  hands.forEach((hand, index) => {
    if (!hand?.length) return;
    const side = handednesses[index]?.[0]?.categoryName || "mano-" + index;
    current.add(side);
    const wrist = hand[0];
    const old = previousHandCenters.get(side);
    if (old) {
      const dx = wrist.x - old.x;
      const dy = wrist.y - old.y;
      if (Math.hypot(dx, dy) > 0.012) {
        if (Math.abs(dx) > Math.abs(dy)) label = dx > 0 ? "Movimiento a la derecha" : "Movimiento a la izquierda";
        else label = dy > 0 ? "Movimiento hacia abajo" : "Movimiento hacia arriba";
      }
    }
    previousHandCenters.set(side, { x: wrist.x, y: wrist.y });
  });
  for (const key of previousHandCenters.keys()) if (!current.has(key)) previousHandCenters.delete(key);
  return label;
}

function showFace(result) {
  if (!result.faceLandmarks?.length) {
    faceValue.textContent = "Rostro no detectado";
    faceDetail.textContent = "Coloca tu rostro dentro del área de la cámara.";
    faceActionValue.textContent = "Sin actividad";
    faceActionDetail.textContent = "Se detectarán guiños y parpadeos.";
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
  faceValue.textContent = smile > 0.48 ? "Sonrisa visible" : mouthOpen > 0.55 ? "Boca abierta" : "Rostro detectado";
  faceDetail.textContent = "Rasgos faciales visibles; no identifica a la persona ni determina emociones.";
  let action = "Ojos abiertos";
  if (blinkLeft > 0.62 && blinkRight < 0.38) action = "Guiño del ojo izquierdo";
  else if (blinkRight > 0.62 && blinkLeft < 0.38) action = "Guiño del ojo derecho";
  else if (blinkLeft > 0.62 && blinkRight > 0.62) action = "Parpadeo";
  else if (mouthOpen > 0.55) action = "Boca abierta";
  faceActionValue.textContent = action;
  faceActionDetail.textContent = "Ojo izq. " + Math.round(blinkLeft*100) + "% · Ojo der. " + Math.round(blinkRight*100) + "% · Boca " + Math.round(mouthOpen*100) + "%";
  if (isFeatureSelected("head")) updateHeadPose(result.facialTransformationMatrixes?.[0]);
  else { headValue.textContent = "Desactivado"; headDetail.textContent = "Activa Movimiento de cabeza en la configuración."; headModel.style.transform = "rotateX(0deg) rotateY(0deg) rotateZ(0deg)"; }
  if (!isFeatureSelected("face")) { faceValue.textContent = "Desactivado"; faceDetail.textContent = "Activa Rostro en la configuración."; }
  if (isFeatureSelected("faceActions")) { /* acciones faciales ya calculadas arriba */ }
  else { faceActionValue.textContent = "Desactivado"; faceActionDetail.textContent = "Activa Acciones faciales en la configuración."; }
}

function updateHeadPose(matrix) {
  const values = matrix?.data ? Array.from(matrix.data) : [];
  if (values.length < 16) {
    headValue.textContent = "Calculando orientación";
    headDetail.textContent = "Esperando la matriz facial 3D.";
    return;
  }
  // Matriz 4x4 de transformación facial; conversión de su bloque de rotación a Euler.
  const r00 = values[0], r10 = values[1], r20 = values[2];
  const r21 = values[6], r22 = values[10];
  const pitch = Math.asin(Math.max(-1, Math.min(1, -r20))) * 180 / Math.PI;
  const yaw = Math.atan2(r10, r00) * 180 / Math.PI;
  const roll = Math.atan2(r21, r22) * 180 / Math.PI;
  const smooth = 0.32;
  smoothedHeadAngles.yaw += (yaw - smoothedHeadAngles.yaw) * smooth;
  smoothedHeadAngles.pitch += (pitch - smoothedHeadAngles.pitch) * smooth;
  smoothedHeadAngles.roll += (roll - smoothedHeadAngles.roll) * smooth;
  const y = smoothedHeadAngles.yaw;
  const p = smoothedHeadAngles.pitch;
  const r = smoothedHeadAngles.roll;
  let direction = "Mirando al frente";
  if (Math.abs(y) > 12) direction = y > 0 ? "Giro hacia un lado" : "Giro hacia el otro lado";
  else if (Math.abs(p) > 12) direction = p > 0 ? "Cabeza hacia abajo" : "Cabeza hacia arriba";
  else if (Math.abs(r) > 12) direction = r > 0 ? "Inclinación lateral" : "Inclinación lateral";
  headValue.textContent = direction;
  headDetail.textContent = "Yaw " + y.toFixed(0) + "° · Pitch " + p.toFixed(0) + "° · Roll " + r.toFixed(0) + "°";
  headModel.style.transform = "rotateX(" + (-p) + "deg) rotateY(" + y + "deg) rotateZ(" + (-r) + "deg)";
}

async function loadZeroShotObjectModel() {
  if (zeroShotObjectDetector || zeroShotLoading || zeroShotFailed || !isFeatureSelected("objects")) return;
  zeroShotLoading = true;
  setMessage("Descargando el modelo adicional de objetos. La primera carga puede tardar...");
  try {
    const { pipeline } = await import("@huggingface/transformers");
    zeroShotObjectDetector = await pipeline(
      "zero-shot-object-detection",
      "Xenova/owlvit-base-patch32"
    );
    if (isRunning && isFeatureSelected("objects")) {
      setMessage("Modelo adicional listo: también buscará lentes, audífonos y más objetos.");
      appStatus.textContent = "MODELOS DE OBJETOS LISTOS";
    }
  } catch (error) {
    console.error("No se pudo cargar OWL-ViT:", error);
    zeroShotFailed = true;
    setMessage("El modelo adicional no pudo descargarse. El detector general seguirá funcionando.", true);
  } finally {
    zeroShotLoading = false;
  }
}

async function runZeroShotObjectDetection(timestamp) {
  if (!zeroShotObjectDetector || zeroShotBusy || !isFeatureSelected("objects") || !isRunning) return;
  zeroShotBusy = true;
  lastZeroShotDetectionTime = timestamp;
  try {
    const frameWidth = video.videoWidth;
    const frameHeight = video.videoHeight;
    if (!frameWidth || !frameHeight) return;

    // Prioriza accesorios del rostro para responder más rápido al buscar lentes.
    const face = latestFaceResult.faceLandmarks?.[0];
    if (face?.length) {
      const xs = face.map(point => point.x * frameWidth);
      const ys = face.map(point => point.y * frameHeight);
      const faceLeft = Math.max(0, Math.min(...xs));
      const faceRight = Math.min(frameWidth, Math.max(...xs));
      const faceTop = Math.max(0, Math.min(...ys));
      const faceBottom = Math.min(frameHeight, Math.max(...ys));
      const faceWidth = faceRight - faceLeft;
      const faceHeight = faceBottom - faceTop;

      if (faceWidth > 30 && faceHeight > 30) {
        const padX = faceWidth * 0.22;
        const padY = faceHeight * 0.22;
        const cropX = Math.max(0, faceLeft - padX);
        const cropY = Math.max(0, faceTop - padY);
        const cropRight = Math.min(frameWidth, faceRight + padX);
        const cropBottom = Math.min(frameHeight, faceBottom + padY);
        const cropWidth = cropRight - cropX;
        const cropHeight = cropBottom - cropY;

        faceObjectContext.clearRect(0, 0, faceObjectCanvas.width, faceObjectCanvas.height);
        faceObjectContext.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0,
          faceObjectCanvas.width, faceObjectCanvas.height);

        const accessoryPredictions = await zeroShotObjectDetector(
          faceObjectCanvas,
          FACE_ACCESSORY_LABELS,
          { threshold: 0.045, top_k: 5 }
        );
        if (!isRunning || !isFeatureSelected("objects")) return;

        const faceScaleX = cropWidth / faceObjectCanvas.width;
        const faceScaleY = cropHeight / faceObjectCanvas.height;
        latestFaceAccessoryDetections = (accessoryPredictions || []).map(item => ({
          boundingBox: {
            originX: cropX + item.box.xmin * faceScaleX,
            originY: cropY + item.box.ymin * faceScaleY,
            width: (item.box.xmax - item.box.xmin) * faceScaleX,
            height: (item.box.ymax - item.box.ymin) * faceScaleY
          },
          categories: [{ categoryName: item.label, score: item.score }]
        }));
      } else {
        latestFaceAccessoryDetections = [];
      }
    } else {
      latestFaceAccessoryDetections = [];
    }

    // La búsqueda general se ejecuta con menor frecuencia para reducir carga.
    if (timestamp - lastGeneralZeroShotTime >= GENERAL_ZERO_SHOT_INTERVAL_MS) {
      objectInputContext.drawImage(video, 0, 0, objectInputCanvas.width, objectInputCanvas.height);
      const generalPredictions = await zeroShotObjectDetector(
        objectInputCanvas,
        ZERO_SHOT_LABELS,
        { threshold: 0.11, top_k: 18 }
      );
      if (!isRunning || !isFeatureSelected("objects")) return;

      const scaleX = frameWidth / objectInputCanvas.width;
      const scaleY = frameHeight / objectInputCanvas.height;
      latestGeneralZeroShotDetections = (generalPredictions || []).map(item => ({
        boundingBox: {
          originX: item.box.xmin * scaleX,
          originY: item.box.ymin * scaleY,
          width: (item.box.xmax - item.box.xmin) * scaleX,
          height: (item.box.ymax - item.box.ymin) * scaleY
        },
        categories: [{ categoryName: item.label, score: item.score }]
      }));
      lastGeneralZeroShotTime = timestamp;
    }

    // Evita mostrar etiquetas de lentes duplicadas sobre la misma zona.
    const combined = [...latestFaceAccessoryDetections, ...latestGeneralZeroShotDetections];
    latestZeroShotDetections = combined.filter((item, index) => {
      const label = item.categories[0].categoryName;
      const box = item.boundingBox;
      const eyewear = ["eyeglasses", "glasses", "sunglasses"];
      const duplicate = combined.slice(0, index).some(previous => {
        const previousLabel = previous.categories[0].categoryName;
        if (!eyewear.includes(label) || !eyewear.includes(previousLabel)) return false;
        const p = previous.boundingBox;
        const intersectionW = Math.max(0, Math.min(box.originX + box.width, p.originX + p.width) - Math.max(box.originX, p.originX));
        const intersectionH = Math.max(0, Math.min(box.originY + box.height, p.originY + p.height) - Math.max(box.originY, p.originY));
        const intersection = intersectionW * intersectionH;
        const union = box.width * box.height + p.width * p.height - intersection;
        return union > 0 && intersection / union > 0.45;
      });
      return !duplicate;
    });

    latestObjectResult = { detections: smoothObjectDetections([...latestBaseObjectDetections, ...latestZeroShotDetections]) };
    showObjects(latestObjectResult);
  } catch (error) {
    console.error("Error al detectar objetos con OWL-ViT:", error);
    objectDetail.textContent = "Error en el modelo adicional: " + (error?.message || "revisa la consola del navegador");
  } finally {
    zeroShotBusy = false;
  }
}

function smoothObjectDetections(detections) {
  // Asocia detecciones consecutivas por etiqueta y cercanía para suavizar el movimiento
  // de los recuadros sin ejecutar otro modelo pesado en cada fotograma.
  const nextTracks = [];
  const used = new Set();
  for (const detection of detections) {
    const box = detection.boundingBox;
    if (!box) continue;
    const label = detection.categories?.[0]?.categoryName || "object";
    const cx = box.originX + box.width / 2;
    const cy = box.originY + box.height / 2;
    let bestIndex = -1;
    let bestDistance = Infinity;
    smoothedObjectBoxes.forEach((track, index) => {
      if (used.has(index) || track.label !== label) return;
      const distance = Math.hypot(cx - track.cx, cy - track.cy);
      const limit = Math.max(70, Math.max(box.width, box.height) * 0.9);
      if (distance < limit && distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    let smoothed = { ...box };
    if (bestIndex >= 0) {
      used.add(bestIndex);
      const old = smoothedObjectBoxes[bestIndex];
      const alpha = 0.62;
      smoothed = {
        originX: old.box.originX + (box.originX - old.box.originX) * alpha,
        originY: old.box.originY + (box.originY - old.box.originY) * alpha,
        width: old.box.width + (box.width - old.box.width) * alpha,
        height: old.box.height + (box.height - old.box.height) * alpha
      };
    }
    nextTracks.push({
      label,
      cx: smoothed.originX + smoothed.width / 2,
      cy: smoothed.originY + smoothed.height / 2,
      box: smoothed
    });
    detection.boundingBox = smoothed;
  }
  smoothedObjectBoxes = nextTracks;
  return detections;
}

function showObjects(result) {
  // La sección de objetos ignora personas; el reconocimiento facial sigue funcionando por separado.
  const detections = (result.detections || []).filter(item =>
    (item.categories?.[0]?.categoryName || "").toLowerCase() !== "person"
  );
  if (!detections.length) {
    objectValue.textContent = "Sin objetos detectados";
    objectDetail.textContent = "Busca objetos cotidianos; las personas se excluyen de esta sección.";
    return;
  }
  const labels = detections.slice(0, 10).map(item => {
    const category = item.categories?.[0];
    return translateObject(category?.categoryName || "Objeto") + " (" + Math.round((category?.score || 0) * 100) + "%)";
  });
  objectValue.textContent = labels.join(" · ");
  objectDetail.textContent = detections.length + " objeto(s) · Confianza visible en cada recuadro · Seguimiento suavizado activo.";
}

function translateObject(label) {
  const names = {
    // Categorías habituales del modelo COCO
    bicycle:"bicicleta", car:"auto", motorcycle:"motocicleta", airplane:"avión",
    bus:"autobús", train:"tren", truck:"camión", boat:"barco", traffic_light:"semáforo",
    fire_hydrant:"hidrante", stop_sign:"señal de pare", parking_meter:"parquímetro",
    bench:"banca", bird:"pájaro", cat:"gato", dog:"perro", horse:"caballo", sheep:"oveja",
    cow:"vaca", elephant:"elefante", bear:"oso", zebra:"cebra", giraffe:"jirafa",
    backpack:"mochila", umbrella:"paraguas", handbag:"bolso", tie:"corbata", suitcase:"maleta",
    frisbee:"frisbi", skis:"esquís", snowboard:"tabla de snowboard", sports_ball:"pelota",
    kite:"cometa", baseball_bat:"bate de béisbol", baseball_glove:"guante de béisbol",
    skateboard:"patineta", surfboard:"tabla de surf", tennis_racket:"raqueta de tenis",
    bottle:"botella", wine_glass:"copa", cup:"taza", fork:"tenedor", knife:"cuchillo",
    spoon:"cuchara", bowl:"tazón", banana:"plátano", apple:"manzana", sandwich:"sándwich",
    orange:"naranja", broccoli:"brócoli", carrot:"zanahoria", hot_dog:"perro caliente",
    pizza:"pizza", donut:"dona", cake:"pastel", chair:"silla", couch:"sofá",
    potted_plant:"planta", bed:"cama", dining_table:"mesa", toilet:"inodoro",
    tv:"televisor", laptop:"computadora portátil", mouse:"ratón", remote:"control remoto",
    keyboard:"teclado", cell_phone:"celular", microwave:"microondas", oven:"horno",
    toaster:"tostadora", sink:"lavamanos", refrigerator:"refrigeradora", book:"libro",
    clock:"reloj", vase:"florero", scissors:"tijeras", teddy_bear:"oso de peluche",
    hair_drier:"secadora de cabello", toothbrush:"cepillo de dientes",
    // Etiquetas disponibles si se carga un modelo personalizado entrenado para ellas
    glasses:"lentes", eyeglasses:"lentes", spectacles:"lentes", "optical glasses":"lentes", sunglasses:"gafas de sol",
    "wireless earbuds":"audífonos inalámbricos", "earbuds charging case":"estuche de audífonos",
    headphones:"audífonos", "headphone case":"estuche de audífonos",
    "glasses case":"estuche de lentes", "sunglasses case":"estuche de gafas",
    earbuds:"audífonos inalámbricos", wireless_earbuds:"audífonos inalámbricos",
    airpods:"audífonos inalámbricos", earbud_case:"estuche de audífonos",
    headphones_case:"estuche de audífonos", "computer mouse":"ratón",
    "smartphone":"celular", "coffee mug":"taza", "water bottle":"botella",
    "remote control":"control remoto", "game controller":"control de videojuegos",
    "power bank":"batería portátil", "flash drive":"memoria USB",
    "perfume bottle":"perfume", "coin purse":"monedero", "sunglasses":"gafas de sol",
    wallet:"billetera", keys:"llaves", key:"llave", pen:"esfero", pencil:"lápiz",
    notebook:"cuaderno", stapler:"grapadora", remote_control:"control remoto",
    charger:"cargador", power_bank:"batería portátil", watch:"reloj", monitor:"monitor",
    tablet:"tableta", game_controller:"control de videojuegos",
    "tv remote control":"control remoto de TV",
    "computer mouse":"mouse", "computer keyboard":"teclado de computadora",
    "computer monitor":"monitor", "laptop computer":"computadora portátil",
    "desktop computer":"computadora de escritorio", "phone case":"funda de celular",
    "over-ear headphones":"audífonos de diadema", "earbuds charging case":"estuche de audífonos",
    "webcam":"cámara web", "digital camera":"cámara digital", "camera lens":"lente de cámara",
    "microphone":"micrófono", "tripod":"trípode", "portable speaker":"parlante portátil",
    "bluetooth speaker":"parlante Bluetooth", "video game controller":"control de videojuegos",
    "game console":"consola de videojuegos", "wi-fi router":"router", "sd memory card":"tarjeta SD",
    "usb flash drive":"memoria USB", "usb cable":"cable USB", "phone charger":"cargador de celular",
    "power adapter":"adaptador de corriente", "extension cord":"extensión eléctrica",
    "digital wristwatch":"reloj digital", "ballpoint pen":"esfero", "marker":"marcador",
    "highlighter":"resaltador", "eraser":"borrador", "pencil sharpener":"sacapuntas",
    "geometry compass":"compás", "protractor":"transportador", "textbook":"libro de texto",
    "paper folder":"carpeta", "ring binder":"archivador", "pencil case":"cartuchera",
    "paper clips":"clips", "tape dispenser":"dispensador de cinta", "glue stick":"barra de pegamento",
    "sticky notes":"notas adhesivas", "desk organizer":"organizador de escritorio",
    "thermos":"termo", "drinking glass":"vaso", "ceramic plate":"plato",
    "electric kettle":"hervidor eléctrico", "frying pan":"sartén", "cooking pot":"olla",
    "food storage container":"recipiente para alimentos", "lunch box":"lonchera",
    "wall clock":"reloj de pared", "alarm clock":"despertador", "table lamp":"lámpara",
    "light bulb":"foco", "flashlight":"linterna", "tissue box":"caja de pañuelos",
    "soap dispenser":"dispensador de jabón", "plant pot":"maceta", "picture frame":"marco de fotos",
    "door key":"llave", "sneaker":"zapatilla", "baseball cap":"gorra",
    "deodorant stick":"desodorante", "toothpaste tube":"pasta dental",
    "hairbrush":"cepillo para cabello", "comb":"peine", "hair dryer":"secador de cabello",
    "shampoo bottle":"champú", "lotion bottle":"crema", "makeup brush":"brocha de maquillaje",
    "screwdriver":"destornillador", "hammer":"martillo", "wrench":"llave inglesa",
    "pliers":"alicates", "measuring tape":"cinta métrica", "soccer ball":"balón de fútbol",
    "basketball":"balón de baloncesto", "tennis racket":"raqueta de tenis",
    "tennis ball":"pelota de tenis", "toy car":"carro de juguete",
    "building blocks":"bloques de construcción", "playing cards":"naipes",
    "board game box":"caja de juego de mesa", "dice":"dados"
  };
  return names[label.toLowerCase()] || label.replaceAll("_", " ");
}

function drawConnections(points, connections, color, width) {
  if (!connections?.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (const link of connections) {
    const a = points[link.start ?? link[0]];
    const b = points[link.end ?? link[1]];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
    ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
    ctx.stroke();
  }
}

function drawPoints(handResult, faceResult, objectResult) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const hand of handResult.landmarks || []) {
    drawConnections(hand, HAND_CONNECTIONS, "rgba(190,145,255,0.98)", 4);
    for (const point of hand) {
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#a879ff";
      ctx.fill();
      ctx.strokeStyle = "#f4eaff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  const mesh = FaceLandmarker.FACE_LANDMARKS_TESSELATION || [];
  for (const face of (isFeatureSelected("face") ? faceResult.faceLandmarks : []) || []) {
    drawConnections(face, mesh, "rgba(110,240,185,0.42)", 1);
    for (let i = 0; i < face.length; i += 3) {
      const point = face[i];
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 1.7, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(174,255,214,0.9)";
      ctx.fill();
    }
  }
  for (const detection of (isFeatureSelected("objects") ? objectResult.detections : []).filter(item =>
    (item.categories?.[0]?.categoryName || "").toLowerCase() !== "person"
  )) {
    const box = detection.boundingBox;
    if (!box) continue;
    ctx.strokeStyle = "#43e8ff";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(box.originX, box.originY, box.width, box.height);
    const category = detection.categories?.[0];
    const caption = translateObject(category?.categoryName || "Objeto") + " " + Math.round((category?.score || 0)*100) + "%";
    ctx.font = "bold 13px Manrope, sans-serif";
    const y = Math.max(18, box.originY - 7);
    ctx.fillStyle = "rgba(4,12,22,.88)";
    ctx.fillRect(box.originX, Math.max(0, box.originY - 24), ctx.measureText(caption).width + 12, 23);
    ctx.fillStyle = "#43e8ff";
    ctx.fillText(caption, box.originX + 6, y);
  }
}

async function loadCustomGestureModel(file) {
  try {
    setMessage("Cargando modelo personalizado de señas...");
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const modelAssetBuffer = new Uint8Array(await file.arrayBuffer());
    const recognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2
    });
    gestureRecognizer = recognizer;
    handDetail.textContent = "Modelo de señas personalizado activo: " + file.name;
    setMessage("Modelo de señas cargado correctamente.");
  } catch (error) {
    console.error(error);
    setMessage("Archivo no compatible. Necesitas un modelo .task preparado para MediaPipe Gesture Recognizer.", true);
  }
}

async function loadCustomObjectModel(file) {
  try {
    setMessage("Cargando modelo personalizado...");
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const modelAssetBuffer = new Uint8Array(await file.arrayBuffer());
    const detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer, delegate: "GPU" },
      runningMode: "VIDEO",
      scoreThreshold: 0.4,
      maxResults: 10
    });
    objectDetector = detector;
    objectDetail.textContent = "Modelo personalizado activo: " + file.name;
    setMessage("Modelo personalizado cargado correctamente.");
  } catch (error) {
    console.error(error);
    setMessage("El archivo no es compatible. Necesitas un modelo TFLite de detección de objetos con metadatos MediaPipe.", true);
  }
}

function stopCamera() {
  isRunning = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  previousHandCenters.clear();
  placeholder.hidden = false;
  cameraStage.classList.remove("scanning");
  startButton.disabled = false;
  stopButton.disabled = true;
  cameraStatus.innerHTML = "<span></span> CÁMARA APAGADA";
  cameraStatus.classList.remove("active");
  appStatus.textContent = "SISTEMA EN ESPERA";
  gestureValue.textContent = "Esperando...";
  handDetail.textContent = "La IA identificará dedos y gestos.";
  faceValue.textContent = "Esperando...";
  faceDetail.textContent = "La IA analizará los rasgos visibles.";
  headValue.textContent = "Esperando...";
  headDetail.textContent = "Se estimará la orientación 3D.";
  latestBaseObjectDetections = [];
  latestZeroShotDetections = [];
  latestObjectResult = { detections: [] };
  objectValue.textContent = "Esperando...";
  objectDetail.textContent = "El detector buscará objetos.";
  faceActionValue.textContent = "Esperando...";
  faceActionDetail.textContent = "Se analizarán parpadeos y guiños.";
  headModel.style.transform = "rotateX(0deg) rotateY(0deg) rotateZ(0deg)";
  setMessage("Escáner detenido. La cámara está apagada.");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
customModelButton.addEventListener("click", () => customModelInput.click());
customModelInput.addEventListener("change", event => {
  const file = event.target.files?.[0];
  if (file) loadCustomObjectModel(file);
});
