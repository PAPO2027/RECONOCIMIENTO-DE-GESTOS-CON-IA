import { FilesetResolver, GestureRecognizer, FaceLandmarker } from "@mediapipe/tasks-vision";
import "./style.css";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
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

const MODELS = {
  gesture: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
};

let stream = null;
let gestureRecognizer = null;
let faceLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let isRunning = false;

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
  message.style.color = isError ? "#ff9b9b" : "";
}

async function loadModels() {
  setMessage("Cargando los modelos de inteligencia artificial. La primera carga puede tardar un poco.");
  appStatus.textContent = "Cargando IA";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  [gestureRecognizer, faceLandmarker] = await Promise.all([
    GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.gesture, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2
    }),
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.face, delegate: "CPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    })
  ]);
}

async function startCamera() {
  startButton.disabled = true;
  try {
    if (!gestureRecognizer || !faceLandmarker) await loadModels();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    isRunning = true;
    lastVideoTime = -1;
    placeholder.hidden = true;
    stopButton.disabled = false;
    cameraStatus.textContent = "Cámara activa";
    cameraStatus.classList.add("active");
    appStatus.textContent = "Analizando";
    setMessage("Cámara activa. Coloca tus manos y rostro frente a ella.");
    renderLoop();
  } catch (error) {
    console.error(error);
    const reason = error.name === "NotAllowedError"
      ? "No se concedió permiso para usar la cámara. Revisa los permisos del navegador."
      : error.name === "NotFoundError"
        ? "No se encontró una cámara disponible."
        : "No se pudo iniciar la cámara o cargar los modelos. Comprueba tu conexión e inténtalo de nuevo.";
    setMessage(reason, true);
    appStatus.textContent = "Revisar configuración";
    startButton.disabled = false;
  }
}

function renderLoop() {
  if (!isRunning) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const timestamp = performance.now();
    const handResult = gestureRecognizer.recognizeForVideo(video, timestamp);
    const faceResult = faceLandmarker.detectForVideo(video, timestamp);
    showHands(handResult);
    showFace(faceResult);
    drawPoints(handResult, faceResult);
  }
  animationId = requestAnimationFrame(renderLoop);
}

function showHands(result) {
  const gestures = result.gestures ?? [];
  if (!gestures.length) {
    gestureValue.textContent = "No detectada";
    handDetail.textContent = "Muestra una mano dentro de la imagen.";
    return;
  }
  const best = gestures[0][0];
  const label = gestureNames[best.categoryName] || best.categoryName || "Gesto detectado";
  const percent = Math.round((best.score || 0) * 100);
  gestureValue.textContent = label;
  handDetail.textContent = `Confianza del modelo: ${percent}% · Manos detectadas: ${gestures.length}`;
}

function showFace(result) {
  if (!result.faceLandmarks?.length) {
    faceValue.textContent = "No detectado";
    faceDetail.textContent = "Mira hacia la cámara para analizar puntos faciales.";
    headValue.textContent = "—";
    headDetail.textContent = "Sin rostro para estimar la orientación.";
    return;
  }

  const landmarks = result.faceLandmarks[0];
  const categories = result.faceBlendshapes?.[0]?.categories || [];
  const scores = Object.fromEntries(categories.map(item => [item.categoryName, item.score]));
  const smile = ((scores.mouthSmileLeft || 0) + (scores.mouthSmileRight || 0)) / 2;
  const mouthOpen = scores.jawOpen || 0;

  let visualLabel = "Expresión neutral aproximada";
  if (smile > 0.45) visualLabel = "Sonrisa detectada";
  else if (mouthOpen > 0.5) visualLabel = "Boca abierta";

  faceValue.textContent = visualLabel;
  faceDetail.textContent = "Estimación basada en movimientos faciales visibles; no determina emociones.";

  // Estimación sencilla de orientación a partir de la posición relativa de la nariz y los ojos.
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const horizontalOffset = nose.x - eyeCenterX;
  const verticalOffset = nose.y - eyeCenterY;
  let direction = "De frente";
  if (horizontalOffset > 0.035) direction = "Giro hacia un lado";
  if (horizontalOffset < -0.035) direction = "Giro hacia el otro lado";
  if (verticalOffset < 0.12) direction = "Cabeza inclinada hacia arriba";
  if (verticalOffset > 0.22) direction = "Cabeza inclinada hacia abajo";
  headValue.textContent = direction;
  headDetail.textContent = "Orientación aproximada; puede variar según la posición y la cámara.";
}

function drawPoints(handResult, faceResult) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#b9f27c";
  for (const hand of handResult.landmarks || []) {
    for (const point of hand) {
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#75d9ff";
  for (const face of faceResult.faceLandmarks || []) {
    for (let i = 0; i < face.length; i += 5) {
      const point = face[i];
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function stopCamera() {
  isRunning = false;
  if (animationId) cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  cameraStatus.textContent = "Cámara apagada";
  cameraStatus.classList.remove("active");
  appStatus.textContent = "En espera";
  gestureValue.textContent = "—";
  handDetail.textContent = "Esperando una mano frente a la cámara.";
  faceValue.textContent = "—";
  faceDetail.textContent = "Esperando un rostro.";
  headValue.textContent = "—";
  headDetail.textContent = "Se estimará usando puntos faciales.";
  setMessage("Cámara detenida. El video no se guarda.");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
