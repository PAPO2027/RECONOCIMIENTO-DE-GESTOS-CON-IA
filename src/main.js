import { FilesetResolver, GestureRecognizer, FaceLandmarker } from "@mediapipe/tasks-vision";
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
  message.style.color = isError ? "#ff8fa3" : "";
}

async function loadModels() {
  setMessage("Preparando inteligencia artificial. La primera carga puede tardar unos segundos.");
  appStatus.textContent = "CARGANDO MODELOS";
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
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("Tu navegador no permite acceder a la cámara. Abre la página mediante HTTPS o desde localhost.", true);
    return;
  }
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
  handDetail.textContent = `${gestures.length} mano(s) reconocida(s) · Confianza principal: ${confidence}%`;
}

function showFace(result) {
  if (!result.faceLandmarks?.length) {
    faceValue.textContent = "Rostro no detectado";
    faceDetail.textContent = "Coloca tu rostro dentro del área de la cámara.";
    headValue.textContent = "Sin datos";
    headDetail.textContent = "Se necesita detectar un rostro para estimar su orientación.";
    return;
  }

  const landmarks = result.faceLandmarks[0];
  const categories = result.faceBlendshapes?.[0]?.categories || [];
  const scores = Object.fromEntries(categories.map(item => [item.categoryName, item.score]));
  const smile = ((scores.mouthSmileLeft || 0) + (scores.mouthSmileRight || 0)) / 2;
  const mouthOpen = scores.jawOpen || 0;

  let visualLabel = "Rostro detectado";
  if (smile > 0.45) visualLabel = "Sonrisa detectada";
  else if (mouthOpen > 0.5) visualLabel = "Boca abierta";
  faceValue.textContent = visualLabel;
  faceDetail.textContent = "Se detectaron puntos faciales. La etiqueta describe rasgos visibles, no emociones.";

  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const horizontalOffset = nose.x - eyeCenterX;
  const verticalOffset = nose.y - eyeCenterY;

  let direction = "De frente";
  if (Math.abs(horizontalOffset) > 0.035) {
    direction = horizontalOffset > 0 ? "Girando hacia un lado" : "Girando hacia el otro lado";
  }
  if (verticalOffset < 0.12) direction = "Mirando hacia arriba";
  else if (verticalOffset > 0.22) direction = "Mirando hacia abajo";
  headValue.textContent = direction;
  headDetail.textContent = "Estimación aproximada según la posición de los puntos faciales.";
}

function drawPoints(handResult, faceResult) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#43e8ff";
  for (const hand of handResult.landmarks || []) {
    for (const point of hand) {
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#b69aff";
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
  animationId = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.hidden = false;
  cameraStage.classList.remove("scanning");
  startButton.disabled = false;
  stopButton.disabled = true;
  cameraStatus.innerHTML = "<span></span> CÁMARA APAGADA";
  cameraStatus.classList.remove("active");
  appStatus.textContent = "SISTEMA EN ESPERA";
  gestureValue.textContent = "Esperando...";
  handDetail.textContent = "La IA identificará gestos cuando actives la cámara.";
  faceValue.textContent = "Esperando...";
  faceDetail.textContent = "La IA buscará un rostro y analizará sus rasgos visibles.";
  headValue.textContent = "Esperando...";
  headDetail.textContent = "Se estimará la orientación de la cabeza.";
  setMessage("Escáner detenido. La cámara está apagada.");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
