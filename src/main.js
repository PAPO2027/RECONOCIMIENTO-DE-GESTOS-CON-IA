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
    cameraStage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
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
  const movement = getHandMovement(result.landmarks || [], result.handednesses || []);
  handDetail.textContent = `${gestures.length} mano(s) reconocida(s) · Confianza principal: ${confidence}% · ${movement}`;
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

function getHandMovement(hands, handednesses) {
  previousFrameTime = performance.now();
  const currentKeys = new Set();
  let label = "Movimiento estable";
  hands.forEach((hand, index) => {
    if (!hand.length) return;
    const side = handednesses[index]?.[0]?.categoryName || `mano-${index}`;
    currentKeys.add(side);
    const wrist = hand[0];
    const previous = previousHandCenters.get(side);
    if (previous) {
      const dx = wrist.x - previous.x;
      const dy = wrist.y - previous.y;
      if (Math.hypot(dx, dy) > 0.012) {
        if (Math.abs(dx) > Math.abs(dy)) label = dx > 0 ? "Movimiento hacia la derecha" : "Movimiento hacia la izquierda";
        else label = dy > 0 ? "Movimiento hacia abajo" : "Movimiento hacia arriba";
      }
    }
    previousHandCenters.set(side, { x: wrist.x, y: wrist.y });
  });
  for (const key of previousHandCenters.keys()) if (!currentKeys.has(key)) previousHandCenters.delete(key);
  return label;
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

function drawPoints(handResult, faceResult) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Manos: estructura completa, líneas gruesas y articulaciones muy visibles.
  for (const hand of handResult.landmarks || []) {
    drawConnections(hand, HAND_CONNECTIONS, "rgba(190,145,255,0.98)", 4);
    for (let i = 0; i < hand.length; i++) {
      const point = hand[i];
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, i === 0 ? 7 : 5.5, 0, Math.PI * 2);
      ctx.fillStyle = "#a879ff";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = "#f4eaff";
      ctx.fill();
    }
  }

  // Rostro: malla triangular completa más contornos de ojos, labios y cara.
  const mesh = FaceLandmarker.FACE_LANDMARKS_TESSELATION || [];
  const faceGroups = [
    [FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, "#75f0b7", 3.2],
    [FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, "#b5ff9b", 2.5],
    [FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, "#b5ff9b", 2.5],
    [FaceLandmarker.FACE_LANDMARKS_LIPS, "#75f0b7", 2.8],
    [FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, "#75f0b7", 2.2],
    [FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, "#75f0b7", 2.2]
  ];

  for (const face of faceResult.faceLandmarks || []) {
    drawConnections(face, mesh, "rgba(110,240,185,0.48)", 1.15);
    for (const [connections, color, width] of faceGroups) {
      drawConnections(face, connections || [], color, width);
    }

    // Puntos más densos para que la malla se aprecie como en la referencia.
    for (let i = 0; i < face.length; i += 2) {
      const point = face[i];
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 2.15, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(174,255,214,0.9)";
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
  previousHandCenters.clear();
  previousFrameTime = 0;
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
