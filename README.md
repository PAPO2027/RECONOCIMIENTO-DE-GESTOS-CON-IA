# Reconocimiento de Gestos con IA

Aplicación web experimental que utiliza la cámara del dispositivo y MediaPipe Tasks Vision para reconocer gestos comunes de las manos y analizar puntos faciales en tiempo real.

## Funciones de esta primera versión

- Reconocimiento de gestos comunes de manos.
- Detección de puntos de referencia del rostro.
- Estimación visual sencilla de sonrisa o apertura de boca.
- Estimación aproximada de la orientación de la cabeza.
- Interfaz adaptable a computadoras y celulares.

> Las expresiones mostradas son estimaciones de movimientos faciales visibles. No identifican emociones reales, pensamientos ni estados mentales.

## Tecnologías

- HTML, CSS y JavaScript.
- Vite.
- MediaPipe Tasks Vision (Gesture Recognizer y Face Landmarker).

## Ejecutar en local

Necesitas tener Node.js y npm instalados.

```bash
npm install
npm run dev
```

Abre en el navegador la dirección local que muestra Vite.

## Cámara y privacidad

El navegador solicitará permiso para acceder a la cámara. La aplicación procesa los fotogramas en el dispositivo y no contiene funciones para grabar, guardar o enviar el video. Para usar la cámara desde un sitio publicado, se necesita HTTPS.

## Publicación

El proyecto está preparado para desplegarse en Vercel. Conecta este repositorio y utiliza el comando de compilación `npm run build`; el directorio de salida de Vite es `dist`.

## Modelos

Los modelos de MediaPipe se cargan desde los recursos públicos de Google cuando se inicia la aplicación, por lo que se requiere conexión a internet durante la primera carga.
