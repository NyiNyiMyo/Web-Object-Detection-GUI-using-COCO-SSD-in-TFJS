import React from "react";
import ReactDOM from "react-dom";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import "./styles.css";

class App extends React.Component {
  videoRef = React.createRef();
  imageRef = React.createRef();
  canvasRef = React.createRef();
  fpsRef = React.createRef();
  fileInputRef = React.createRef(); // ✅ added for reset
  animationId = null;
  stream = null;

  state = {
    modelType: "lite_mobilenet_v2",
    model: null,
    running: false,
    loadingModel: false,
    confThreshold: 0.3,
    imageMode: false,
  };

  componentWillUnmount() {
    this.stopDetection();
  }

  startWebcam = async () => {
  try {
    const getMedia =
      (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ||
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia;

    if (!getMedia) {
      alert("❌ Your browser does not support camera access. Please open in Chrome or Safari.");
      return;
    }

    const constraints = { audio: false, video: { facingMode: "user" } };

    // ✅ Handle both Promise and callback-style APIs
    const stream = await new Promise((resolve, reject) => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia(constraints)
          .then(resolve)
          .catch(reject);
      } else {
        getMedia.call(navigator, constraints, resolve, reject);
      }
    });

    this.stream = stream;
    const videoEl = this.videoRef.current;
    videoEl.srcObject = stream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => resolve();
    });

    // ✅ Ensure playback starts on iOS
    if (videoEl.play) {
      await videoEl.play().catch(() => {});
    }

    console.log("✅ Webcam ready (polyfilled)");
  } catch (err) {
    console.error("❌ Webcam error:", err);
    alert("Camera access failed. Please allow permissions or use Safari/Chrome.");
  }
};

  stopWebcam = () => {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      console.log("🛑 Webcam stopped");
    }
  };

  loadModel = async (modelType) => {
    this.setState({ loadingModel: true });
    const model = await cocoSsd.load({ base: modelType });
    this.setState({ model, loadingModel: false });
    console.log("✅ Model loaded:", modelType);
    return model;
  };

  startDetection = async () => {
    if (this.state.running) return;

    await this.startWebcam();
    const model =
      this.state.model || (await this.loadModel(this.state.modelType));

    const video = this.videoRef.current;
    const waitForVideo = () =>
      new Promise((resolve) => {
        const check = () => {
          if (video.videoWidth > 0 && video.videoHeight > 0) resolve();
          else requestAnimationFrame(check);
        };
        check();
      });
    await waitForVideo();

    this.setState({ running: true, imageMode: false });
    this.lastTime = performance.now();
    this.detectFrame(video, model);
  };

  stopDetection = () => {
    cancelAnimationFrame(this.animationId);
    this.stopWebcam();
    this.setState({ running: false });

    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (this.fpsRef.current) this.fpsRef.current.innerText = "FPS: --";
  };

  toggleDetection = async () => {
    if (this.state.running) this.stopDetection();
    else await this.startDetection();
  };

  detectFrame = (video, model) => {
    if (!this.state.running || !model) return;

    model.detect(video).then((predictions) => {
      const now = performance.now();
      const fps = (1000 / (now - this.lastTime)).toFixed(1);
      this.lastTime = now;
      if (this.fpsRef.current) this.fpsRef.current.innerText = `FPS: ${fps}`;
      this.renderPredictions(predictions, video);
      this.animationId = requestAnimationFrame(() =>
        this.detectFrame(video, model)
      );
    });
  };

  handleImageSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    this.stopDetection();

    const model =
      this.state.model || (await this.loadModel(this.state.modelType));

    const img = this.imageRef.current;
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      this.setState({ imageMode: true });

      const predictions = await model.detect(img);
      this.renderPredictions(predictions, img);

      // ✅ reset so same file triggers again
      if (this.fileInputRef.current) this.fileInputRef.current.value = "";
    };
  };

  renderPredictions = (predictions, sourceEl) => {
    const canvas = this.canvasRef.current;
    const ctx = canvas.getContext("2d");

    const srcWidth = sourceEl.videoWidth || sourceEl.width;
    const srcHeight = sourceEl.videoHeight || sourceEl.height;
    const displayWidth = sourceEl.clientWidth;
    const displayHeight = sourceEl.clientHeight;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const aspectVideo = srcWidth / srcHeight;
    const aspectDisplay = displayWidth / displayHeight;

    let scaleX,
      scaleY,
      offsetX = 0,
      offsetY = 0;
    if (aspectVideo > aspectDisplay) {
      scaleX = displayWidth / srcWidth;
      scaleY = scaleX;
      offsetY = (displayHeight - srcHeight * scaleY) / 2;
    } else {
      scaleY = displayHeight / srcHeight;
      scaleX = scaleY;
      offsetX = (displayWidth - srcWidth * scaleX) / 2;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions
      .filter((p) => p.score > this.state.confThreshold)
      .forEach((pred) => {
        const [x, y, width, height] = pred.bbox;
        const scaledX = x * scaleX + offsetX;
        const scaledY = y * scaleY + offsetY;
        const scaledW = width * scaleX;
        const scaledH = height * scaleY;

        const text = `${pred.class} ${(pred.score * 100).toFixed(1)}%`;

        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

        ctx.fillStyle = "#00FFFF";
        const textWidth = ctx.measureText(text).width;
        const textHeight = parseInt(font, 10);
        ctx.fillRect(scaledX, scaledY, textWidth + 4, textHeight + 4);

        ctx.fillStyle = "#000000";
        ctx.fillText(text, scaledX, scaledY);
      });
  };

  handleModelChange = async (e) => {
    const newModelType = e.target.value;
    this.setState({ modelType: newModelType, model: null });
    if (this.state.running) {
      this.stopDetection();
      await this.loadModel(newModelType);
      this.startDetection();
    }
  };

  render() {
    const { modelType, loadingModel } = this.state;

    return (
      <div style={{ textAlign: "center" }}>
        <h2>Makers - Real-Time Object Detection (COCO-SSD)</h2>
        <h3>TensorFlow.js — 80 Classes</h3>

        {/* --- Model Switcher --- */}
        <div style={{ marginBottom: "10px" }}>
          <label style={{ fontWeight: "bold", marginRight: "8px" }}>
            Select Model:
          </label>
          <select
            value={modelType}
            onChange={this.handleModelChange}
            style={{
              padding: "6px 10px",
              borderRadius: "8px",
              fontSize: "14px",
            }}
          >
            <option value="lite_mobilenet_v2">Lite MobileNet V2 (Fast)</option>
            <option value="mobilenet_v1">MobileNet V1 (Accurate)</option>
          </select>
        </div>

        {loadingModel && (
          <div style={{ color: "#FF4444", marginBottom: "8px" }}>
            Loading model, please wait...
          </div>
        )}

        <div
          ref={this.fpsRef}
          style={{
            fontSize: "18px",
            fontWeight: "bold",
            color: "#00FFFF",
            marginBottom: "10px",
          }}
        >
          FPS: --
        </div>

        {/* --- Video + Canvas + Controls --- */}
        <div
          style={{
            position: "relative",
            display: "inline-block",
          }}
        >
          {/* ✅ keep same h/w size as before for image */}
          <video
  ref={this.videoRef}
  autoPlay
  playsInline
  muted
  style={{
    display: this.state.imageMode ? "none" : "block",
    borderRadius: "10px",
    boxShadow: "0 0 10px #ccc",
    width: "90vw",
    maxWidth: "640px",
    height: "auto",
    aspectRatio: "4 / 3", // remove this line
  }}
/>

<img
  ref={this.imageRef}
  alt=""
  style={{
    display: this.state.imageMode ? "block" : "none",
    borderRadius: "10px",
    boxShadow: "0 0 10px #ccc",
    width: "90vw",
    maxWidth: "640px",
    height: "auto",
    objectFit: "contain", // ✅ show full image, preserve aspect
    backgroundColor: "black", // optional – looks better for letterbox areas
  }}
/>

          <canvas
            ref={this.canvasRef}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              zIndex: 1,
              width: "100%",
              height: "100%",
            }}
          />

          {/* --- Buttons --- */}
<div
  style={{
    position: "absolute",
    bottom: window.innerWidth < 600 ? "18px" : "12px", // slightly higher on mobile
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    gap: "10px",
    zIndex: 2,
    flexWrap: "nowrap",
  }}
>
  <button
    onClick={this.toggleDetection}
    disabled={this.state.loadingModel}
    style={{
      backgroundColor: this.state.running ? "#FF5555" : "#00CC88",
      color: "#fff",
      border: "none",
      borderRadius: "25px",
      padding: window.innerWidth < 600 ? "6px 12px" : "8px 16px",
      fontSize: window.innerWidth < 600 ? "12px" : "14px",
      fontWeight: "bold",
      cursor: "pointer",
      boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
      whiteSpace: "nowrap", // ✅ prevent 2-line wrapping
      flexShrink: 0,
    }}
  >
    {this.state.running ? "⏹ Stop" : "▶ Start"}
  </button>

  <label
    style={{
      backgroundColor: "#007BFF",
      color: "#fff",
      border: "none",
      borderRadius: "25px",
      padding: window.innerWidth < 600 ? "6px 12px" : "8px 16px",
      fontSize: window.innerWidth < 600 ? "12px" : "14px",
      fontWeight: "bold",
      cursor: "pointer",
      boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
      whiteSpace: "nowrap", // ✅ no multi-line
      flexShrink: 0,
    }}
  >
    📷 Select Image
    <input
      ref={this.fileInputRef}
      type="file"
      accept="image/*"
      onChange={this.handleImageSelect}
      style={{ display: "none" }}
    />
  </label>
</div>

          {/* --- Confidence Slider --- */}
          <div
            style={{
              position: "absolute",
              top: "50%",
              right: "-60px",
              transform: "translateY(-50%) rotate(-90deg)",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              zIndex: 2,
            }}
          >
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={this.state.confThreshold}
              onChange={(e) =>
                this.setState({ confThreshold: parseFloat(e.target.value) })
              }
              style={{
                width: "120px",
                accentColor: "#00ffbfff",
                cursor: "pointer",
              }}
            />
            <span
              style={{
                color: "#9000ffff",
                fontSize: "12px",
                fontWeight: "bold",
                transform: "rotate(90deg)",
              }}
            >
              {`Conf: ${this.state.confThreshold.toFixed(2)}`}
            </span>
          </div>
        </div>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
