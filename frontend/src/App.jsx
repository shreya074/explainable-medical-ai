import { useEffect, useRef, useState } from "react";
import "./App.css";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/jpg",
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [error, setError] = useState("");

  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);

  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);


  // ============================================================
  // CLEANUP PREVIEW URL
  // ============================================================

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);


  // ============================================================
  // VALIDATE FILE
  // ============================================================

  const validateFile = (file) => {

    if (!file) {
      return "No file selected.";
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Please select a JPG, JPEG, or PNG image.";
    }

    if (file.size > MAX_FILE_SIZE) {
      return "File size must be less than 10 MB.";
    }

    if (file.size === 0) {
      return "The selected file is empty.";
    }

    return null;
  };


  // ============================================================
  // PROCESS SELECTED FILE
  // ============================================================

  const processFile = (file) => {

    setError("");
    setResult(null);

    const validationError =
      validateFile(file);

    if (validationError) {

      setSelectedFile(null);
      setPreviewUrl(null);

      setError(validationError);

      return;
    }

    setSelectedFile(file);

    const url = URL.createObjectURL(file);

    setPreviewUrl(url);
  };


  // ============================================================
  // FILE INPUT
  // ============================================================

  const handleFileChange = (event) => {

    const file = event.target.files[0];

    processFile(file);
  };


  // ============================================================
  // DRAG OVER
  // ============================================================

  const handleDragOver = (event) => {

    event.preventDefault();

    setIsDragging(true);
  };


  // ============================================================
  // DRAG LEAVE
  // ============================================================

  const handleDragLeave = (event) => {

    event.preventDefault();

    setIsDragging(false);
  };


  // ============================================================
  // DROP
  // ============================================================

  const handleDrop = (event) => {

    event.preventDefault();

    setIsDragging(false);

    const file =
      event.dataTransfer.files[0];

    processFile(file);
  };


  // ============================================================
  // ANALYZE IMAGE
  // ============================================================

  const handleAnalyze = async () => {

    if (!selectedFile) {

      setError(
        "Please upload a chest X-ray first."
      );

      return;
    }

    if (loading) {
      return;
    }

    setError("");
    setResult(null);
    setLoading(true);


    try {

      // --------------------------------------------------------
      // Create form data
      // --------------------------------------------------------

      const formData = new FormData();

      formData.append(
        "file",
        selectedFile
      );


      // --------------------------------------------------------
      // Request backend
      // --------------------------------------------------------

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );


      // --------------------------------------------------------
      // Parse response
      // --------------------------------------------------------

      let data;

      try {

        data = await response.json();

      } catch {

        throw new Error(
          "The AI backend returned an invalid response."
        );
      }


      // --------------------------------------------------------
      // Handle API errors
      // --------------------------------------------------------

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "The AI analysis failed."
        );
      }


      // --------------------------------------------------------
      // Validate response
      // --------------------------------------------------------

      if (
        !data.prediction ||
        data.confidence === undefined
      ) {

        throw new Error(
          "The backend returned an incomplete analysis result."
        );
      }


      // --------------------------------------------------------
      // Store result
      // --------------------------------------------------------

      setResult(data);

    } catch (err) {

      console.error(
        "Prediction error:",
        err
      );


      // --------------------------------------------------------
      // Connection error
      // --------------------------------------------------------

      if (
        err instanceof TypeError
      ) {

        setError(
          "Unable to connect to the AI backend. " +
          "Please make sure the FastAPI server is running."
        );

      } else {

        setError(
          err.message ||
          "Something went wrong during analysis."
        );
      }

    } finally {

      setLoading(false);
    }
  };


  // ============================================================
  // RESET APPLICATION
  // ============================================================

  const handleReset = () => {

    setSelectedFile(null);

    setPreviewUrl(null);

    setResult(null);

    setError("");

    setIsDragging(false);


    if (fileInputRef.current) {

      fileInputRef.current.value = "";

    }

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };


  // ============================================================
  // OPEN FILE SELECTOR
  // ============================================================

  const openFileSelector = () => {

    if (!loading) {

      fileInputRef.current?.click();

    }
  };


  // ============================================================
  // PREDICTION CLASS
  // ============================================================

  const getPredictionClass = () => {

    if (!result) {
      return "";
    }

    if (
      result.prediction === "PNEUMONIA"
    ) {

      return "prediction-pneumonia";

    }

    return "prediction-normal";
  };


  // ============================================================
  // CONFIDENCE CLASS
  // ============================================================

  const getConfidenceClass = () => {

    if (!result) {
      return "";
    }

    if (result.confidence >= 80) {
      return "confidence-high";
    }

    if (result.confidence >= 60) {
      return "confidence-medium";
    }

    return "confidence-low";
  };


  // ============================================================
  // MAIN UI
  // ============================================================

  return (

    <div className="app">


      {/* ======================================================
          HEADER
          ====================================================== */}

      <header className="header">

        <div className="header-content">

          <div className="brand-icon">
            AI
          </div>

          <div>

            <h1>
              Explainable Medical AI
            </h1>

            <p>
              Chest X-Ray Classification & Explanation
            </p>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN
          ====================================================== */}

      <main className="main-container">


        {/* ====================================================
            UPLOAD SECTION
            ==================================================== */}

        <section className="upload-card">

          <div className="section-heading">

            <h2>
              Chest X-Ray Analysis
            </h2>

            <p className="subtitle">
              Upload a chest X-ray image for AI-assisted
              classification.
            </p>

          </div>


          {/* ==================================================
              UPLOAD AREA
              ================================================== */}

          {!previewUrl && (

            <div
              className={`upload-area ${
                isDragging
                  ? "upload-area-dragging"
                  : ""
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={openFileSelector}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {

                if (
                  event.key === "Enter" ||
                  event.key === " "
                ) {

                  openFileSelector();

                }

              }}
            >

              <div className="upload-icon">
                ↑
              </div>

              <h3>
                {isDragging
                  ? "Drop X-Ray Here"
                  : "Upload Chest X-Ray"
                }
              </h3>

              <p>
                {isDragging
                  ? "Release the file to upload it"
                  : "Drag and drop your image here or choose a file"
                }
              </p>

              <span className="choose-button">
                Choose Image
              </span>

              <small>
                JPG, JPEG or PNG • Maximum 10 MB
              </small>

            </div>

          )}


          <input
            ref={fileInputRef}
            id="file-upload"
            type="file"
            accept=".jpg,.jpeg,.png"
            onChange={handleFileChange}
            hidden
          />


          {/* ==================================================
              ERROR MESSAGE
              ================================================== */}

          {error && (

            <div className="error-message">

              <span className="error-icon">
                !
              </span>

              <div>

                <strong>
                  Analysis Error
                </strong>

                <p>
                  {error}
                </p>

              </div>

            </div>

          )}


          {/* ==================================================
              PREVIEW
              ================================================== */}

          {previewUrl && (

            <div className="preview-section">

              <div className="preview-header">

                <h3>
                  Selected X-Ray
                </h3>

                <button
                  className="change-button"
                  onClick={handleReset}
                  type="button"
                  disabled={loading}
                >
                  Change Image
                </button>

              </div>


              <div className="preview-container">

                <img
                  src={previewUrl}
                  alt="Selected chest X-ray"
                  className="xray-preview"
                />

              </div>


              <div className="file-information">

                <span className="file-icon">
                  IMG
                </span>

                <div>

                  <strong>
                    {selectedFile.name}
                  </strong>

                  <span>
                    {(selectedFile.size / 1024).toFixed(1)}
                    {" "}KB
                  </span>

                </div>

              </div>

            </div>

          )}


          {/* ==================================================
              ANALYZE BUTTON
              ================================================== */}

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={
              !selectedFile ||
              loading
            }
          >

            {loading ? (

              <>
                <span className="spinner"></span>

                Analyzing X-Ray...
              </>

            ) : (

              <>
                Analyze X-Ray

                <span className="button-arrow">
                  →
                </span>
              </>

            )}

          </button>


          {/* ==================================================
              LOADING MESSAGE
              ================================================== */}

          {loading && (

            <div className="loading-message">

              <span className="loading-dot"></span>

              AI model is analyzing the uploaded X-ray.
              Please wait...

            </div>

          )}

        </section>


        {/* ====================================================
            RESULT
            ==================================================== */}

        {result && (

          <section className="result-card">

            {/* Result Header */}

            <div className="result-header">

              <div>

                <p className="result-overline">
                  AI ANALYSIS
                </p>

                <h2>
                  Analysis Result
                </h2>

              </div>

              <div className="completed-badge">
                ✓ Complete
              </div>

            </div>


            {/* =================================================
                PREDICTION
                ================================================= */}

            <div
              className={`prediction-banner ${
                getPredictionClass()
              }`}
            >

              <span className="prediction-label">
                CLASSIFICATION
              </span>

              <strong>
                {result.prediction}
              </strong>

              <span>
                AI model classification result
              </span>

            </div>


            {/* =================================================
                RESULT SUMMARY
                ================================================= */}

            <div className="result-grid">


              <div className="result-item">

                <span className="result-label">
                  Prediction
                </span>

                <strong>
                  {result.prediction}
                </strong>

              </div>


              <div className="result-item">

                <span className="result-label">
                  Confidence
                </span>

                <strong>
                  {result.confidence}%
                </strong>

              </div>


              <div className="result-item">

                <span className="result-label">
                  Confidence Level
                </span>

                <strong
                  className={
                    getConfidenceClass()
                  }
                >
                  {result.confidence_level}
                </strong>

              </div>

            </div>


            {/* =================================================
                CONFIDENCE
                ================================================= */}

            <div className="confidence-section">

              <div className="confidence-header">

                <div>

                  <h3>
                    Model Confidence
                  </h3>

                  <p>
                    Confidence associated with this prediction
                  </p>

                </div>

                <strong>
                  {result.confidence}%
                </strong>

              </div>


              <div className="confidence-bar">

                <div
                  className={`confidence-fill ${
                    getConfidenceClass()
                  }`}
                  style={{
                    width: `${Math.min(
                      Math.max(
                        result.confidence,
                        0
                      ),
                      100
                    )}%`,
                  }}
                ></div>

              </div>


              <div className="confidence-scale">

                <span>0%</span>

                <span>60%</span>

                <span>80%</span>

                <span>100%</span>

              </div>


              <div
                className={`confidence-status ${
                  getConfidenceClass()
                }`}
              >

                <strong>

                  {result.confidence_level}
                  {" "}CONFIDENCE

                </strong>

                <span>

                  {result.confidence >= 80
                    ? "The model shows high confidence in this classification."
                    : result.confidence >= 60
                    ? "The model shows moderate confidence in this classification."
                    : "The model shows low confidence in this classification."
                  }

                </span>

              </div>

            </div>


            {/* =================================================
                GRAD-CAM
                ================================================= */}

            <div className="explanation-section">

              <div className="explanation-header">

                <div>

                  <p className="result-overline">
                    EXPLAINABLE AI
                  </p>

                  <h3>
                    Grad-CAM Explanation
                  </h3>

                  <p>
                    Visualization of image regions that
                    contributed to the model prediction.
                  </p>

                </div>

                <div className="gradcam-badge">
                  Grad-CAM
                </div>

              </div>


              <div className="explanation-grid">


                {/* Original */}

                <div className="explanation-card">

                  <div className="explanation-card-header">

                    <h4>
                      Original X-Ray
                    </h4>

                    <span>
                      Input
                    </span>

                  </div>

                  <div className="explanation-image-container">

                    <img
                      src={previewUrl}
                      alt="Original chest X-ray"
                      className="explanation-image"
                    />

                  </div>

                </div>


                {/* Grad-CAM */}

                <div className="explanation-card">

                  <div className="explanation-card-header">

                    <h4>
                      Grad-CAM Heatmap
                    </h4>

                    <span>
                      AI Explanation
                    </span>

                  </div>

                  <div className="explanation-image-container">

                    {result.explanation &&
                    result.explanation.heatmap ? (

                      <img
                        src={`data:image/png;base64,${result.explanation.heatmap}`}
                        alt="Grad-CAM explanation heatmap"
                        className="explanation-image"
                      />

                    ) : (

                      <div className="no-explanation">

                        Grad-CAM explanation unavailable.

                      </div>

                    )}

                  </div>

                </div>

              </div>


              <div className="explanation-description">

                <span className="info-icon">
                  i
                </span>

                <div>

                  <strong>
                    How to interpret this visualization
                  </strong>

                  <p>
                    The Grad-CAM heatmap highlights image
                    regions that had greater influence on the
                    model's classification. This visualization
                    is provided for explainability and academic
                    demonstration.
                  </p>

                </div>

              </div>

            </div>


            {/* =================================================
                SUCCESS
                ================================================= */}

            <div className="success-message">

              <span className="success-icon">
                ✓
              </span>

              <div>

                <strong>
                  Analysis completed successfully
                </strong>

                <p>
                  The AI model has processed the uploaded
                  X-ray image.
                </p>

              </div>

            </div>


            {/* =================================================
                DISCLAIMER
                ================================================= */}

            <div className="result-disclaimer">

              <strong>
                Academic Prototype
              </strong>

              <p>
                This AI system is developed for academic
                demonstration and is not intended for
                clinical diagnosis.
              </p>

            </div>


            {/* =================================================
                RESET
                ================================================= */}

            <button
              className="reset-button"
              onClick={handleReset}
              type="button"
            >
              Analyze Another Image
            </button>

          </section>

        )}


        {/* ====================================================
            PIPELINE
            ==================================================== */}

        {!result && (

          <section className="info-card">

            <h2>
              AI Analysis Pipeline
            </h2>

            <div className="pipeline">

              <div className="pipeline-item">
                <span>1</span>
                <p>Upload X-Ray</p>
              </div>

              <div className="pipeline-arrow">
                →
              </div>

              <div className="pipeline-item">
                <span>2</span>
                <p>Preprocessing</p>
              </div>

              <div className="pipeline-arrow">
                →
              </div>

              <div className="pipeline-item">
                <span>3</span>
                <p>AI Prediction</p>
              </div>

              <div className="pipeline-arrow">
                →
              </div>

              <div className="pipeline-item">
                <span>4</span>
                <p>Confidence</p>
              </div>

              <div className="pipeline-arrow">
                →
              </div>

              <div className="pipeline-item">
                <span>5</span>
                <p>Grad-CAM</p>
              </div>

            </div>

          </section>

        )}

      </main>


      {/* ======================================================
          FOOTER
          ====================================================== */}

      <footer className="footer">

        <p>
          Explainable Medical AI • Academic Project
        </p>

        <p>
          Not intended for clinical diagnosis
        </p>

      </footer>

    </div>
  );
}

export default App;