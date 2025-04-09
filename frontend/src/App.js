import React, { useState } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Spinner,
  Alert,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import ImageDisplay from "./components/ImageDisplay";
import ShapeResult from "./components/ShapeResult";
import SizeResult from "./components/SizeResult";
import ColorResult from "./components/ColorResult";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [processedImageUrl, setProcessedImageUrl] = useState("");
  const [shapes, setShapes] = useState([]);
  const [measurements, setMeasurements] = useState([]);
  const [colors, setColors] = useState([]);
  const [combinedResults, setCombinedResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("upload"); // 'upload', 'shape', 'size', 'color', or 'combined'

  const handleFileUpload = async (file) => {
    setFile(file);
    setProcessedImageUrl("");
    setShapes([]);
    setMeasurements([]);
    setColors([]);
    setCombinedResults([]);
    setError("");

    const fileUrl = URL.createObjectURL(file);
    setImageUrl(fileUrl);

    try {
      setLoading(true);
      setActiveTab("upload");

      const formData = new FormData();
      formData.append("image", file);

      const response = await axios.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setFilename(response.data.filename);
      setLoading(false);
    } catch (err) {
      setError("Error uploading image. Please try again.");
      setLoading(false);
    }
  };

  const handleDetectShape = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("shape");
      setProcessedImageUrl("");
      setShapes([]);
      setError("");

      const response = await axios.post("/detect-shape", { filename });
      setShapes(response.data.shapes);
      setProcessedImageUrl(response.data.processedImage);
      setLoading(false);
    } catch (err) {
      setError("Error detecting shapes. Please try again.");
      setLoading(false);
    }
  };

  const handleDetectSize = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("size");
      setProcessedImageUrl("");
      setMeasurements([]);
      setError("");

      const response = await axios.post("/detect-size", { filename });
      setMeasurements(response.data.measurements);
      setProcessedImageUrl(response.data.processedImage);
      setLoading(false);
    } catch (err) {
      setError("Error detecting sizes. Please try again.");
      setLoading(false);
    }
  };

  const handleDetectColor = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("color");
      setProcessedImageUrl("");
      setColors([]);
      setError("");

      const response = await axios.post("/detect-color", { filename });
      setColors(response.data.colors);
      setProcessedImageUrl(response.data.processedImage);
      setLoading(false);
    } catch (err) {
      setError("Error detecting colors. Please try again.");
      setLoading(false);
    }
  };

  const handleDetectAll = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("combined");
      setProcessedImageUrl("");
      setCombinedResults([]);
      setError("");

      const response = await axios.post("/detect-shape-color", { filename });
      setCombinedResults(response.data.results);
      setProcessedImageUrl(response.data.processedImage);
      setLoading(false);
    } catch (err) {
      setError("Error in combined detection. Please try again.");
      setLoading(false);
    }
  };

  return (
    <Container
      fluid
      className="px-3 py-4"
      style={{ maxWidth: "1200px", margin: "0 auto" }}
    >
      <Card className="shadow-sm">
        <Card.Header className="bg-primary text-white text-center py-3">
          <h2 className="mb-0">Shape, Size, and Color Detector</h2>
        </Card.Header>
        <Card.Body className="px-4 py-4">
          <Row className="mb-4">
            <Col xs={12} className="d-flex justify-content-center">
              <div className="action-buttons d-flex flex-wrap justify-content-center">
                <Button
                  variant={
                    activeTab === "upload" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={() => setActiveTab("upload")}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Upload Image
                </Button>
                <Button
                  variant={
                    activeTab === "shape" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectShape}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M3 3l18 18M10.5 10.5l3 3M5 13l6-6 1.243 1.243M10.984 10.984L16 6l2 2-5.016 5.016M5 19l5-5 5 5"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect Shape
                </Button>
                <Button
                  variant={activeTab === "size" ? "primary" : "outline-primary"}
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectSize}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M21 6H3M21 12H3M21 18H3M8 6v12M16 6v12"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect Size
                </Button>
                <Button
                  variant={
                    activeTab === "color" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectColor}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect Color
                </Button>
                <Button
                  variant={
                    activeTab === "combined" ? "success" : "outline-success"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectAll}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect All
                </Button>
              </div>
            </Col>
          </Row>

          {error && (
            <Alert variant="danger" onClose={() => setError("")} dismissible>
              {error}
            </Alert>
          )}

          <Row className="justify-content-center">
            <Col lg={8} className="mb-4">
              {activeTab === "upload" && (
                <ImageUploader
                  onFileUpload={handleFileUpload}
                  imageUrl={imageUrl}
                />
              )}

              {processedImageUrl &&
                (activeTab === "shape" ||
                  activeTab === "size" ||
                  activeTab === "color" ||
                  activeTab === "combined") && (
                  <ImageDisplay
                    imageUrl={processedImageUrl}
                    title={
                      activeTab === "shape"
                        ? "Shape Detection Result"
                        : activeTab === "size"
                        ? "Size Detection Result"
                        : activeTab === "color"
                        ? "Color Detection Result"
                        : "Combined Detection Result"
                    }
                  />
                )}

              {loading && (
                <div className="text-center py-5">
                  <Spinner animation="border" role="status" variant="primary">
                    <span className="visually-hidden">Loading...</span>
                  </Spinner>
                  <p className="mt-3">Processing image...</p>
                </div>
              )}
            </Col>

            {(activeTab === "shape" && shapes.length > 0) ||
            (activeTab === "size" && measurements.length > 0) ||
            (activeTab === "color" && colors.length > 0) ||
            (activeTab === "combined" && combinedResults.length > 0) ? (
              <Col lg={4}>
                {activeTab === "shape" && shapes.length > 0 && (
                  <ShapeResult shapes={shapes} />
                )}

                {activeTab === "size" && measurements.length > 0 && (
                  <SizeResult measurements={measurements} />
                )}

                {activeTab === "color" && colors.length > 0 && (
                  <ColorResult colors={colors} />
                )}

                {activeTab === "combined" && combinedResults.length > 0 && (
                  <div>
                    <h4 className="mb-3">Combined Detection Results</h4>
                    <div className="results-container">
                      {combinedResults.map((result, index) => (
                        <Card key={index} className="mb-3">
                          <Card.Header className="d-flex justify-content-between align-items-center">
                            <span>{result.shape}</span>
                            <div
                              className="color-swatch"
                              style={{
                                backgroundColor: result.hex,
                                width: "25px",
                                height: "25px",
                                borderRadius: "5px",
                                border: "1px solid #ddd",
                              }}
                            ></div>
                          </Card.Header>
                          <Card.Body>
                            <p>
                              <strong>Color:</strong> {result.color}
                            </p>
                            <p>
                              <strong>Dimensions:</strong> {result.dimensions}
                            </p>
                            <p>
                              <strong>Area:</strong>{" "}
                              {result.area_mm2.toFixed(2)} mm²
                            </p>
                          </Card.Body>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </Col>
            ) : null}
          </Row>
        </Card.Body>
        <Card.Footer className="text-center py-3">
          <p className="text-muted mb-0">
            Upload an image to detect its shape, size, and color.
          </p>
        </Card.Footer>
      </Card>
    </Container>
  );
}

export default App;
