import React, { useRef, useState } from "react";
import { Card, Button, Form } from "react-bootstrap";

const ImageUploader = ({ onFileUpload, imageUrl }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div className="d-flex flex-column align-items-center justify-content-center w-100">
      {/* Display the original image if available */}
      {imageUrl && (
        <div
          className="original-image-container mb-4"
          style={{ maxWidth: "400px" }}
        >
          <h5 className="mb-3 text-center">Original Image</h5>
          <div className="text-center">
            <img
              src={imageUrl}
              alt="Original"
              className="img-fluid rounded"
              style={{
                maxHeight: "300px",
                maxWidth: "100%",
                objectFit: "contain",
                boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              }}
            />
          </div>
        </div>
      )}

      {/* Upload area */}
      <div
        className={`upload-container ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          border: "2px dashed #ccc",
          borderRadius: "5px",
          padding: "40px 20px",
          textAlign: "center",
          cursor: "pointer",
          background: dragActive ? "#f0f8ff" : "#f8f9fa",
          transition: "all 0.3s ease",
          width: "100%",
          maxWidth: "500px",
          margin: "0 auto",
        }}
        onClick={onButtonClick}
      >
        <div className="upload-icon mb-4">
          <svg
            width="50"
            height="50"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            style={{ color: "#0d6efd" }}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <p className="mb-2">
          Drag and drop your image here, or click to select
        </p>
        <small className="text-muted d-block mb-4">
          Supported formats: JPG, JPEG, PNG
        </small>

        <Form.Control
          ref={inputRef}
          type="file"
          style={{ display: "none" }}
          onChange={handleChange}
          accept=".jpg,.jpeg,.png"
        />

        <Button
          variant="primary"
          style={{
            borderRadius: "4px",
            padding: "8px 20px",
            fontSize: "16px",
          }}
          onClick={(e) => {
            e.stopPropagation();
            onButtonClick();
          }}
        >
          Choose Image
        </Button>
      </div>
    </div>
  );
};

export default ImageUploader;
